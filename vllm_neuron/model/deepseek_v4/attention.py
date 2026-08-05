# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 hybrid attention.

Multi-head Latent Attention (MLA) with a hybrid KV layout:

* **Sliding window** — every layer attends over the last ``sliding_window``
  latent KV entries. This stream lives in vLLM's paged KV cache, declared as a
  ``SlidingWindowSpec`` layer with one KV head of width ``head_dim``.
* **Compressed Sparse Attention (CSA)** — layers with ``compress_ratio == 4``
  additionally attend over a compressed KV stream, with the attended slots
  chosen per query by a learned :class:`Indexer`.
* **Heavily Compressed Attention (HCA)** — layers with
  ``compress_ratio == sliding_window`` attend over *all* compressed slots.

Both compressed streams are model-owned buffers rather than vLLM paged blocks:
their time granularity is one slot per ``compress_ratio`` tokens, so the
runner's per-token ``slot_mapping`` does not address them. See the module README
for the consequences (no prefix-cache reuse of compressed state).

Q heads are sharded across TP ranks. The latent KV is a single head shared by
all Q heads, so it is replicated on every rank — as in the reference
implementation, where ``wkv`` is an unsharded :class:`Linear`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.model.deepseek_v4.config import DeepseekV4Config
from vllm_neuron.model.deepseek_v4.layers import (
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    apply_interleaved_rope,
    fake_quant_fp4,
    fake_quant_fp8,
    hadamard_rotate,
    rms_normalize,
)
from vllm_neuron.utils.weight_loader import set_weight_loader

from .weight_loaders import cast_weight_loader, fp8_dequant_weight_loader


# Selected slots processed per attention pass. The gather in
# sparse_latent_attention materializes [tokens, chunk, head_dim] in fp32; with
# head_dim=512 and a 640-wide selection, doing it in one pass is a ~640 MiB
# operator that neuronx-cc expands past its 5M-instruction budget
# (NCC_ELUR015). Chunking trades a longer graph for a bounded operator size.
SPARSE_ATTENTION_CHUNK = 128


def sparse_latent_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
    chunk_size: int = SPARSE_ATTENTION_CHUNK,
) -> torch.Tensor:
    """Gather-based sparse attention over a shared latent KV.

    The latent KV acts as both keys and values (MLA), so a slot contributes the
    same vector to the logit and the weighted sum. ``attn_sink`` adds a learned
    per-head logit that carries no value, letting a head attend to "nothing".

    The selection is processed in chunks with an online (FlashAttention-style)
    softmax, so the gathered KV never exceeds ``[tokens, chunk_size, head_dim]``.
    Results are identical to a single-pass softmax up to fp32 rounding.

    Args:
        q: ``[T, heads, head_dim]`` queries.
        kv: ``[S, head_dim]`` latent KV slots.
        attn_sink: ``[heads]`` sink logits (fp32).
        topk_idxs: ``[T, topk]`` int32 slot indices into ``kv``; ``-1`` marks an
            unused slot and is masked out.
        scale: Softmax scale (``head_dim ** -0.5``).
        chunk_size: Selected slots per pass.

    Returns:
        ``[T, heads, head_dim]`` attention output in ``q``'s dtype.
    """
    tokens, heads, head_dim = q.shape
    topk = topk_idxs.shape[-1]
    q_f32 = q.float()

    # Running online-softmax state: max logit, denominator, weighted sum.
    lowest = torch.finfo(torch.float32).min
    running_max = torch.full(
        (tokens, heads), lowest, device=q.device, dtype=torch.float32
    )
    denominator = torch.zeros(
        tokens, heads, device=q.device, dtype=torch.float32
    )
    accumulator = torch.zeros(
        tokens, heads, head_dim, device=q.device, dtype=torch.float32
    )

    for start in range(0, topk, chunk_size):
        stop = min(start + chunk_size, topk)
        idx = topk_idxs[:, start:stop].long()
        valid = idx >= 0
        width = stop - start

        # [T, width, head_dim] — gather this chunk's latent slots per query.
        gathered = kv.index_select(0, idx.clamp_min(0).reshape(-1)).view(
            tokens, width, head_dim
        )

        logits = torch.einsum("thd,tkd->thk", q_f32, gathered.float())
        # Scale and the mask fill value are materialized as fp32 tensors: XLA
        # types bare Python floats as f64, which neuronx-cc rejects.
        logits = logits * torch.full_like(logits, scale)
        logits = torch.where(
            valid.unsqueeze(1), logits, torch.full_like(logits, lowest)
        )

        # Rescale the running state to the new max, then fold in this chunk.
        chunk_max = torch.maximum(running_max, logits.amax(dim=-1))
        rescale = torch.exp(running_max - chunk_max)
        weights = torch.exp(logits - chunk_max.unsqueeze(-1))

        accumulator = accumulator * rescale.unsqueeze(-1) + torch.einsum(
            "thk,tkd->thd", weights, gathered.float()
        )
        denominator = denominator * rescale + weights.sum(dim=-1)
        running_max = chunk_max

    # The sink is a logit with no value: it only enlarges the denominator, which
    # is how a head attends to "nothing".
    denominator = denominator + torch.exp(
        attn_sink.float().view(1, heads) - running_max
    )
    return (accumulator / denominator.unsqueeze(-1)).to(q.dtype)


class Compressor(nn.Module):
    """Compresses ``compress_ratio`` latent KV entries into one, by gated pooling.

    A learned gate scores each position inside the window (plus a per-offset
    absolute position embedding ``ape``), and the compressed slot is the
    softmax-weighted sum over the window. Ratio-4 compressors use *overlapping*
    windows: the projection is twice as wide, with the first half covering the
    previous window and the second half the current one, which smooths the
    compression boundary.

    Computation runs in fp32 — the checkpoint stores these weights in bf16 but
    the reference upcasts them, and the softmax over a 128-wide window is
    precision-sensitive.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        # Only the ratio-4 (CSA) compressors overlap windows.
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        width = (1 + self.overlap) * head_dim

        self.ape = nn.Parameter(torch.empty(compress_ratio, width, dtype=torch.float32))
        self.wkv = nn.Parameter(
            torch.empty(width, self.hidden_size, dtype=torch.float32)
        )
        self.wgate = nn.Parameter(
            torch.empty(width, self.hidden_size, dtype=torch.float32)
        )
        self.norm = DeepseekV4RMSNorm(head_dim, config.rms_norm_eps)

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        # Compressor weights are unquantized bf16 in the checkpoint and
        # replicated across ranks (the latent KV is not head-sharded).
        for param in (self.wkv, self.wgate):
            set_weight_loader(param, cast_weight_loader(out_dtype=torch.float32))
        set_weight_loader(self.ape, cast_weight_loader(out_dtype=torch.float32))
        set_weight_loader(
            self.norm.weight, cast_weight_loader(out_dtype=torch.float32)
        )

    def _overlap_transform(self, tensor: torch.Tensor, fill: float) -> torch.Tensor:
        """Rearrange ``[N, ratio, 2 * d]`` into ``[N, 2 * ratio, d]``.

        The second half of each window's features stays in place; the first half
        is shifted one window later, so window ``n`` sees window ``n - 1``'s
        overlap features. Window 0 has no predecessor and is filled.

        Assembled with ``cat`` rather than slice assignment into a preallocated
        buffer: the FX pass turns slice assignment into ``slice_scatter``, which
        trips an internal neuronx-cc error (NCC_ILSA902) in this position.
        """
        num_windows, ratio, _ = tensor.shape
        d = self.head_dim

        current = tensor[:, :, d:]  # [N, ratio, d] — this window's features
        shifted = tensor[:, :, :d]  # [N, ratio, d] — to be shifted one window on
        first = shifted.new_full((1, ratio, d), fill)
        previous = torch.cat([first, shifted[:-1]], dim=0)

        return torch.cat([previous, current], dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Compress a whole prefill sequence.

        Args:
            hidden_states: ``[T, H]`` layer input (pre-attention, post-HC).
            cos: ``[T // ratio, rope_dim // 2]`` tables for the compressed
                positions — i.e. sampled every ``ratio`` tokens.
            sin: Matching sine tables.

        Returns:
            ``[T // ratio, head_dim]`` compressed latent KV. Trailing tokens
            that do not fill a whole window are dropped, matching the reference.
        """
        ratio = self.compress_ratio
        tokens = hidden_states.shape[0]
        num_windows = tokens // ratio

        if num_windows == 0:
            # Prompt shorter than one window: pad up to a single window so the
            # output is one (causally masked) slot rather than a zero-length
            # tensor, which the Neuron compiler cannot lower.
            hidden_states = F.pad(hidden_states, (0, 0, 0, ratio - tokens))
            tokens = ratio
            num_windows = 1

        x = hidden_states.float()
        cutoff = num_windows * ratio
        kv = F.linear(x[:cutoff], self.wkv).unflatten(0, (num_windows, ratio))
        score = F.linear(x[:cutoff], self.wgate).unflatten(0, (num_windows, ratio))
        score = score + self.ape

        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            # Padding scores must vanish under the softmax below. finfo.min
            # rather than -inf: XLA types a bare float literal as f64, which
            # neuronx-cc rejects, and finfo.min underflows to 0 all the same.
            score = self._overlap_transform(score, torch.finfo(score.dtype).min)

        compressed = (kv * score.softmax(dim=1)).sum(dim=1)
        compressed = self.norm(compressed.to(hidden_states.dtype))

        rope_dim = self.rope_head_dim
        rotated = apply_interleaved_rope(compressed[:, -rope_dim:], cos, sin)
        compressed = torch.cat([compressed[:, :-rope_dim], rotated], dim=-1)

        # QAT simulation, matching the reference implementation.
        if self.rotate:
            # Indexer stream: Hadamard-rotate the full vector (RoPE dims
            # included) to spread magnitude, then simulate the FP4 grid.
            compressed = hadamard_rotate(compressed)
            compressed = fake_quant_fp4(compressed, block_size=32)
        else:
            # Attention stream: FP8 grid on the content dims only; RoPE dims
            # stay bf16 to preserve positional precision.
            content = fake_quant_fp8(compressed[:, :-rope_dim], block_size=64)
            compressed = torch.cat([content, compressed[:, -rope_dim:]], dim=-1)
        return compressed


class Indexer(nn.Module):
    """Selects which compressed slots a CSA layer attends to.

    Scores every compressed slot against a low-rank projection of the query with
    its own head set, then keeps the top ``index_topk``. The indexer maintains a
    *separate* compressed stream (its own :class:`Compressor`, at
    ``index_head_dim`` width) used only for scoring — the attended values come
    from the attention layer's own compressed stream.
    """

    def __init__(self, config: DeepseekV4Config, compress_ratio: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = compress_ratio
        self.dtype = config.torch_dtype
        self.softmax_scale = self.head_dim**-0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if self.num_heads % self.world_size:
            raise ValueError(
                f"index_n_heads ({self.num_heads}) must be divisible by the TP "
                f"degree ({self.world_size})"
            )
        self.num_heads_per_rank = self.num_heads // self.world_size

        self.wq_b = nn.Parameter(
            torch.empty(
                self.num_heads_per_rank * self.head_dim,
                self.q_lora_rank,
                dtype=self.dtype,
            )
        )
        self.weights_proj = nn.Parameter(
            torch.empty(self.num_heads_per_rank, self.hidden_size, dtype=self.dtype)
        )
        self.compressor = Compressor(
            config, compress_ratio, self.head_dim, rotate=True
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        head_shard = self.num_heads_per_rank * self.head_dim
        set_weight_loader(
            self.wq_b,
            fp8_dequant_weight_loader(
                shard_dim=0,
                shard_size=head_shard,
                num_shards=self.world_size,
                out_dtype=self.dtype,
            ),
        )
        set_weight_loader(
            self.weights_proj,
            cast_weight_loader(
                out_dtype=self.dtype,
                shard_dim=0,
                shard_size=self.num_heads_per_rank,
                num_shards=self.world_size,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        compressed_kv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_slot_limit: torch.Tensor,
        slot_offset: int,
    ) -> torch.Tensor:
        """Score compressed slots and return the selected indices.

        Args:
            hidden_states: ``[T, H]`` layer input.
            q_lora: ``[T, q_lora_rank]`` shared low-rank query projection.
            compressed_kv: ``[S_c, index_head_dim]`` the indexer's own
                compressed stream, used for scoring only.
            cos: ``[T, rope_dim // 2]`` per-query rotary tables.
            sin: Matching sine tables.
            causal_slot_limit: ``[T]`` exclusive upper bound on the compressed
                slot each query may see.
            slot_offset: Value added to the returned indices so they address the
                caller's concatenated ``[window | compressed]`` KV buffer.

        Returns:
            ``[T, min(index_topk, S_c)]`` int32 indices, ``-1`` where masked.
        """
        tokens = hidden_states.shape[0]
        num_slots = compressed_kv.shape[0]
        # Callers pass the addressable slots only — the scratch slot the decode
        # append path writes to must be excluded, or a stream sized exactly
        # index_topk would appear one slot too long and take the top-k path.

        if num_slots <= self.index_topk:
            # Every slot fits in the top-k budget, so selection is a no-op:
            # return all slots, causally masked. Skipping the top-k here is not
            # just an optimization — XLA lowers a small top-k to `sort`, which
            # the Neuron compiler rejects on trn2.
            return _hca_indices_prefill(causal_slot_limit, num_slots, slot_offset)

        q = F.linear(q_lora, self.wq_b)
        q = q.view(tokens, self.num_heads_per_rank, self.head_dim)
        rope_dim = self.rope_head_dim
        rotated = apply_interleaved_rope(q[..., -rope_dim:], cos, sin)
        q = torch.cat([q[..., :-rope_dim], rotated], dim=-1)
        # Indexer scores are computed on the FP4 grid (see Compressor.rotate).
        q = fake_quant_fp4(hadamard_rotate(q), block_size=32)

        # Per-head weights, scaled so the summed score is head-count invariant.
        weights = F.linear(hidden_states, self.weights_proj) * (
            self.softmax_scale * self.num_heads**-0.5
        )

        scores = torch.einsum("thd,sd->ths", q.float(), compressed_kv.float())
        scores = (scores.relu() * weights.float().unsqueeze(-1)).sum(dim=1)

        # Heads are TP-sharded, so the score is a partial sum.
        if self.world_size > 1:
            scores = self.tp_group.all_reduce(scores)

        slot_ids = torch.arange(num_slots, device=scores.device)
        allowed = slot_ids.unsqueeze(0) < causal_slot_limit.unsqueeze(1)
        neg_inf = torch.full_like(scores, torch.finfo(scores.dtype).min)
        scores = torch.where(allowed, scores, neg_inf)

        topk_idxs = scores.topk(self.index_topk, dim=-1)[1]
        # A query whose causal limit is below index_topk gets padding slots back
        # from topk(); mask those to -1 so the attention drops them.
        selected_allowed = torch.gather(allowed, 1, topk_idxs)
        return torch.where(
            selected_allowed, topk_idxs + slot_offset, torch.full_like(topk_idxs, -1)
        ).to(torch.int32)


class DeepseekV4Attention(nn.Module):
    """MLA with a sliding window plus an optional compressed KV stream.

    Projection chain, all TP-sharded on the Q-head axis:

    ``wq_a`` (low rank) → ``q_norm`` → ``wq_b`` (per-head) → RMS-normalize →
    RoPE, and ``wkv`` → ``kv_norm`` → RoPE for the single shared latent KV head.
    Output goes through a grouped low-rank projection ``wo_a`` → ``wo_b``.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.qk_nope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.sliding_window = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.softmax_scale = self.head_dim**-0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_attention_heads = config.num_attention_heads
        if self.num_attention_heads % self.world_size:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by the TP degree ({self.world_size})"
            )
        self.num_heads_per_rank = self.num_attention_heads // self.world_size

        # MLA: one latent KV head, replicated on every rank (not sharded).
        self.num_key_value_heads_per_rank = 1

        # Grouped output projection. Each group covers a contiguous span of
        # attention output features — heads_per_group = num_heads / o_groups.
        self.num_groups = config.o_groups
        heads_per_group, remainder = divmod(self.num_attention_heads, self.num_groups)
        if remainder:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by o_groups ({self.num_groups})"
            )

        # Two regimes, depending on whether a rank's heads cover whole groups:
        #
        # * TP <= o_groups: each rank owns whole groups. wo_a is group-sharded,
        #   and each group's projection consumes all of that group's features.
        # * TP > o_groups: a group's heads are split across ranks. Every rank
        #   then holds all groups it touches, but only the wo_a *input columns*
        #   for its own heads, producing a partial sum that the all-reduce after
        #   wo_b completes.
        if self.num_heads_per_rank >= heads_per_group:
            self.num_groups_per_rank = self.num_heads_per_rank // heads_per_group
            self.group_is_split = False
            self.heads_per_local_group = heads_per_group
        else:
            self.num_groups_per_rank = 1
            self.group_is_split = True
            self.heads_per_local_group = self.num_heads_per_rank
        self.heads_per_group = heads_per_group

        # ── Parameters ────────────────────────────────────────────────────
        self.attn_sink = nn.Parameter(
            torch.empty(self.num_heads_per_rank, dtype=torch.float32)
        )
        self.wq_a = nn.Parameter(
            torch.empty(self.q_lora_rank, self.hidden_size, dtype=self.dtype)
        )
        self.q_norm = DeepseekV4RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Parameter(
            torch.empty(
                self.num_heads_per_rank * self.head_dim,
                self.q_lora_rank,
                dtype=self.dtype,
            )
        )
        self.wkv = nn.Parameter(
            torch.empty(self.head_dim, self.hidden_size, dtype=self.dtype)
        )
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, self.eps)
        # Grouped low-rank output. Per local group the projection consumes this
        # rank's slice of that group's attention features
        # (heads_per_local_group * head_dim) and emits o_lora_rank.
        self.wo_a = nn.Parameter(
            torch.empty(
                self.num_groups_per_rank,
                self.o_lora_rank,
                self.heads_per_local_group * self.head_dim,
                dtype=self.dtype,
            )
        )
        self.wo_b = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.num_groups_per_rank * self.o_lora_rank,
                dtype=self.dtype,
            )
        )

        # ── Compressed KV stream ──────────────────────────────────────────
        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
            # Only CSA (ratio 4) layers select slots; HCA layers take them all.
            self.indexer = (
                Indexer(config, self.compress_ratio)
                if self.compress_ratio == 4
                else None
            )
        else:
            self.compressor = None
            self.indexer = None

        # ── Rotary tables ─────────────────────────────────────────────────
        rope_scaling = config.rope_scaling or {}
        if self.compress_ratio:
            # Compressed streams use the long base with YaRN enabled.
            self.rotary_emb = DeepseekV4RotaryEmbedding(
                self.rope_head_dim,
                config.max_position_embeddings,
                config.compress_rope_theta,
                rope_scaling.get("original_max_position_embeddings", 0),
                rope_scaling.get("factor", 1.0),
                rope_scaling.get("beta_fast", 32),
                rope_scaling.get("beta_slow", 1),
            )
        else:
            # Sliding-window-only layers disable YaRN and use the base theta.
            self.rotary_emb = DeepseekV4RotaryEmbedding(
                self.rope_head_dim,
                config.max_position_embeddings,
                config.rope_theta,
            )

        # Paged latent KV cache, bound by the runner via bind_kv_cache().
        self.k_cache = None
        self.v_cache = None

        # Compressed KV streams. Allocated alongside the paged cache in
        # bind_kv_cache() (see compressed_state.py for why they are
        # model-owned) and, like k_cache/v_cache, set as plain attributes so
        # they are already real device tensors by the time the graph is traced.
        self.compressed_kv = None
        self.compressed_index_kv = None
        # Rolling window of the most recent hidden states, so decode can close
        # a compression window without re-reading the prompt. Shaped
        # [compress_ratio, hidden_size]; slot p % compress_ratio holds position
        # p. Allocated in bind_kv_cache() alongside the streams.
        self.compress_window_hidden = None
        # Number of compressed slots currently valid, as a device scalar so the
        # append path stays inside the traced graph.
        self.compressed_length = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        head_shard = self.num_heads_per_rank * self.head_dim

        # wq_a / wkv: replicated (they feed the shared low-rank / latent path).
        set_weight_loader(
            self.wq_a, fp8_dequant_weight_loader(out_dtype=self.dtype)
        )
        set_weight_loader(
            self.wkv, fp8_dequant_weight_loader(out_dtype=self.dtype)
        )
        # wq_b: sharded on the Q-head axis.
        set_weight_loader(
            self.wq_b,
            fp8_dequant_weight_loader(
                shard_dim=0,
                shard_size=head_shard,
                num_shards=self.world_size,
                out_dtype=self.dtype,
            ),
        )
        # wo_a: checkpoint stores [o_groups * o_lora_rank, group_features],
        # reshaped to [groups_per_rank, o_lora_rank, local_group_features].
        set_weight_loader(
            self.wo_a,
            _grouped_wo_a_loader(
                num_groups=self.num_groups,
                num_groups_per_rank=self.num_groups_per_rank,
                o_lora_rank=self.o_lora_rank,
                head_dim=self.head_dim,
                heads_per_group=self.heads_per_group,
                heads_per_local_group=self.heads_per_local_group,
                num_heads_per_rank=self.num_heads_per_rank,
                group_is_split=self.group_is_split,
                out_dtype=self.dtype,
            ),
        )
        # wo_b: row-parallel over the group axis. When a group is split across
        # ranks every one of them holds the same wo_b columns and contributes a
        # partial sum, so the all-reduce after wo_b completes the group.
        if self.group_is_split:
            wo_b_loader = _split_group_wo_b_loader(
                o_lora_rank=self.o_lora_rank,
                heads_per_group=self.heads_per_group,
                num_heads_per_rank=self.num_heads_per_rank,
                out_dtype=self.dtype,
            )
        else:
            wo_b_loader = fp8_dequant_weight_loader(
                shard_dim=1,
                shard_size=self.num_groups_per_rank * self.o_lora_rank,
                num_shards=self.world_size,
                out_dtype=self.dtype,
            )
        set_weight_loader(self.wo_b, wo_b_loader)
        # attn_sink stays fp32: it is a softmax logit, not a matmul input.
        set_weight_loader(
            self.attn_sink,
            cast_weight_loader(
                out_dtype=torch.float32,
                shard_dim=0,
                shard_size=self.num_heads_per_rank,
                num_shards=self.world_size,
            ),
        )
        for norm in (self.q_norm, self.kv_norm):
            set_weight_loader(
                norm.weight, cast_weight_loader(out_dtype=torch.float32)
            )

    # ── Projections ──────────────────────────────────────────────────────

    def project_query(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and rotate queries.

        Returns ``(q, q_lora)``: ``q`` is ``[T, heads, head_dim]`` ready for
        attention; ``q_lora`` is the ``[T, q_lora_rank]`` intermediate, which the
        indexer reuses.
        """
        tokens = hidden_states.shape[0]
        q_lora = self.q_norm(F.linear(hidden_states, self.wq_a))
        q = F.linear(q_lora, self.wq_b).view(
            tokens, self.num_heads_per_rank, self.head_dim
        )
        # Per-head RMS normalization (no learned weight) before RoPE.
        q = rms_normalize(q.float(), self.eps).to(hidden_states.dtype)
        rope_dim = self.rope_head_dim
        rotated = apply_interleaved_rope(q[..., -rope_dim:], cos, sin)
        return torch.cat([q[..., :-rope_dim], rotated], dim=-1), q_lora

    def project_latent_kv(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Project and rotate the shared latent KV. Returns ``[T, head_dim]``.

        The content dims get the QAT FP8 grid the model was trained with; the
        RoPE dims stay bf16.
        """
        kv = self.kv_norm(F.linear(hidden_states, self.wkv))
        rope_dim = self.rope_head_dim
        rotated = apply_interleaved_rope(kv[:, -rope_dim:], cos, sin)
        content = fake_quant_fp8(kv[:, :-rope_dim], block_size=64)
        return torch.cat([content, rotated], dim=-1)

    def project_output(
        self, attn_out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """De-rotate, then apply the grouped low-rank output projection.

        The rotation applied to Q/KV leaves the attention output in rotated
        space; the reference de-rotates with the conjugate before projecting.
        """
        tokens = attn_out.shape[0]
        rope_dim = self.rope_head_dim
        derotated = apply_interleaved_rope(
            attn_out[..., -rope_dim:], cos, sin, inverse=True
        )
        attn_out = torch.cat([attn_out[..., :-rope_dim], derotated], dim=-1)

        # Split the head axis into output-projection groups.
        grouped = attn_out.reshape(tokens, self.num_groups_per_rank, -1)
        projected = torch.einsum("tgd,grd->tgr", grouped, self.wo_a)
        out = F.linear(projected.flatten(1), self.wo_b)

        # wo_b is row-parallel: each rank holds a partial sum, and the
        # all-reduce completes it. This is exact even when a group is split
        # across ranks: those ranks share identical wo_b columns but their wo_a
        # slices are disjoint, so their o_lora_rank vectors sum to the group's
        # full vector and wo_b's linearity carries that through.
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out


    # ── KV cache ─────────────────────────────────────────────────────────

    @property
    def layer_name(self) -> str:
        return f"layers.{self.layer_idx}.self_attn"

    def write_latent_kv(
        self,
        latent_kv: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Write the latent KV into the paged cache at ``slot_mapping``.

        MLA has a single KV head whose value doubles as key and value, so the
        same tensor goes into both halves of the cache.

        Padding tokens carry ``slot_mapping == -1``. Rather than masking them
        out, the slot index is clamped into range so those writes land on the
        last slot of the cache, which the block manager never hands out while a
        request is live. Clamping instead of ``torch.where`` is deliberate: a
        select feeding an in-place index_put_ trips an internal neuronx-cc
        error (NCC_ILSA902) during legalization.
        """
        num_blocks = self.k_cache.shape[0]
        max_slot = num_blocks * block_size
        safe_slots = slot_mapping.clamp(0, max_slot - 1)
        block_idx = (safe_slots // block_size).long()
        pos_idx = (safe_slots % block_size).long()

        values = latent_kv.to(self.k_cache.dtype)
        head_idx = torch.zeros_like(block_idx)
        self.k_cache.index_put_((block_idx, head_idx, pos_idx), values)
        self.v_cache.index_put_((block_idx, head_idx, pos_idx), values)

    def read_latent_kv(
        self, block_table: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Gather a request's cached latent KV into ``[num_slots, head_dim]``.

        ``num_slots`` is ``max_blocks_per_seq * block_size``; entries beyond the
        request's length hold stale data and must be excluded by the caller's
        index mask.
        """
        blocks = block_table.long().reshape(-1)
        # [num_blocks_sel, block_size, head_dim] — head dim is 1 for MLA.
        gathered = self.k_cache.index_select(0, blocks)[:, 0]
        return gathered.reshape(-1, self.head_dim)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
    ) -> torch.Tensor:
        """Run attention for this layer.

        Args:
            hidden_states: ``[T, H]`` post-hyper-connection layer input.
            positions: ``[T]`` int32 absolute positions.
            attn_metadata: Per-layer runner metadata (see the onboarding guide).

        Returns:
            ``[T, H]`` attention output.
        """
        meta = attn_metadata[self.layer_name]
        max_query_len = meta["max_query_len"]
        decode_token_threshold = meta["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(hidden_states, positions, meta)
        return self.forward_prefill(hidden_states, positions, meta)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        meta: dict,
    ) -> torch.Tensor:
        """Whole-sequence prefill for a single request.

        Builds the compressed stream from scratch (the compressors are causal
        pooling operators over the prompt), concatenates it after the
        sliding-window slots, and runs one sparse attention over the union.
        """
        tokens = hidden_states.shape[0]
        block_size = meta["block_size"]
        slot_mapping = meta["slot_mapping"]

        cos, sin = self.rotary_emb(positions, dtype=hidden_states.dtype)
        q, q_lora = self.project_query(hidden_states, cos, sin)
        latent_kv = self.project_latent_kv(hidden_states, cos, sin)

        self.write_latent_kv(latent_kv, slot_mapping, block_size)

        window = self.sliding_window
        # Sliding-window slots: index directly into the local latent_kv, which
        # holds the whole prompt during prefill.
        window_idxs = _window_indices_prefill(tokens, window, latent_kv.device)

        if self.compress_ratio:
            ratio = self.compress_ratio
            num_slots = tokens // ratio

            # Build the compressed streams and write them into the persistent
            # buffers, then attend over the *buffers* rather than the locals.
            # The buffers have a fixed, config-derived length, so the attention
            # shapes stay identical across prefill buckets — including the case
            # where the prompt is shorter than one compression window and
            # num_slots is 0, which would otherwise create a zero-length tensor
            # the compiler cannot lower.
            compressed_kv = self.compressor(
                hidden_states,
                *self._compressed_rope(num_slots, ratio, hidden_states),
            )
            index_kv = None
            if self.indexer is not None:
                index_kv = self.indexer.compressor(
                    hidden_states,
                    *self._compressed_rope(num_slots, ratio, hidden_states),
                )
            self._store_compressed(compressed_kv, index_kv, hidden_states, tokens)

            capacity = self.compressed_capacity
            offset = latent_kv.shape[0]
            # A query at position t may see compressed slot s only once the
            # whole window it summarizes is in the past: s < (t + 1) // ratio.
            causal_limit = (
                torch.arange(1, tokens + 1, device=latent_kv.device) // ratio
            ).clamp(max=capacity)

            if self.indexer is not None:
                compressed_idxs = self.indexer(
                    hidden_states,
                    q_lora,
                    self.compressed_index_kv[:capacity],
                    cos,
                    sin,
                    causal_limit,
                    offset,
                )
            else:
                compressed_idxs = _hca_indices_prefill(
                    causal_limit, capacity, offset
                )

            kv = torch.cat([latent_kv, self.compressed_kv[:capacity]], dim=0)
            topk_idxs = torch.cat([window_idxs, compressed_idxs], dim=-1)
        else:
            kv = latent_kv
            topk_idxs = window_idxs

        attn_out = sparse_latent_attention(
            q, kv, self.attn_sink, topk_idxs, self.softmax_scale
        )
        return self.project_output(attn_out, cos, sin)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        meta: dict,
    ) -> torch.Tensor:
        """Single-token decode, reading the sliding window from the paged cache."""
        tokens = hidden_states.shape[0]
        block_size = meta["block_size"]
        slot_mapping = meta["slot_mapping"]
        block_table = meta["block_table_tensor"]

        cos, sin = self.rotary_emb(positions, dtype=hidden_states.dtype)
        q, q_lora = self.project_query(hidden_states, cos, sin)
        latent_kv = self.project_latent_kv(hidden_states, cos, sin)

        self.write_latent_kv(latent_kv, slot_mapping, block_size)

        # Read back the cached window. block_table is [num_reqs, blocks]; decode
        # runs one token per request, so row r serves query r.
        cached = self.read_latent_kv(block_table, block_size)
        slots_per_req = block_table.shape[1] * block_size
        window_idxs = _window_indices_decode(
            positions, self.sliding_window, slots_per_req, tokens, cached.device
        )

        if self.compress_ratio:
            # Advance the compressed stream when this step closes a window.
            self._append_compressed(hidden_states, positions)

            capacity = self.compressed_capacity
            offset = cached.shape[0]
            causal_limit = (
                (positions.long() + 1) // self.compress_ratio
            ).clamp(max=capacity)

            if self.indexer is not None:
                compressed_idxs = self.indexer(
                    hidden_states,
                    q_lora,
                    self.compressed_index_kv[:capacity],
                    cos,
                    sin,
                    causal_limit,
                    offset,
                )
            else:
                compressed_idxs = _hca_indices_prefill(
                    causal_limit, capacity, offset
                )
            kv = torch.cat([cached, self.compressed_kv[:capacity]], dim=0)
            topk_idxs = torch.cat([window_idxs, compressed_idxs], dim=-1)
        else:
            kv = cached
            topk_idxs = window_idxs

        attn_out = sparse_latent_attention(
            q, kv, self.attn_sink, topk_idxs, self.softmax_scale
        )
        return self.project_output(attn_out, cos, sin)

    # ── Compressed stream maintenance ────────────────────────────────────

    @property
    def compressed_capacity(self) -> int:
        """Addressable compressed slots, excluding the trailing scratch slot."""
        return self.compressed_kv.shape[0] - 1

    def _compressed_rope(
        self, num_slots: int, ratio: int, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotary tables for the compressed slot positions.

        Slot ``s`` summarizes positions ``[s * ratio, (s + 1) * ratio)`` and is
        rotated at the window's first position. At least one entry is always
        produced: a prompt shorter than one window yields no real slots, but a
        zero-length table would make the compressor emit a zero-length tensor
        that the Neuron compiler cannot lower. The extra slot is masked out by
        the causal limit, so it never contributes to attention.
        """
        count = max(num_slots, 1)
        positions = torch.arange(
            0, count * ratio, ratio, device=reference.device
        )
        return self.rotary_emb(positions, dtype=reference.dtype)

    def _store_compressed(
        self,
        compressed_kv: torch.Tensor,
        index_kv: torch.Tensor | None,
        hidden_states: torch.Tensor,
        tokens: int,
    ) -> None:
        """Seed the persistent buffers from a freshly built prefill stream.

        Also primes the rolling hidden-state window with the prompt's trailing
        partial window, so the first decode step that closes a window sees the
        same inputs the reference implementation would.
        """
        if self.compressed_kv is None:
            return
        ratio = self.compress_ratio
        capacity = self.compressed_capacity
        keep = min(compressed_kv.shape[0], capacity)

        # Each buffer is rebuilt whole and written with a single copy_.
        # Slice assignment would work numerically, but the FX pass turns it into
        # slice_scatter, and a conditional in-place update of a buffer trips an
        # internal neuronx-cc error (NCC_ILSA902) during legalization. Building
        # the full tensor with cat keeps the write a flat copy.
        self.compressed_kv.copy_(
            _pad_to_rows(compressed_kv[:keep], self.compressed_kv)
        )
        if index_kv is not None and self.compressed_index_kv is not None:
            self.compressed_index_kv.copy_(
                _pad_to_rows(index_kv[:keep], self.compressed_index_kv)
            )
        self.compressed_length.copy_(
            torch.full_like(self.compressed_length, keep)
        )

        # Carry over the tail positions that did not fill a whole window, so the
        # first decode step that closes a window sees the same inputs as the
        # reference.
        remainder = tokens % ratio
        tail = (
            hidden_states[tokens - remainder : tokens]
            if remainder
            else hidden_states[:0]
        )
        self.compress_window_hidden.copy_(
            _pad_to_rows(tail, self.compress_window_hidden)
        )

    def _append_compressed(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """Close a compression window if this decode step completes one.

        Decode advances one position at a time, so the hidden state is stashed in
        a rolling ``[compress_ratio, hidden_size]`` buffer. Every
        ``compress_ratio``-th position the buffer holds a full window and is run
        through the compressors to append one slot to each stream.

        The compressors run on every step regardless — the traced graph must
        have a single shape-static path. Steps that do not close a window are
        neutralized by redirecting the write to a scratch slot at the end of the
        buffer, which the causal index mask never selects. Steering the *index*
        rather than blending the *value* keeps the write a plain scatter; a
        conditional read-modify-write of the buffer hits an internal
        neuronx-cc error (NCC_ILSA902) during legalization.
        """
        if self.compressed_kv is None:
            return
        ratio = self.compress_ratio

        # Stash this step's hidden state at its slot in the rolling window.
        # Decode is one token per request; position 0 is this request's token.
        slot = (positions[0].long() % ratio).clamp(0, ratio - 1)
        self.compress_window_hidden.index_copy_(
            0,
            slot.unsqueeze(0),
            hidden_states[:1].to(self.compress_window_hidden.dtype),
        )

        window = self.compress_window_hidden
        compressed_positions = (positions[0].long() + 1 - ratio).clamp_min(0)
        c_cos, c_sin = self.rotary_emb(
            compressed_positions.unsqueeze(0), dtype=window.dtype
        )
        new_slot = self.compressor(window, c_cos, c_sin)

        # A window closes when this position is the last of its group. When it
        # does not, aim the write at the scratch slot (the last one, which
        # _allocate_compressed_states reserves beyond the addressable capacity).
        closes = ((positions[0].long() + 1) % ratio) == 0
        scratch = self.compressed_kv.shape[0] - 1
        length = self.compressed_length.long().clamp(0, scratch - 1)
        # Arithmetic blend rather than torch.where: a select feeding an in-place
        # index_copy_ trips an internal neuronx-cc error (NCC_ILSA902).
        closes_i = closes.to(length.dtype)
        write_at = closes_i * length + (1 - closes_i) * scratch

        self.compressed_kv.index_copy_(
            0, write_at, new_slot[:1].to(self.compressed_kv.dtype)
        )
        if self.indexer is not None and self.compressed_index_kv is not None:
            new_index_slot = self.indexer.compressor(window, c_cos, c_sin)
            self.compressed_index_kv.index_copy_(
                0, write_at, new_index_slot[:1].to(self.compressed_index_kv.dtype)
            )

        advanced = (
            self.compressed_length + closes.to(self.compressed_length.dtype)
        ).clamp(max=scratch)
        self.compressed_length.copy_(advanced)


def _pad_to_rows(rows: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Zero-pad ``rows`` along dim 0 to match ``like``'s row count and dtype.

    Returns a freshly built tensor so callers can write a buffer with a single
    flat ``copy_`` instead of a slice assignment.
    """
    rows = rows.to(like.dtype)
    missing = like.shape[0] - rows.shape[0]
    if missing <= 0:
        return rows[: like.shape[0]]
    padding = rows.new_zeros((missing, *like.shape[1:]))
    return torch.cat([rows, padding], dim=0)


def _window_indices_prefill(
    tokens: int, window: int, device: torch.device
) -> torch.Tensor:
    """Causal sliding-window indices into a ``[tokens, head_dim]`` local KV.

    Row ``t`` lists the up to ``window`` positions ``(t - window, t]``, padded
    with ``-1`` at the start of the sequence.
    """
    query_pos = torch.arange(tokens, device=device).unsqueeze(1)
    offsets = torch.arange(window, device=device).unsqueeze(0)
    idxs = query_pos - window + 1 + offsets
    return torch.where(idxs < 0, torch.full_like(idxs, -1), idxs).to(torch.int32)


def _window_indices_decode(
    positions: torch.Tensor,
    window: int,
    slots_per_req: int,
    tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Sliding-window indices into the gathered paged cache for decode.

    The gathered cache is ``[num_reqs * slots_per_req, head_dim]`` laid out in
    block-table order, so request ``r``'s absolute position ``p`` lives at flat
    index ``r * slots_per_req + p``.
    """
    request_ids = torch.arange(tokens, device=device).unsqueeze(1)
    pos = positions.long().unsqueeze(1)
    offsets = torch.arange(window, device=device).unsqueeze(0)
    absolute = pos - window + 1 + offsets
    flat = request_ids * slots_per_req + absolute
    valid = (absolute >= 0) & (absolute < slots_per_req)
    return torch.where(valid, flat, torch.full_like(flat, -1)).to(torch.int32)


def _hca_indices_prefill(
    causal_limit: torch.Tensor, num_slots: int, offset: int
) -> torch.Tensor:
    """All-compressed-slot indices for HCA layers, causally masked.

    HCA layers skip the indexer and attend over every compressed slot the query
    is allowed to see.
    """
    slot_ids = torch.arange(num_slots, device=causal_limit.device).unsqueeze(0)
    allowed = slot_ids < causal_limit.unsqueeze(1)
    return torch.where(
        allowed, slot_ids + offset, torch.full_like(slot_ids, -1)
    ).to(torch.int32)


_FP8_SCALE_BLOCK = 128


def _dequant_fp8_rows_cols(
    weight_slice,
    scale_slice,
    row_range: tuple[int, int],
    col_range: tuple[int, int] | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an FP8 sub-block, reading only the slice that is needed.

    Row and column ranges must be multiples of the 128 scale block; every
    shipped DeepSeek-V4 config satisfies this (``o_lora_rank`` and ``head_dim``
    are both multiples of 128), so a misaligned request is a programming error
    rather than a case to fall back on.
    """
    from .weight_loaders import dequant_fp8_blockwise

    block = _FP8_SCALE_BLOCK
    row_start, row_stop = row_range
    if row_start % block or (row_stop - row_start) % block:
        raise ValueError(
            f"wo_a/wo_b row range {row_range} is not a multiple of the {block} "
            "FP8 scale block"
        )

    if col_range is None:
        weight = weight_slice[row_start:row_stop]
        scale = scale_slice[row_start // block : row_stop // block]
    else:
        col_start, col_stop = col_range
        if col_start % block or (col_stop - col_start) % block:
            raise ValueError(
                f"wo_a/wo_b column range {col_range} is not a multiple of the "
                f"{block} FP8 scale block"
            )
        weight = weight_slice[row_start:row_stop, col_start:col_stop]
        scale = scale_slice[
            row_start // block : row_stop // block,
            col_start // block : col_stop // block,
        ]
    return dequant_fp8_blockwise(weight, scale, out_dtype=out_dtype)


def _grouped_wo_a_loader(
    num_groups: int,
    num_groups_per_rank: int,
    o_lora_rank: int,
    head_dim: int,
    heads_per_group: int,
    heads_per_local_group: int,
    num_heads_per_rank: int,
    group_is_split: bool,
    out_dtype: torch.dtype,
):
    """Loader for ``wo_a``, producing ``[groups_per_rank, o_lora_rank, in]``.

    The checkpoint stores ``[o_groups * o_lora_rank, group_features]`` — the
    group axis folded into dim 0, and each row spanning one group's whole slice
    of the attention output.

    Two sharding regimes, matching ``DeepseekV4Attention.__init__``:

    * **Whole groups per rank** (TP <= o_groups): take this rank's contiguous
      run of groups, all input columns.
    * **Split group** (TP > o_groups): every rank takes the single group its
      heads fall in, but only the input columns for its own heads. The result is
      a partial sum over the group, completed by the all-reduce after ``wo_b``.
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) != 2:
            raise ValueError(
                f"wo_a loader expects [weight, scale] slices, got {len(slices)}"
            )
        weight_slice, scale_slice = slices

        if not group_is_split:
            rows = num_groups_per_rank * o_lora_rank
            row_start = rank * rows
            dequantized = _dequant_fp8_rows_cols(
                weight_slice, scale_slice, (row_start, row_start + rows), None, out_dtype
            )
            return dequantized.view(
                num_groups_per_rank, o_lora_rank, -1
            ).contiguous()

        # This rank's global head range, and the group those heads live in.
        first_head = rank * num_heads_per_rank
        group = first_head // heads_per_group
        head_within_group = first_head % heads_per_group

        row_start = group * o_lora_rank
        col_start = head_within_group * head_dim
        col_stop = col_start + heads_per_local_group * head_dim

        dequantized = _dequant_fp8_rows_cols(
            weight_slice,
            scale_slice,
            (row_start, row_start + o_lora_rank),
            (col_start, col_stop),
            out_dtype,
        )
        return dequantized.view(1, o_lora_rank, -1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _split_group_wo_b_loader(
    o_lora_rank: int,
    heads_per_group: int,
    num_heads_per_rank: int,
    out_dtype: torch.dtype,
):
    """Loader for ``wo_b`` when an output group is split across ranks.

    Every rank sharing a group holds that group's full ``o_lora_rank`` columns
    of ``wo_b``. Each contributes a partial sum, because its ``wo_a`` covers only
    some of the group's heads; the all-reduce after ``wo_b`` adds them.

    Replicating the columns is exact rather than double-counting: the per-rank
    ``o_lora_rank`` vectors sum to the group's full vector, and ``wo_b`` is
    linear, so applying it before the reduction gives the same result as after.
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) != 2:
            raise ValueError(
                f"wo_b loader expects [weight, scale] slices, got {len(slices)}"
            )
        weight_slice, scale_slice = slices
        group = (rank * num_heads_per_rank) // heads_per_group
        col_start = group * o_lora_rank
        rows = weight_slice.get_shape()[0]
        return _dequant_fp8_rows_cols(
            weight_slice,
            scale_slice,
            (0, rows),
            (col_start, col_start + o_lora_rank),
            out_dtype,
        ).contiguous()

    return SafetensorsWeightLoader(transform=transform)
