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


def sparse_latent_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Gather-based sparse attention over a shared latent KV.

    The latent KV acts as both keys and values (MLA), so a slot contributes the
    same vector to the logit and the weighted sum. ``attn_sink`` adds a learned
    per-head logit that carries no value, letting a head attend to "nothing".

    Args:
        q: ``[T, heads, head_dim]`` queries.
        kv: ``[S, head_dim]`` latent KV slots.
        attn_sink: ``[heads]`` sink logits (fp32).
        topk_idxs: ``[T, topk]`` int32 slot indices into ``kv``; ``-1`` marks an
            unused slot and is masked out.
        scale: Softmax scale (``head_dim ** -0.5``).

    Returns:
        ``[T, heads, head_dim]`` attention output in ``q``'s dtype.
    """
    tokens, heads, head_dim = q.shape
    topk = topk_idxs.shape[-1]

    idx = topk_idxs.long()
    valid = idx >= 0
    safe_idx = idx.clamp_min(0)

    # [T, topk, head_dim] — gather the selected latent slots per query.
    gathered = kv.index_select(0, safe_idx.reshape(-1)).view(tokens, topk, head_dim)

    logits = torch.einsum("thd,tkd->thk", q.float(), gathered.float()) * scale
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))

    # Append the sink as an extra softmax column, then drop it from the average.
    sink = attn_sink.float().view(1, heads, 1).expand(tokens, heads, 1)
    probs = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)[..., :topk]

    out = torch.einsum("thk,tkd->thd", probs, gathered.float())
    return out.to(q.dtype)


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
        """
        num_windows, ratio, _ = tensor.shape
        d = self.head_dim
        out = tensor.new_full((num_windows, 2 * ratio, d), fill)
        out[:, ratio:] = tensor[:, :, d:]
        out[1:, :ratio] = tensor[:-1, :, :d]
        return out

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
            return hidden_states.new_zeros((0, self.head_dim))

        x = hidden_states.float()
        cutoff = num_windows * ratio
        kv = F.linear(x[:cutoff], self.wkv).unflatten(0, (num_windows, ratio))
        score = F.linear(x[:cutoff], self.wgate).unflatten(0, (num_windows, ratio))
        score = score + self.ape

        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))

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
        topk = min(self.index_topk, max(num_slots, 1))

        if num_slots == 0:
            return hidden_states.new_full((tokens, topk), -1, dtype=torch.int32)

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
        scores = scores.masked_fill(~allowed, float("-inf"))

        topk_idxs = scores.topk(topk, dim=-1)[1]
        # A query whose causal limit is below topk gets padding slots back from
        # topk(); mask those to -1 so the attention drops them.
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

        self.num_groups = config.o_groups
        if self.num_groups % self.world_size and self.world_size % self.num_groups:
            raise ValueError(
                f"o_groups ({self.num_groups}) and the TP degree "
                f"({self.world_size}) must divide one another"
            )
        self.num_groups_per_rank = max(1, self.num_groups // self.world_size)

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
        # Grouped low-rank output: per group, [o_lora_rank, heads*head_dim/groups].
        self.wo_a = nn.Parameter(
            torch.empty(
                self.num_groups_per_rank,
                self.o_lora_rank,
                (self.num_attention_heads * self.head_dim) // self.num_groups,
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
        # wo_a: checkpoint stores [groups * o_lora_rank, heads*head_dim/groups];
        # shard on the group axis, then reshape to [groups_per_rank, rank, in].
        set_weight_loader(
            self.wo_a,
            _grouped_wo_a_loader(
                num_groups_per_rank=self.num_groups_per_rank,
                o_lora_rank=self.o_lora_rank,
                num_shards=self.world_size,
                out_dtype=self.dtype,
            ),
        )
        # wo_b: row-parallel — shard the input (group) axis, all-reduce after.
        set_weight_loader(
            self.wo_b,
            fp8_dequant_weight_loader(
                shard_dim=1,
                shard_size=self.num_groups_per_rank * self.o_lora_rank,
                num_shards=self.world_size,
                out_dtype=self.dtype,
            ),
        )
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

        # wo_b is row-parallel: each rank holds a partial sum.
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
        same tensor goes into both halves of the cache. Padding tokens carry
        ``slot_mapping == -1`` and are redirected to a scratch slot that is
        never read, avoiding a data-dependent mask.
        """
        num_blocks = self.k_cache.shape[0]
        max_slot = num_blocks * block_size
        valid = (slot_mapping >= 0) & (slot_mapping < max_slot)
        # Park invalid writes on slot 0 of the last block, which the block
        # manager never hands out while a request is live.
        safe_slots = torch.where(
            valid, slot_mapping, torch.full_like(slot_mapping, max_slot - 1)
        )
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
        compressed_state: "CompressedKVState | None" = None,
    ) -> torch.Tensor:
        """Run attention for this layer.

        Args:
            hidden_states: ``[T, H]`` post-hyper-connection layer input.
            positions: ``[T]`` int32 absolute positions.
            attn_metadata: Per-layer runner metadata (see the onboarding guide).
            compressed_state: Per-request compressed KV buffers for this layer,
                or None on sliding-window-only layers.

        Returns:
            ``[T, H]`` attention output.
        """
        meta = attn_metadata[self.layer_name]
        max_query_len = meta["max_query_len"]
        decode_token_threshold = meta["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, meta, compressed_state
            )
        return self.forward_prefill(
            hidden_states, positions, meta, compressed_state
        )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        meta: dict,
        compressed_state: "CompressedKVState | None",
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
            compressed_positions = torch.arange(
                0, num_slots * ratio, ratio, device=latent_kv.device
            )
            c_cos, c_sin = self.rotary_emb(
                compressed_positions, dtype=hidden_states.dtype
            )
            compressed_kv = self.compressor(hidden_states, c_cos, c_sin)

            # A query at position t may see compressed slot s only once the
            # whole window it summarizes is in the past: s < (t + 1) // ratio.
            causal_limit = (
                torch.arange(1, tokens + 1, device=latent_kv.device) // ratio
            )
            offset = latent_kv.shape[0]

            if self.indexer is not None:
                index_positions = compressed_positions
                i_cos, i_sin = self.rotary_emb(
                    index_positions, dtype=hidden_states.dtype
                )
                index_kv = self.indexer.compressor(hidden_states, i_cos, i_sin)
                compressed_idxs = self.indexer(
                    hidden_states, q_lora, index_kv, cos, sin, causal_limit, offset
                )
            else:
                compressed_idxs = _hca_indices_prefill(
                    causal_limit, num_slots, offset
                )

            kv = torch.cat([latent_kv, compressed_kv], dim=0)
            topk_idxs = torch.cat([window_idxs, compressed_idxs], dim=-1)

            if compressed_state is not None:
                compressed_state.store(
                    compressed_kv,
                    index_kv if self.indexer is not None else None,
                )
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
        compressed_state: "CompressedKVState | None",
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

        if self.compress_ratio and compressed_state is not None:
            compressed_kv, index_kv = compressed_state.load()
            num_slots = compressed_kv.shape[0]
            offset = cached.shape[0]
            causal_limit = (positions.long() + 1) // self.compress_ratio
            causal_limit = causal_limit.clamp(max=num_slots)

            if self.indexer is not None:
                compressed_idxs = self.indexer(
                    hidden_states, q_lora, index_kv, cos, sin, causal_limit, offset
                )
            else:
                compressed_idxs = _hca_indices_prefill(
                    causal_limit, num_slots, offset
                )
            kv = torch.cat([cached, compressed_kv], dim=0)
            topk_idxs = torch.cat([window_idxs, compressed_idxs], dim=-1)
        else:
            kv = cached
            topk_idxs = window_idxs

        attn_out = sparse_latent_attention(
            q, kv, self.attn_sink, topk_idxs, self.softmax_scale
        )
        return self.project_output(attn_out, cos, sin)


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


def _grouped_wo_a_loader(
    num_groups_per_rank: int,
    o_lora_rank: int,
    num_shards: int,
    out_dtype: torch.dtype,
):
    """Loader for ``wo_a``: FP8 dequant, shard on groups, reshape to 3-D.

    The checkpoint stores ``[num_groups * o_lora_rank, heads * head_dim /
    num_groups]``, with the group axis folded into dim 0. The parameter wants
    ``[groups_per_rank, o_lora_rank, in_features]`` so the per-group einsum can
    index it directly.
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    from .weight_loaders import dequant_fp8_blockwise

    shard_size = num_groups_per_rank * o_lora_rank

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) != 2:
            raise ValueError(
                f"wo_a loader expects [weight, scale] slices, got {len(slices)}"
            )
        weight_slice, scale_slice = slices
        start = (rank % num_shards) * shard_size
        stop = start + shard_size

        # The scale grid is 128-blocked; o_lora_rank is a multiple of 128 for
        # every shipped config, so shard boundaries stay block-aligned.
        block = 128
        if start % block or shard_size % block:
            weight = weight_slice[:][start:stop]
            scale = scale_slice[:]
            dequantized = dequant_fp8_blockwise(
                weight_slice[:], scale, out_dtype=out_dtype
            )[start:stop]
        else:
            weight = weight_slice[start:stop]
            scale = scale_slice[start // block : stop // block]
            dequantized = dequant_fp8_blockwise(weight, scale, out_dtype=out_dtype)

        return dequantized.view(num_groups_per_rank, o_lora_rank, -1).contiguous()

    return SafetensorsWeightLoader(transform=transform)
