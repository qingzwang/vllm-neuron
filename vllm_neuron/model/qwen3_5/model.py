# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 text decoder for Neuron.

A hybrid stack: 18 gated-DeltaNet layers (``deltanet.py``) interleaved with 6
full-attention layers, in the pattern ``[linear, linear, linear, full] x 6``.
Every layer also has a dense SwiGLU MLP.

Three things here differ from the Qwen3-VL decoder next door, and each of them
is a silent-wrong-output hazard rather than a crash:

* **RMSNorm is zero-centred.** ``Qwen3_5RMSNorm`` scales by ``1 + weight``, not
  ``weight``, and the checkpoint's tensors are distributed around 0 accordingly.
  Using the usual form would multiply activations by roughly zero. Note the
  DeltaNet's own output norm is the *other* kind (plain ``weight``) — see
  ``Qwen3_5RMSNormGated`` upstream.
* **Attention output is gated.** ``q_proj`` emits twice the query width; per
  head it is ``[query | gate]``, and the attention output is scaled by
  ``sigmoid(gate)`` before ``o_proj``.
* **Rotary is partial and interleaved.** Only the first
  ``head_dim * partial_rotary_factor`` = 64 of each head's 256 dims are rotated;
  the rest pass through untouched. The 3 mRoPE axes are interleaved
  ``THWTHW...`` rather than laid out in chunks.

``head_dim`` is 256, above the flash-attention kernel's ``MAX_HEAD_DIM`` of 128,
so the 6 attention layers run in torch rather than on the kernel. That is the
expected performance ceiling for this model and is why they are only 6 of 24.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.interfaces import SupportsMRoPE
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, RecurrentLayerSpec
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from .config import Qwen3_5Config, Qwen3_5TextConfig
from .deltanet import Qwen3_5GatedDeltaNet

HF_TEXT_PREFIX = "model.language_model"

LINEAR_ATTENTION = "linear_attention"


class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with a zero-centred weight: ``normalise(x) * (1 + weight)``.

    Matches ``transformers``' ``Qwen3_5RMSNorm``, including doing the scale in
    float32 before casting back — ``(x * w).to(dtype)`` rather than
    ``x.to(dtype) * w``.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * (1.0 + self.weight.float())).to(input_dtype)


class Qwen3_5RotaryEmbedding(nn.Module):
    """Partial, interleaved mRoPE.

    ``inv_freq`` is sized from ``rotary_dim`` (``head_dim *
    partial_rotary_factor``), not ``head_dim``, so ``cos``/``sin`` come back
    ``[T, rotary_dim / 2]`` and only that prefix of each head is rotated.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        dim = config.rotary_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = list(config.mrope_section)
        if sum(self.mrope_section) != dim // 2:
            raise ValueError(
                f"mrope_section {self.mrope_section} sums to "
                f"{sum(self.mrope_section)}, expected rotary_dim // 2 = {dim // 2}"
            )

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[3, T]`` (or ``[T]``) positions -> ``(cos, sin)`` of ``[T, rotary_dim/2]``."""
        if position_ids.ndim == 1:
            position_ids = position_ids[None, None, :].expand(3, 1, -1)
        elif position_ids.ndim == 2 and position_ids.shape[0] == 3:
            position_ids = position_ids.unsqueeze(1)
        else:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        freqs = (inv_freq @ position_ids[:, :, None, :].float()).transpose(2, 3)
        freqs = self.interleave_mrope(freqs, self.mrope_section)

        cos, sin = freqs.cos(), freqs.sin()
        if cos.shape[0] == 1:
            cos, sin = cos.squeeze(0), sin.squeeze(0)
        return cos.to(dtype), sin.to(dtype)

    @staticmethod
    def interleave_mrope(
        freqs: torch.Tensor, mrope_section: list[int]
    ) -> torch.Tensor:
        """Reorder ``[3, ...]`` per-axis frequencies into one interleaved tensor.

        Slot ``i`` takes its frequency from axis ``i % 3`` while that axis still
        has section budget left; the leftover tail stays temporal. Written with
        ``torch.where`` rather than slice assignment so it traces cleanly.
        """
        indices = torch.arange(
            freqs.shape[-1], device=freqs.device, dtype=torch.int64
        )
        out = freqs[0]
        for axis, offset in enumerate((1, 2), start=1):
            mask = (indices % 3 == offset) & (indices < mrope_section[axis] * 3)
            out = torch.where(mask, freqs[axis], out)
        return out


def apply_partial_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate the first ``2 * cos.shape[-1]`` dims of ``x``, pass the rest through.

    ``x`` is ``[heads, tokens, head_dim]`` and ``cos``/``sin`` are
    ``[tokens, rotary_dim / 2]``, doubled here to ``rotary_dim`` for the
    ``rotate_half`` convention.
    """
    rotary_dim = 2 * cos.shape[-1]
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0)
    rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
    return torch.cat((rot * cos + rotated * sin, passthrough), dim=-1)


class Qwen3_5Attention(nn.Module):
    """Gated GQA with per-head QK norm and partial mRoPE.

    Q, K and V heads are TP-sharded. With 2 KV heads and TP=4 the KV heads are
    replicated: ranks 0-1 both hold KV head 0 (matching their Q heads 0-3) and
    ranks 2-3 hold KV head 1.

    Both paths compute attention in torch: ``head_dim`` 256 rules out
    ``NF.flash_attention``, and the decode megakernel fuses a QKV projection and
    full-width RoPE that do not match this layer's gate and partial rotary.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.dtype = config.torch_dtype
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.scaling = config.head_dim**-0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        if num_heads % self.world_size:
            raise ValueError(
                f"num_attention_heads={num_heads} must be divisible by "
                f"tp_size={self.world_size}"
            )
        self.num_heads_per_rank = num_heads // self.world_size
        if self.world_size >= num_kv_heads:
            self.num_kv_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // num_kv_heads
        else:
            self.num_kv_heads_per_rank = num_kv_heads // self.world_size
            self.num_kv_replicas = 1
        self.num_kv_groups = self.num_heads_per_rank // self.num_kv_heads_per_rank

        # q_proj emits [query | gate] per head, hence the factor of two.
        self.q_size = self.num_heads_per_rank * self.head_dim
        self.kv_size = self.num_kv_heads_per_rank * self.head_dim

        self.q_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, 2 * self.q_size, dtype=self.dtype)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.kv_size, dtype=self.dtype)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.kv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.q_size, self.hidden_size, dtype=self.dtype)
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, self.rms_norm_eps, self.dtype)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, self.rms_norm_eps, self.dtype)

        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

        # Each head contributes a contiguous 2 * head_dim block to q_proj, so
        # sharding its output dim by whole heads keeps query and gate together.
        set_weight_loader(
            self.q_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=2 * self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        # Replicated KV: several ranks share one KV head, so the shard index is
        # the KV head this rank's queries attend to, not the TP rank.
        kv_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.kv_size,
            num_shards=max(num_kv_heads // self.num_kv_heads_per_rank, 1),
            is_storage_transposed=True,
        )
        kv_shard = self.rank // self.num_kv_replicas
        for param in (self.k_proj_weight, self.v_proj_weight):
            set_weight_loader(param, with_rank_override(kv_loader, kv_shard))
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def _qkv(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``hidden -> (q, k, v, gate)``, q/k normed and rotated.

        q, k and v come back as ``[heads, tokens, head_dim]``; ``gate`` stays
        flat at ``[tokens, q_size]`` because that is how it multiplies the
        attention output.
        """
        tokens = hidden_states.shape[0]
        hidden_states = hidden_states.to(self.dtype)

        qg = (hidden_states @ self.q_proj_weight).view(
            tokens, self.num_heads_per_rank, 2 * self.head_dim
        )
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(tokens, self.q_size)

        q = self.q_norm(q).transpose(0, 1)
        k = self.k_norm(
            (hidden_states @ self.k_proj_weight).view(
                tokens, self.num_kv_heads_per_rank, self.head_dim
            )
        ).transpose(0, 1)
        v = (
            (hidden_states @ self.v_proj_weight)
            .view(tokens, self.num_kv_heads_per_rank, self.head_dim)
            .transpose(0, 1)
        )

        return (
            apply_partial_rotary(q, cos, sin),
            apply_partial_rotary(k, cos, sin),
            v,
            gate,
        )

    def _write_kv(self, k: torch.Tensor, v: torch.Tensor, metadata: dict) -> None:
        """Scatter ``[heads, tokens, head_dim]`` K/V into the paged cache."""
        slot_mapping = metadata["slot_mapping"]
        block_size = metadata["block_size"]
        num_tokens = slot_mapping.shape[0]
        nkh = self.num_kv_heads_per_rank

        heads = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(num_tokens)
        blocks = (slot_mapping // block_size).repeat(nkh)
        offsets = (slot_mapping % block_size).repeat(nkh)

        self.k_cache.index_put_(
            (blocks, heads, offsets), k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        )
        self.v_cache.index_put_(
            (blocks, heads, offsets), v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        )

    def _finish(self, attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Gate by ``sigmoid(gate)``, then the row-parallel output projection."""
        attn_output = attn_output * torch.sigmoid(gate.float()).to(attn_output.dtype)
        return attn_output @ self.o_proj_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        metadata = attn_metadata[self.layer_name]
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, metadata
            )
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, position_embeddings, metadata)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        metadata: dict,
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        tokens = hidden_states.shape[0]
        q, k, v, gate = self._qkv(hidden_states, cos, sin)

        self._write_kv(k, v, metadata)

        k = k.repeat_interleave(self.num_kv_groups, dim=0)
        v = v.repeat_interleave(self.num_kv_groups, dim=0)

        scores = (q.float() @ k.float().transpose(-1, -2)) * self.scaling
        causal = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(v.dtype)
        attn_output = (attn @ v).transpose(0, 1).reshape(tokens, self.q_size)

        output = self._finish(attn_output, gate)
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        metadata: dict,
    ) -> torch.Tensor:
        block_table = metadata["block_table_tensor"]
        num_reqs = block_table.shape[0]
        cos, sin = position_embeddings
        q, k, v, gate = self._qkv(hidden_states, cos, sin)
        self._write_kv(k, v, metadata)

        # Gather this batch's pages: [reqs, blocks, kv_heads, block_size, dim]
        # -> [reqs, kv_heads, ctx, dim].
        blocks = block_table.to(torch.long)
        nkh = self.num_kv_heads_per_rank
        k_ctx = self.k_cache[blocks].permute(0, 2, 1, 3, 4).reshape(
            num_reqs, nkh, -1, self.head_dim
        )
        v_ctx = self.v_cache[blocks].permute(0, 2, 1, 3, 4).reshape(
            num_reqs, nkh, -1, self.head_dim
        )
        ctx_len = k_ctx.shape[2]

        q = q.transpose(0, 1).reshape(num_reqs, self.num_heads_per_rank, 1, self.head_dim)
        k_ctx = k_ctx.repeat_interleave(self.num_kv_groups, dim=1)
        v_ctx = v_ctx.repeat_interleave(self.num_kv_groups, dim=1)

        scores = (q.float() @ k_ctx.float().transpose(-1, -2)) * self.scaling
        # Everything at or before this token's position is visible; the rest of
        # the page is uninitialised.
        key_positions = torch.arange(ctx_len, device=q.device).view(1, 1, 1, ctx_len)
        visible = key_positions <= positions.view(num_reqs, 1, 1, 1)
        scores = scores.masked_fill(~visible, float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(v_ctx.dtype)
        attn_output = (attn @ v_ctx).reshape(num_reqs, self.q_size)

        output = self._finish(attn_output, gate)
        if self.world_size > 1:
            self.tp_group.all_reduce(output)
        return output


class Qwen3_5MLP(nn.Module):
    """Dense SwiGLU MLP with the intermediate dim TP-sharded."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        hidden = config.hidden_size
        if config.intermediate_size % self.world_size:
            raise ValueError(
                f"intermediate_size={config.intermediate_size} must be divisible "
                f"by tp_size={self.world_size}"
            )
        inter = config.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(torch.empty(hidden, inter, dtype=self.dtype))
        self.up_proj_weight = nn.Parameter(torch.empty(hidden, inter, dtype=self.dtype))
        self.down_proj_weight = nn.Parameter(torch.empty(inter, hidden, dtype=self.dtype))

        for param in (self.gate_proj_weight, self.up_proj_weight):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=inter,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=inter,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        out = (
            F.silu(hidden_states @ self.gate_proj_weight)
            * (hidden_states @ self.up_proj_weight)
        ) @ self.down_proj_weight

        if self.world_size > 1:
            if is_prefill:
                out = self.tp_group.reduce_scatter(out, dim=0).contiguous()
            else:
                self.tp_group.all_reduce(out)
        return out


class Qwen3_5DecoderLayer(nn.Module):
    """One layer: either a DeltaNet mixer or full attention, then the MLP."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear_attention = config.layer_types[layer_idx] == LINEAR_ATTENTION
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        if self.is_linear_attention:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
            self.mixer_name = self.linear_attn.layer_name
        else:
            self.self_attn = Qwen3_5Attention(config, layer_idx)
            self.mixer_name = self.self_attn.layer_name
        self.mlp = Qwen3_5MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        metadata = attn_metadata[self.mixer_name]
        is_decode = metadata["max_query_len"] <= metadata["decode_token_threshold"]

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear_attention:
            hidden_states = self.linear_attn(hidden_states, positions, attn_metadata)
        else:
            hidden_states = self.self_attn(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        return residual + hidden_states


class Qwen3_5TextModel(nn.Module):
    """Embedding -> 24 hybrid decoder layers -> final norm."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                is_storage_transposed=False,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor | None,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first = self.layers[0].mixer_name
        is_prefill = (
            attn_metadata[first]["max_query_len"]
            > attn_metadata[first]["decode_token_threshold"]
        )

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )
        position_embeddings = self.rotary_emb(
            rotary_position_ids if rotary_position_ids is not None else positions,
            dtype=self.config.torch_dtype,
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, positions, position_embeddings, attn_metadata
            )

        hidden_states = self.norm(hidden_states)
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module, SupportsMRoPE):
    """Text-only Qwen3.5 with a tied LM head.

    The checkpoint has ``tie_word_embeddings: true`` and genuinely ships no
    ``lm_head`` tensor, so the head is loaded from ``embed_tokens``.
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.text_config = config.text_config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.language_model = Qwen3_5TextModel(config.text_config)

        nc = config.text_config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (
            nc.max_logprobs != 0 or nc.debug_logits_dir is not None
        )

        self.lm_head = neuron_nn.ColumnParallelLinear(
            self.text_config.hidden_size,
            self.text_config.vocab_size,
            bias=False,
            dtype=self.text_config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.text_config.vocab_size // self.world_size,
                num_shards=self.world_size,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    # ── mRoPE ────────────────────────────────────────────────────────────

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list,
    ) -> tuple[torch.Tensor, int]:
        """3D mRoPE positions for a text-only prompt.

        vLLM derives ``uses_mrope`` from the HF config and then *requires* this
        protocol, so a text-only port of an mRoPE model still has to provide it.
        With no vision items the three axes carry identical values, which is
        exactly what a plain 1D position sequence expands to, and the decode
        offset is zero because the last position is ``len - 1``.
        """
        if mm_features:
            raise NotImplementedError(
                "Qwen3.5 on Neuron is text-only so far: mRoPE positions for "
                f"{len(mm_features)} multimodal item(s) need the vision grid "
                "layout. Launch with limit_mm_per_prompt={'image': 0, "
                "'video': 0}."
            )
        positions = torch.arange(len(input_tokens), dtype=torch.int64)
        return positions.unsqueeze(0).expand(3, -1).contiguous(), 0

    # ── KV / state cache ─────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        """Report both kinds of layer: paged KV for attention, state for DeltaNet."""
        layers: list[LayerSpec] = []
        recurrent: list[RecurrentLayerSpec] = []
        for layer in self.language_model.layers:
            if layer.is_linear_attention:
                mixer = layer.linear_attn
                recurrent.append(
                    RecurrentLayerSpec(
                        name=mixer.layer_name,
                        shapes=self.text_config.state_shapes(self.world_size),
                        dtypes=self.text_config.state_dtypes(),
                    )
                )
            else:
                mixer = layer.self_attn
                layers.append(
                    LayerSpec(
                        name=mixer.layer_name,
                        num_kv_heads=mixer.num_kv_heads_per_rank,
                        head_size=mixer.head_dim,
                        dtype=mixer.dtype,
                        sliding_window_size=None,
                        chunk_size=None,
                    )
                )
        return KVSpec(layers=layers, recurrent_layers=recurrent)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer in self.language_model.layers:
            mixer = layer.linear_attn if layer.is_linear_attention else layer.self_attn
            if mixer.layer_name not in kv_caches:
                raise Exception(f"cache for layer {mixer.layer_name} not initialized")
            tensors = kv_caches[mixer.layer_name]
            if layer.is_linear_attention:
                mixer.conv_state, mixer.recurrent_state = tensors
            else:
                mixer.k_cache, mixer.v_cache = tensors

    # ── Forward ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        # Accepted and ignored. The runner passes these for any checkpoint whose
        # HF config has a vision_config, which this one does even though the
        # tower is not implemented yet; refusing them would block a text-only
        # launch. Requests carrying images are rejected at the frontend instead —
        # see the limit_mm_per_prompt in examples/.../qwen3_5/run.py. Consuming
        # them is the VL stage's job.
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        first = self.language_model.layers[0].mixer_name
        is_prefill = (
            attn_metadata[first]["max_query_len"]
            > attn_metadata[first]["decode_token_threshold"]
        )
        num_tokens = input_ids.shape[0]
        if is_prefill and (
            num_tokens <= self.world_size or num_tokens % self.world_size
        ):
            raise ValueError(
                f"prompt length ({num_tokens}) must be a multiple of world_size "
                f"({self.world_size}) and larger than it, for sequence parallelism"
            )

        hidden_states = self.language_model(
            input_ids,
            positions,
            rotary_position_ids,
            attn_metadata,
            rank=rank,
        )

        hidden_states = torch.index_select(hidden_states, 0, sampling_positions)
        logits = self.lm_head(hidden_states)

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)
        return sampled_tokens, gathered_logits

    # ── Construction and weights ─────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config,
        text_neuron_config=None,
        vision_neuron_config=None,
    ) -> "Qwen3_5ForCausalLM":
        return cls(
            Qwen3_5Config.from_configs(
                hf_config,
                text_neuron_config=text_neuron_config,
                vision_neuron_config=vision_neuron_config,
            )
        )

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        mappings: dict[str, object] = {}
        for i, layer in enumerate(self.language_model.layers):
            hf = f"{HF_TEXT_PREFIX}.layers.{i}"
            ours = f"language_model.layers.{i}"

            mappings[f"{ours}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
            mappings[f"{ours}.post_attention_layernorm.weight"] = (
                f"{hf}.post_attention_layernorm.weight"
            )
            mappings[f"{ours}.mlp.gate_proj_weight"] = f"{hf}.mlp.gate_proj.weight"
            mappings[f"{ours}.mlp.up_proj_weight"] = f"{hf}.mlp.up_proj.weight"
            mappings[f"{ours}.mlp.down_proj_weight"] = f"{hf}.mlp.down_proj.weight"

            if layer.is_linear_attention:
                mixer_hf = f"{hf}.linear_attn"
                mixer = f"{ours}.linear_attn"
                mappings[f"{mixer}.in_proj_qkv_weight"] = (
                    f"{mixer_hf}.in_proj_qkv.weight"
                )
                mappings[f"{mixer}.in_proj_z_weight"] = f"{mixer_hf}.in_proj_z.weight"
                # Order matters: the loader concatenates b then a.
                mappings[f"{mixer}.in_proj_ba_weight"] = [
                    f"{mixer_hf}.in_proj_b.weight",
                    f"{mixer_hf}.in_proj_a.weight",
                ]
                mappings[f"{mixer}.conv1d_weight"] = f"{mixer_hf}.conv1d.weight"
                mappings[f"{mixer}.dt_bias"] = f"{mixer_hf}.dt_bias"
                mappings[f"{mixer}.A_log"] = f"{mixer_hf}.A_log"
                mappings[f"{mixer}.norm_weight"] = f"{mixer_hf}.norm.weight"
                mappings[f"{mixer}.out_proj_weight"] = f"{mixer_hf}.out_proj.weight"
            else:
                mixer_hf = f"{hf}.self_attn"
                mixer = f"{ours}.self_attn"
                mappings[f"{mixer}.q_proj_weight"] = f"{mixer_hf}.q_proj.weight"
                mappings[f"{mixer}.k_proj_weight"] = f"{mixer_hf}.k_proj.weight"
                mappings[f"{mixer}.v_proj_weight"] = f"{mixer_hf}.v_proj.weight"
                mappings[f"{mixer}.o_proj_weight"] = f"{mixer_hf}.o_proj.weight"
                mappings[f"{mixer}.q_norm.weight"] = f"{mixer_hf}.q_norm.weight"
                mappings[f"{mixer}.k_norm.weight"] = f"{mixer_hf}.k_norm.weight"

        mappings["language_model.embed_tokens.weight"] = (
            f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        )
        mappings["language_model.norm.weight"] = f"{HF_TEXT_PREFIX}.norm.weight"
        if not self.text_config.tie_word_embeddings:
            raise NotImplementedError(
                "Qwen3.5 without tie_word_embeddings is not implemented; this "
                "checkpoint ships no lm_head tensor."
            )
        mappings["lm_head.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank,
            self.world_size,
            self,
            mappings,
            device,
            strict=False,
        ).state_dict

        # The checkpoint loader is necessarily strict=False (it does not know
        # about buffers like rotary inv_freq), which means a parameter whose
        # mapping key is wrong stays at its uninitialised ``torch.empty`` value
        # and the model generates fluent garbage with no error anywhere. Check
        # explicitly instead.
        expected = {name for name, _ in self.named_parameters()}
        unfilled = sorted(expected - set(rank_sharded))
        if unfilled:
            raise RuntimeError(
                f"{len(unfilled)} parameter(s) got no checkpoint tensor and "
                f"would stay uninitialised: {unfilled[:8]}"
                + (" ..." if len(unfilled) > 8 else "")
            )

        # dt_bias / A_log stay float32 (they feed a softplus/exp whose result
        # decides the decay rate); everything else follows the model dtype.
        target = self.text_config.torch_dtype
        keep_fp32 = {"dt_bias", "A_log"}
        for name, tensor in rank_sharded.items():
            if name.rsplit(".", 1)[-1] in keep_fp32:
                rank_sharded[name] = tensor.to(torch.float32)
            elif tensor.dtype != target:
                rank_sharded[name] = tensor.to(target)

        self.load_state_dict(rank_sharded, strict=False, assign=True)


__all__ = ["Qwen3_5ForCausalLM", "Qwen3_5TextModel"]
