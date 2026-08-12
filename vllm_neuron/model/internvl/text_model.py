# SPDX-License-Identifier: Apache-2.0
"""Qwen2 decoder for InternVL3 (BF16).

Adapted from the Qwen3-VL text backbone in ``vllm_neuron/model/qwen3_vl``. Three
architectural differences drive every change:

  1. Qwen2 has **bias on q/k/v** (none on o_proj). Both the prefill
     ``NF.qkv_proj`` and the decode ``NF.attention_decode`` megakernel take a
     fused QKV bias, so the bias rides along in the same fused ops.
  2. Qwen2 has **no per-head QK normalization**, so the qk_norm fusion is off.
  3. Positions are plain 1-D RoPE, not 3-D M-RoPE — InternVL has no spatial
     position encoding, image tokens occupy ordinary sequential positions.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen2-specific. Change when porting.

PARALLELISM SHARDING:
  TP:  Attention heads sharded, MLP intermediate sharded.
  SP:  Prefill runs sequence-parallel (all-gather before attention,
       reduce-scatter after); decode all-reduces instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.qkv import NormType
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import InternVLTextConfig


class Qwen2RMSNorm(nn.Module):
    """RMSNorm in float32, matching HF Qwen2RMSNorm."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * x).to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    """Plain 1-D RoPE.

    Returns ``(cos, sin)`` of shape ``[T, head_dim/2]``, the same contract the
    Qwen3-VL M-RoPE module uses, so the attention code below is unchanged in
    shape handling. Callers double to full head_dim via ``cat((cos, cos))``.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: InternVLTextConfig):
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [T] x [head_dim/2] -> [T, head_dim/2]
        pos = position_ids.reshape(-1).to(torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(torch.float32))
        return freqs.cos().to(dtype), freqs.sin().to(dtype)


class Qwen2Attention(nn.Module):
    """GQA attention with fused QKV (+bias) and no QK normalization."""

    def __init__(self, config: InternVLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: head sharding, replicating KV heads when TP exceeds them <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in = (self.num_attention_heads * self.head_dim) // self.world_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        # <-- MODEL-SPECIFIC: Qwen2 has q/k/v bias (Qwen3 does not).
        self.qkv_proj_bias = nn.Parameter(torch.empty(qkv_size, dtype=self.dtype))
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in, self.hidden_size, dtype=self.dtype)
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]
        self.k_cache = None
        self.v_cache = None
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Checkpoint stores q/k/v separately; fuse and shard them per rank."""
        for param, shard_dim in ((self.qkv_proj_weight, 1), (self.qkv_proj_bias, 0)):
            set_weight_loader(
                param,
                fused_qkv_weight_loader(
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    shard_dim=shard_dim,
                    num_shards=self.world_size,
                    is_storage_transposed=shard_dim == 1,
                    num_kv_replicas=self.num_kv_replicas,
                ),
            )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=(self.num_attention_heads * self.head_dim)
                // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        is_decode: bool = False,
    ) -> torch.Tensor:
        if is_decode:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    def _write_kv_cache(self, k, v, attn_metadata):
        """Scatter K/V into the paged cache for this layer."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        nkh = self.num_key_value_heads_per_rank

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        head_idx = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(slot_mapping.shape[0])
        self.k_cache.index_put_(
            (block_indices.repeat(nkh), head_idx, position_indices.repeat(nkh)),
            k.reshape(-1, self.head_dim).to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices.repeat(nkh), head_idx, position_indices.repeat(nkh)),
            v.reshape(-1, self.head_dim).to(self.v_cache.dtype),
        )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """QKV proj (+bias) -> RoPE -> flash attention -> O proj."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape

        cos, sin = position_embeddings
        cos_cache = torch.cat((cos, cos), dim=-1).unsqueeze(0)
        sin_cache = torch.cat((sin, sin), dim=-1).unsqueeze(0)

        # <-- MODEL-SPECIFIC: bias passed through, qk_norm fusion left off.
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=self.qkv_proj_bias,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            d_head=self.head_dim,
            norm_type=NormType.NO_NORM,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        nah, nkh, d = (
            self.num_attention_heads_per_rank,
            self.num_key_value_heads_per_rank,
            self.head_dim,
        )
        q = q.view(tokens, nah, d).transpose(0, 1)
        k = k.view(tokens, nkh, d).transpose(0, 1)
        v = v.view(tokens, nkh, d).transpose(0, 1)

        self._write_kv_cache(k, v, attn_metadata)

        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.repeat_interleave(self.num_key_value_groups, dim=0)
        attn_output = NF.flash_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v,
            scale=self.scaling,
            tp_q=False,
            tp_out=True,
        )
        attn_output = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight).squeeze(0)

        # >>> PARALLELISM: reduce-scatter back to the SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ) -> torch.Tensor:
        """Fused decode megakernel, with QKV bias and no QK norm."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S = tokens // B
        assert tokens == B * S, f"decode tokens {tokens} not divisible by batch {B}"

        hidden_states = hidden_states.to(self.dtype)
        nkh = self.num_key_value_heads_per_rank
        half_d = self.head_dim // 2

        cos, sin = position_embeddings
        def _for_kernel(t):
            return (
                t[:, :half_d]
                .view(B, S, half_d)
                .permute(2, 0, 1)
                .contiguous()
                .to(self.dtype)
            )

        cos_k, sin_k = _for_kernel(cos), _for_kernel(sin)

        k_cache = (
            self.k_cache.squeeze(1)
            if self.k_cache.dim() == 4 and nkh
            else self.k_cache
        )
        v_cache = (
            self.v_cache.squeeze(1)
            if self.v_cache.dim() == 4 and nkh
            else self.v_cache
        )

        output, K_new, V_new = NF.attention_decode(
            X=hidden_states.view(B, S, hidden),
            W_qkv=self.qkv_proj_weight,
            # <-- MODEL-SPECIFIC: Qwen2 QKV bias; QK norm stays disabled.
            bias_qkv=self.qkv_proj_bias,
            rmsnorm_X_enabled=False,
            rmsnorm_QK_pre_rope_enabled=False,
            cos=cos_k,
            sin=sin_k,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=block_table,
            K_cache=k_cache,
            V_cache=v_cache,
            pos_ids=positions.view(B, S).to(torch.float32),
            swa_start_pos_ids=None,
            softmax_scale=self.scaling,
            update_cache=False,
            W_out=self.o_proj_weight,
            transposed_out=False,
            out_in_sb=False,
        )

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]

        k_new = (
            K_new.permute(1, 2, 0)
            .reshape(B, nkh, S, self.head_dim)
            .transpose(0, 1)
            .reshape(nkh, B * S, self.head_dim)
        )
        head_idx = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        self.k_cache.index_put_(
            (block_indices.repeat(nkh), head_idx, position_indices.repeat(nkh)),
            k_new.reshape(-1, self.head_dim).to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices.repeat(nkh), head_idx, position_indices.repeat(nkh)),
            V_new.transpose(0, 1).reshape(-1, self.head_dim).to(self.v_cache.dtype),
        )

        # >>> PARALLELISM: TP all-reduce after the megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)
        return output

    def build_weight_mappings(self, prefix: str) -> dict[str, str | list[str]]:
        """Parameter name -> checkpoint key(s); fused QKV takes [q, k, v]."""
        return {
            "qkv_proj_weight": [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ],
            "qkv_proj_bias": [
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
            ],
            "o_proj_weight": f"{prefix}.self_attn.o_proj.weight",
        }


class Qwen2MLP(nn.Module):
    """SwiGLU MLP, no bias. TP shards the intermediate dimension."""

    def __init__(self, config: InternVLTextConfig):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        inter_per_rank = config.intermediate_size // self.world_size
        self.inter_per_rank = inter_per_rank

        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, inter_per_rank, dtype=self.dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, inter_per_rank, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(inter_per_rank, config.hidden_size, dtype=self.dtype)
        )
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        for param in (self.gate_proj_weight, self.up_proj_weight):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=self.inter_per_rank,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.inter_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        out = NF.mlp(
            hidden=hidden_states,
            gate_w=self.gate_proj_weight,
            up_w=self.up_proj_weight,
            down_w=self.down_proj_weight,
        )
        # >>> PARALLELISM: SP reduce-scatter on prefill, all-reduce on decode <<<
        if self.world_size > 1:
            if is_prefill:
                out = self.tp_group.reduce_scatter(out, dim=0)
            else:
                out = self.tp_group.all_reduce(out)
        return out

    def build_weight_mappings(self, prefix: str) -> dict[str, str]:
        return {
            "gate_proj_weight": f"{prefix}.mlp.gate_proj.weight",
            "up_proj_weight": f"{prefix}.mlp.up_proj.weight",
            "down_proj_weight": f"{prefix}.mlp.down_proj.weight",
        }


class Qwen2DecoderLayer(nn.Module):
    """Pre-norm decoder layer: norm -> attn -> residual -> norm -> MLP -> residual."""

    def __init__(self, config: InternVLTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        is_decode: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, position_embeddings, attn_metadata, is_decode
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        return residual + hidden_states

    def build_weight_mappings(self, prefix: str) -> dict[str, str | list[str]]:
        m: dict[str, str | list[str]] = {
            "input_layernorm.weight": f"{prefix}.input_layernorm.weight",
            "post_attention_layernorm.weight": (
                f"{prefix}.post_attention_layernorm.weight"
            ),
        }
        for name, key in self.self_attn.build_weight_mappings(prefix).items():
            m[f"self_attn.{name}"] = key
        for name, key in self.mlp.build_weight_mappings(prefix).items():
            m[f"mlp.{name}"] = key
        return m
