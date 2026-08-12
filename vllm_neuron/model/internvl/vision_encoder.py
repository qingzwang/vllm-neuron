# SPDX-License-Identifier: Apache-2.0
"""InternViT-300M vision encoder (BF16) for InternVL3.

PatchEmbed -> [CLS] + pos embed -> 24 x (LayerNorm, Attention, LayerScale,
LayerNorm, MLP, LayerScale) -> drop CLS.

Simpler than the Qwen3-VL encoder in one important way: InternVL's dynamic
tiling always emits **448x448 tiles**, so every item is exactly
``grid_size**2 = 1024`` patches. There is no variable-length packing, no
bin-packing across items and no block-local attention bounds — each tile is an
independent sequence of a fixed length, so the whole batch is just
``[num_tiles, 1 + 1024, hidden]`` and attention runs per tile.

torch_neuronx replaces F.gelu/nn.GELU with a C extension that Dynamo cannot
trace, so this module uses an erf-based equivalent (same approach as the
Qwen3-VL encoder).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    InternViT-specific. Change when porting.

PARALLELISM SHARDING:
  TP:  Attention heads sharded (16 heads), MLP intermediate sharded (4096).
       Uses the vision TP group, independent of the text model's TP group.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from vllm_neuron.parallel.neuron_parallel_state import get_neuron_vision_tp_group
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .config import InternVLVisionConfig
from .weight_loaders import (
    patch_embed_weight_loader,
    vis_qkv_bias_loader,
    vis_qkv_weight_loader,
)


@torch.compiler.allow_in_graph
def _gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU via erf.

    torch_neuronx patches F.gelu with a C extension Dynamo cannot trace through,
    so spell it out. InternViT's hidden_act is "gelu" (exact, not tanh).
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class InternVisionPatchEmbed(nn.Module):
    """Conv2d(14x14, stride 14) expressed as reshape + matmul.

    Patch size equals stride, so the convolution is a non-overlapping tiling:
    fold each 14x14x3 patch into a 588-vector and matmul with the flattened
    conv weight. That avoids a real conv on device and keeps the graph to a
    single GEMM.

    Weight stays in checkpoint layout ``[hidden, 3, 14, 14]`` and is flattened
    at load time to ``[588, hidden]``.
    """

    def __init__(self, config: InternVLVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.patch_size = config.patch_size
        self.grid = config.grid_size
        patch_dim = config.num_channels * config.patch_size**2

        # <-- MODEL-SPECIFIC: stored transposed relative to Conv2d for the matmul.
        self.proj_weight = nn.Parameter(
            torch.empty(patch_dim, config.hidden_size, dtype=dtype)
        )
        self.proj_bias = nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))
        self.class_embedding = nn.Parameter(
            torch.empty(1, 1, config.hidden_size, dtype=dtype)
        )
        # [1, 1 + grid**2, hidden]. The HF implementation interpolates the patch
        # part to the actual grid; for 448/14 tiles that is the identity, so it
        # is used as stored. config.py asserts the shapes agree.
        self.position_embedding = nn.Parameter(
            torch.empty(
                1, 1 + config.num_patches_per_tile, config.hidden_size, dtype=dtype
            )
        )
        set_weight_loader(self.proj_weight, patch_embed_weight_loader())

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[num_tiles, 3, H, W] -> [num_tiles, 1 + grid**2, hidden]."""
        n, c, h, w = pixel_values.shape
        p, g = self.patch_size, self.grid
        if h != p * g or w != p * g:
            raise ValueError(
                f"InternViT expects {p * g}x{p * g} tiles, got {h}x{w}. "
                f"Dynamic tiling should have resized every tile to image_size."
            )
        x = pixel_values.to(self.dtype)
        # [n, c, g, p, g, p] -> [n, g, g, c, p, p] -> [n, g*g, c*p*p]
        x = x.reshape(n, c, g, p, g, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(n, g * g, c * p * p)
        patches = torch.matmul(x, self.proj_weight) + self.proj_bias

        cls = self.class_embedding.to(self.dtype).expand(n, 1, -1)
        embeddings = torch.cat([cls, patches], dim=1)
        return embeddings + self.position_embedding.to(self.dtype)


class InternVisionAttention(nn.Module):
    """Multi-head self-attention over one tile's fixed-length sequence.

    Plain softmax attention with no mask: every tile attends within itself and
    the batch dim already separates tiles.
    """

    def __init__(self, config: InternVLVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        # >>> PARALLELISM: vision TP group, decoupled from the text model <<<
        self.tp_group = get_neuron_vision_tp_group()
        self.tp_size = self.tp_group.world_size if self.tp_group else 1
        if config.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                f"InternViT heads ({config.num_attention_heads}) must be "
                f"divisible by vision TP size ({self.tp_size})"
            )
        self.num_heads_per_rank = config.num_attention_heads // self.tp_size
        qkv_out = 3 * self.num_heads_per_rank * self.head_dim

        # <-- MODEL-SPECIFIC: fused qkv WITH bias (config.qkv_bias=True)
        self.qkv_weight = nn.Parameter(
            torch.empty(config.hidden_size, qkv_out, dtype=dtype)
        )
        self.qkv_bias = nn.Parameter(torch.empty(qkv_out, dtype=dtype))
        self.proj_weight = nn.Parameter(
            torch.empty(
                self.num_heads_per_rank * self.head_dim,
                config.hidden_size,
                dtype=dtype,
            )
        )
        # proj bias is added once after the TP all-reduce, so only rank 0 holds it
        # (adding it on every rank would multiply it by tp_size).
        self.proj_bias = nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))
        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        hidden = self.config.hidden_size
        set_weight_loader(
            self.qkv_weight,
            vis_qkv_weight_loader(self.num_heads_per_rank, self.head_dim, hidden),
        )
        set_weight_loader(
            self.qkv_bias,
            vis_qkv_bias_loader(self.num_heads_per_rank, self.head_dim, hidden),
        )
        # proj is row-parallel: shard its input dim (the head dim), all-reduce after.
        set_weight_loader(
            self.proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_heads_per_rank * self.head_dim,
                num_shards=self.tp_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[n, s, hidden] -> [n, s, hidden]."""
        n, s, _ = hidden_states.shape
        h, d = self.num_heads_per_rank, self.head_dim

        qkv = torch.matmul(hidden_states, self.qkv_weight) + self.qkv_bias
        q, k, v = qkv.split(h * d, dim=-1)
        # [n, s, h*d] -> [n, h, s, d]
        q = q.reshape(n, s, h, d).transpose(1, 2)
        k = k.reshape(n, s, h, d).transpose(1, 2)
        v = v.reshape(n, s, h, d).transpose(1, 2)

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = torch.softmax(attn.float(), dim=-1).to(self.dtype)
        out = torch.matmul(attn, v)
        # [n, h, s, d] -> [n, s, h*d]
        out = out.transpose(1, 2).reshape(n, s, h * d)

        out = torch.matmul(out, self.proj_weight)
        # >>> PARALLELISM: row-parallel reduction, then the un-sharded bias <<<
        if self.tp_size > 1:
            out = self.tp_group.all_reduce(out)
        return out + self.proj_bias


class InternVisionMLP(nn.Module):
    """fc1 -> GELU -> fc2, both with bias. Intermediate dim is TP-sharded."""

    def __init__(self, config: InternVLVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.tp_group = get_neuron_vision_tp_group()
        self.tp_size = self.tp_group.world_size if self.tp_group else 1
        if config.intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"InternViT intermediate_size ({config.intermediate_size}) must be "
                f"divisible by vision TP size ({self.tp_size})"
            )
        inter_per_rank = config.intermediate_size // self.tp_size

        self.fc1_weight = nn.Parameter(
            torch.empty(config.hidden_size, inter_per_rank, dtype=dtype)
        )
        self.fc1_bias = nn.Parameter(torch.empty(inter_per_rank, dtype=dtype))
        self.fc2_weight = nn.Parameter(
            torch.empty(inter_per_rank, config.hidden_size, dtype=dtype)
        )
        # As with attention proj: added after the all-reduce, unsharded.
        self.fc2_bias = nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))
        self._setup_weight_loaders(inter_per_rank)

    def _setup_weight_loaders(self, inter_per_rank: int) -> None:
        # fc1 is column-parallel: shard the output dim.
        set_weight_loader(
            self.fc1_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=inter_per_rank,
                num_shards=self.tp_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.fc1_bias,
            sharding_weight_loader(
                shard_dim=0, shard_size=inter_per_rank, num_shards=self.tp_size
            ),
        )
        # fc2 is row-parallel: shard the input dim.
        set_weight_loader(
            self.fc2_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=inter_per_rank,
                num_shards=self.tp_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(hidden_states, self.fc1_weight) + self.fc1_bias
        x = _gelu(x)
        x = torch.matmul(x, self.fc2_weight)
        if self.tp_size > 1:
            x = self.tp_group.all_reduce(x)
        return x + self.fc2_bias


class InternVisionEncoderLayer(nn.Module):
    """Pre-norm block with LayerScale on both residual branches.

    HF reference:
        h = h + attn(norm1(h)) * ls1
        h = h + mlp(norm2(h)) * ls2
    """

    def __init__(self, config: InternVLVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )
        self.norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )
        self.attn = InternVisionAttention(config, dtype)
        self.mlp = InternVisionMLP(config, dtype)
        # <-- MODEL-SPECIFIC: LayerScale, one scalar per channel per branch.
        self.ls1 = nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))
        self.ls2 = nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states)) * self.ls1
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2
        return hidden_states


class InternVisionModel(nn.Module):
    """Full InternViT tower.

    forward: ``[num_tiles, 3, 448, 448]`` -> ``[num_tiles, 1024, 1024]``
    (CLS token dropped, matching HF ``extract_feature``'s ``vit_embeds[:, 1:, :]``).
    """

    def __init__(
        self, config: InternVLVisionConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.embeddings = InternVisionPatchEmbed(config, dtype)
        self.layers = nn.ModuleList(
            [
                InternVisionEncoderLayer(config, dtype)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # Drop the CLS token; only patch tokens feed the projector.
        return hidden_states[:, 1:, :]

    def build_weight_mappings(
        self, prefix: str = "vision_model"
    ) -> dict[str, str | list[str]]:
        """Parameter name -> checkpoint key(s).

        The fused qkv weight/bias take a single checkpoint tensor each and are
        re-sharded by the loaders in weight_loaders.py.
        """
        m: dict[str, str | list[str]] = {
            "embeddings.proj_weight": f"{prefix}.embeddings.patch_embedding.weight",
            "embeddings.proj_bias": f"{prefix}.embeddings.patch_embedding.bias",
            "embeddings.class_embedding": f"{prefix}.embeddings.class_embedding",
            "embeddings.position_embedding": f"{prefix}.embeddings.position_embedding",
        }
        for i in range(self.config.num_hidden_layers):
            p = f"{prefix}.encoder.layers.{i}"
            m.update(
                {
                    f"layers.{i}.norm1.weight": f"{p}.norm1.weight",
                    f"layers.{i}.norm1.bias": f"{p}.norm1.bias",
                    f"layers.{i}.norm2.weight": f"{p}.norm2.weight",
                    f"layers.{i}.norm2.bias": f"{p}.norm2.bias",
                    f"layers.{i}.ls1": f"{p}.ls1",
                    f"layers.{i}.ls2": f"{p}.ls2",
                    f"layers.{i}.attn.qkv_weight": f"{p}.attn.qkv.weight",
                    f"layers.{i}.attn.qkv_bias": f"{p}.attn.qkv.bias",
                    f"layers.{i}.attn.proj_weight": f"{p}.attn.proj.weight",
                    f"layers.{i}.attn.proj_bias": f"{p}.attn.proj.bias",
                    f"layers.{i}.mlp.fc1_weight": f"{p}.mlp.fc1.weight",
                    f"layers.{i}.mlp.fc1_bias": f"{p}.mlp.fc1.bias",
                    f"layers.{i}.mlp.fc2_weight": f"{p}.mlp.fc2.weight",
                    f"layers.{i}.mlp.fc2_bias": f"{p}.mlp.fc2.bias",
                }
            )
        return m

    def load_weights(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        *,
        cpu_mode: bool = True,
    ) -> None:
        """Load vision weights using the **vision** TP rank.

        This has to be separate from the top-level load_weights: that one passes
        the text TP rank, and the vision tower lives in its own TP group (default
        resolution is vision tp=1, dp=world_size). Feeding a text rank of 1..3 to
        a tp=1 vision loader walks the slice past the end of the tensor and
        silently returns a short shard.
        """
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_vision_tp_group,
        )
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tp_group = get_neuron_vision_tp_group()
        tp_rank = tp_group.rank_in_group if tp_group else 0
        tp_size = tp_group.world_size if tp_group else 1

        mappings = self.build_weight_mappings()
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        loader = (
            checkpoint.load_sharded if cpu_mode else checkpoint.load_sharded_pipelined
        )
        sd = loader(
            rank=tp_rank,
            world_size=tp_size,
            model=self,
            mappings=mappings,
            device=device,
        ).state_dict
        for name, tensor in sd.items():
            if tensor.dtype != self.dtype:
                sd[name] = tensor.to(self.dtype)
        self.load_state_dict(sd, strict=False, assign=True)
