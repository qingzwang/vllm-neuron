# SPDX-License-Identifier: Apache-2.0
"""InternViT-300M vision encoder (BF16) for InternVL3.

PatchEmbed -> [CLS] + pos embed -> 24 x (LayerNorm, Attention, LayerScale,
LayerNorm, MLP, LayerScale) -> drop CLS.

Simpler than the Qwen3-VL encoder in one important way: InternVL's dynamic
tiling always emits **448x448 tiles**, so every item is exactly
``grid_size**2 = 1024`` patches. There is no variable-length packing and no
bin-packing across items — each tile is an independent sequence of a fixed
length, so the whole batch is just ``[num_tiles, 1 + 1024, hidden]`` and
attention runs per tile.

Attention goes through ``NF.flash_attention``. A naive
``softmax(q @ k^T) @ v`` here is correct but materialises
``[tiles, heads, s, s]`` — 67 MB per tile at vision tp=1, so over 1 GB per layer
at 16 tiles — and measured superlinear in the tile count even though the FLOPs
are linear (see BENCHMARK_REPORT.md). Attention bounds are used only to mask the
sequence padding that the kernel's tile alignment requires, not for packing.

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
import torch.nn.functional as F

import vllm_neuron.functional as NF
from vllm_neuron.parallel.neuron_parallel_state import get_neuron_vision_tp_group
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .config import InternVLConfig, InternVLVisionConfig
from .weight_loaders import (
    patch_embed_weight_loader,
    vis_qkv_bias_loader,
    vis_qkv_weight_loader,
)

# The sequence-packed attention kernel loads bounds in tiles of this size and reads
# ceil(s / align) whole tiles, so the attention sequence length must be a multiple
# of it or the NEFF bakes an out-of-bounds DMA that the runtime rejects at load.
# InternViT's s is 1 + grid_size**2 = 1025, so the tower always pads (see
# InternVisionModel.tower). Same constant and same reason as the Qwen3-VL encoder.
_ATTN_SEQ_ALIGN = 128


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

    Bidirectional and unmasked in substance — every tile attends within itself and
    the batch dim already separates tiles. The only masking is the attention
    bounds, which hide the sequence padding the kernel's tile alignment forces
    (see ``_ATTN_SEQ_ALIGN`` and ``InternVisionModel.tower``).
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """``[n, s, hidden] -> [n, s, hidden]``, attending within each tile.

        Args:
            hidden_states: ``[n, s, hidden]``, s already padded to
                ``_ATTN_SEQ_ALIGN``.
            bound_min: ``[n, s, 1]`` int32, inclusive lower KV bound per query.
            bound_max: ``[n, s, 1]`` int32, exclusive upper KV bound per query.
        """
        n, s, _ = hidden_states.shape
        h, d = self.num_heads_per_rank, self.head_dim

        qkv = torch.matmul(hidden_states, self.qkv_weight) + self.qkv_bias
        q, k, v = qkv.split(h * d, dim=-1)
        # The kernel folds heads into the batch dim: [n, s, h*d] -> [n*h, s, d].
        q = q.reshape(n, s, h, d).permute(0, 2, 1, 3).reshape(n * h, s, d)
        k = k.reshape(n, s, h, d).permute(0, 2, 1, 3).reshape(n * h, s, d)
        v = v.reshape(n, s, h, d).permute(0, 2, 1, 3).reshape(n * h, s, d)

        # Bounds are per tile; the kernel wants them per (tile, head) row.
        out = NF.flash_attention(
            q,
            k,
            v,
            scale=self.scale,
            # <-- MODEL-SPECIFIC: InternViT attention is bidirectional.
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=bound_min.repeat_interleave(h, dim=0),
            bound_max=bound_max.repeat_interleave(h, dim=0),
        )

        # [n*h, s, d] -> [n, s, h*d]
        out = out.reshape(n, h, s, d).permute(0, 2, 1, 3).reshape(n, s, h * d)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = (
            hidden_states
            + self.attn(self.norm1(hidden_states), bound_min, bound_max) * self.ls1
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2
        return hidden_states


class InternVisionModel(nn.Module):
    """InternViT tower **plus** the projector, with the encoder-cache write.

    Structured after ``Qwen3VLVisionModel``: the whole pixels-to-LLM-space path
    lives in one module so ``torch.compile`` covers it, and ``forward`` ends by
    scatter-writing into the on-device encoder cache from **inside** the graph.
    Keeping the projector outside and writing the buffer with Python indexing
    leaves those ops eager, and eager ops on Neuron compile one at a time.

    Two staged helpers are kept so the CPU validator can compare each stage
    against HF: ``tower()`` matches ``InternVisionModel`` output after the CLS
    drop, ``encode_tiles()`` matches ``extract_feature``.
    """

    def __init__(
        self, config: InternVLConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        from .projector import InternVLProjector

        self.full_config = config
        self.config = config.vision_config
        self.dtype = dtype
        self.embeddings = InternVisionPatchEmbed(self.config, dtype)
        self.layers = nn.ModuleList(
            [
                InternVisionEncoderLayer(self.config, dtype)
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        self.projector = InternVLProjector(config, dtype=dtype)

    def tower(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[tiles, 3, H, W]`` -> ``[tiles, num_patches, vit_hidden]``, CLS dropped."""
        hidden_states = self.embeddings(pixel_values)
        tiles, real_s, _ = hidden_states.shape

        # The attention kernel reads bounds in tiles of _ATTN_SEQ_ALIGN and covers
        # ceil(s / align) whole tiles, so an unaligned s bakes an out-of-bounds DMA
        # into the NEFF that the runtime rejects at load. s here is
        # 1 + grid_size**2 = 1025, which is never aligned, so always pad.
        padded_s = -(-real_s // _ATTN_SEQ_ALIGN) * _ATTN_SEQ_ALIGN
        if padded_s > real_s:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padded_s - real_s))

        # Every real query attends the whole real sequence: this is plain
        # bidirectional attention within a tile, and the bounds exist only to hide
        # the padding. Pad rows get bound_min == bound_max == 0 so they attend to
        # nothing, and real bound_max stops at real_s so nothing attends them.
        device = hidden_states.device
        positions = torch.arange(padded_s, device=device)
        is_real = (positions < real_s).to(torch.int32).reshape(1, padded_s, 1)
        bound_min = torch.zeros(tiles, padded_s, 1, dtype=torch.int32, device=device)
        bound_max = (is_real * real_s).expand(tiles, padded_s, 1).contiguous()

        for layer in self.layers:
            hidden_states = layer(hidden_states, bound_min, bound_max)

        # Drop the CLS token and the alignment padding; only real patch tokens feed
        # the projector.
        return hidden_states[:, 1:real_s, :]

    def encode_tiles(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[tiles, 3, H, W]`` -> ``[tiles, embed_per_tile, llm_hidden]``."""
        return self.projector(self.tower(pixel_values))

    def forward(
        self,
        pixel_values: torch.Tensor,
        encoder_cache_buffer: torch.Tensor,
        write_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tiles and scatter them into the encoder cache in-graph.

        Args:
            pixel_values: ``[num_tiles, 3, image_size, image_size]``. Static shape,
                set by the vision bucket: ``num_tiles = bucket / patches_per_tile``.
            encoder_cache_buffer: ``[num_cache_blocks, cache_block_size, fat_dim]``,
                written in place (input-output alias).
            write_block_ids: ``[num_write_blocks]`` int64, cache block index for
                each block of this call's output.

        Returns:
            The cache buffer, aliased.
        """
        embeds = self.encode_tiles(pixel_values)
        tiles, per_tile, fat_dim = embeds.shape
        cache_block_size = encoder_cache_buffer.shape[1]

        # Regroup [tiles, per_tile, fat] into cache-block layout. With the default
        # vision_attention_block_size (== patches_per_tile) these coincide: one
        # tile is exactly one cache block.
        total = tiles * per_tile
        if total % cache_block_size != 0:
            raise ValueError(
                f"{tiles} tiles x {per_tile} tokens = {total} does not divide the "
                f"cache block size {cache_block_size}"
            )
        # index_put_ needs a freshly laid-out value tensor: feeding it a view with
        # the strides left over from pixel_shuffle's permutes makes the XLA lowering
        # fail with "Input dimension should be either 1 or equal to the output
        # dimension it is broadcasting into". qwen3_vl gets this for free because
        # its fat tensor comes out of torch.cat.
        blocks = embeds.to(encoder_cache_buffer.dtype).reshape(
            total // cache_block_size, cache_block_size, fat_dim
        ).contiguous()
        encoder_cache_buffer.index_put_((write_block_ids,), blocks)
        return encoder_cache_buffer

    def build_weight_mappings(
        self, prefix: str = "vision_model", projector_prefix: str = "mlp1"
    ) -> dict[str, str | list[str]]:
        """Parameter name -> checkpoint key(s), tower and projector together.

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
        for name, key in self.projector.build_weight_mappings(projector_prefix).items():
            m[f"projector.{name}"] = key
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
