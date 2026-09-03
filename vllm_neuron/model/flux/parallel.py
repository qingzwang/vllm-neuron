# SPDX-License-Identifier: Apache-2.0
"""Tensor parallelism for the FLUX transformer.

The transformer is the whole cost of a request -- ~30 invocations against one for
everything else -- and on a single core it is also 15.2 GiB of weights against a
~22 GiB HBM partition. Sharding it across cores cuts both.

Only two things are actually split: attention heads and the feed-forward
intermediate dimension. The residual stream stays full width and identical on
every rank, so norms, modulation projections, embedders and the final `proj_out`
are left replicated and no code outside this module has to know that TP is on.
That also means the attention processor needs no changes at all: it derives the
head count from the tensor it is given (`unflatten(-1, (-1, head_dim))`), so a
column-parallel `to_q` simply hands it fewer heads.

Which layer becomes what:

| Layer | Split | Why |
|---|---|---|
| `attn.to_q/to_k/to_v`, `attn.add_*_proj` | column | one contiguous group of heads per rank |
| `attn.to_out.0`, `attn.to_add_out` | row | input is that rank's heads; sum over ranks |
| `ff.net.0.proj`, `ff_context.net.0.proj` | column | intermediate dim |
| `ff.net.2`, `ff_context.net.2` | row | back to hidden dim |
| single block `proj_mlp` | column | intermediate dim |
| single block `proj_out` | row, in two halves | see below |
| everything else | replicated | operates on the full-width residual stream |

The single-stream block is the one place the standard pattern does not fit. Its
`proj_out` consumes `cat([attn_output, mlp_hidden_states], dim=-1)`, and with the
two producers column-parallel each rank holds
`[attn_dim/tp + mlp_dim/tp]` -- which is *not* the rank's contiguous slice of the
global `[attn_dim + mlp_dim]` input, so a plain row-parallel linear would multiply
the wrong weights. `_SingleBlockProjOut` slices the weight into its attention and
MLP halves, shards each along its own axis, and sums the two partial products
before a single all-reduce. Exact, and one collective rather than two.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

from transformers.models.t5.modeling_t5 import T5Attention

from vllm_neuron.envs import is_native_backend
from vllm_neuron.nn import ColumnParallelLinear

logger = logging.getLogger(__name__)

_NATIVE = is_native_backend()


def _all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    """Sum ``x`` across the TP group, the same way ``RowParallelLinear`` does."""
    if _NATIVE:
        from torch.distributed._functional_collectives import all_reduce

        return all_reduce(x, reduceOp="sum", group=group)
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return x


def tp_world(tp_group=None) -> tuple[int, int, object]:
    """Return ``(tp_size, tp_rank, tp_group)``, falling back to a single rank.

    Args:
        tp_group: Process group to shard over, or None for the default world.

    Returns:
        Size, this process's rank, and the group to pass to collectives.
    """
    if not dist.is_initialized():
        return 1, 0, None
    group = tp_group if tp_group is not None else dist.group.WORLD
    return dist.get_world_size(group), dist.get_rank(group), group


class RowParallelLinear(nn.Module):
    """Row-parallel linear that keeps its bias in the layer's own dtype.

    ``vllm_neuron.nn.RowParallelLinear`` rejects a non-float32 bias, and every
    row-parallel layer in FLUX has a BF16 bias. The bias is not a distributed
    quantity -- it is added once, after the reduce -- so holding it here in the
    checkpoint's dtype changes nothing about the collective.

    Args:
        linear: The dense layer being replaced. Its weights are consumed.
        tp_group: Process group to shard over, or None for the default world.
    """

    def __init__(self, linear: nn.Linear, tp_group=None) -> None:
        super().__init__()
        self.tp_size, tp_rank, self.tp_group = tp_world(tp_group)
        in_features = linear.in_features
        if in_features % self.tp_size:
            raise ValueError(
                f"row-parallel input {in_features} is not divisible by tp_size "
                f"{self.tp_size}"
            )
        shard = in_features // self.tp_size
        start = tp_rank * shard
        self.in_features_per_rank = shard
        self.out_features = linear.out_features
        self.weight = nn.Parameter(
            linear.weight.data[:, start : start + shard].clone(), requires_grad=False
        )
        # Set by lora.wrap_with_lora when adapters are configured.
        self.lora_A = None
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, None)
        if self.lora_A is not None:
            # Before the reduce, deliberately: x is sharded so this rank can only
            # compute a partial A @ x, and lora_B is replicated, so the delta has to
            # ride the same all-reduce as the base. Added after it instead, every
            # rank would end up with a different wrong answer.
            from .lora import lora_delta

            out = out + lora_delta(self, x)
        if self.tp_size > 1:
            out = _all_reduce(out, self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out


class _SingleBlockProjOut(nn.Module):
    """The single-stream block's ``proj_out``, sharded along both its halves.

    Its input is ``cat([attn_output, mlp_hidden_states], dim=-1)``. Because both
    producers are column-parallel, the concatenation this layer receives is
    ``[attn_dim/tp + mlp_dim/tp]``, so the weight is split at ``attn_dim``, each
    half is sharded along its own axis, and the two partial products are summed
    before one all-reduce.

    Args:
        linear: The dense ``proj_out`` being replaced.
        attn_dim: Width of the attention half of its input, i.e. the model's
            hidden size.
        tp_group: Process group to shard over, or None for the default world.
    """

    def __init__(self, linear: nn.Linear, attn_dim: int, tp_group=None) -> None:
        super().__init__()
        self.tp_size, tp_rank, self.tp_group = tp_world(tp_group)
        mlp_dim = linear.in_features - attn_dim
        if mlp_dim <= 0:
            raise ValueError(
                f"proj_out input {linear.in_features} is not wider than the "
                f"attention half {attn_dim}; this is not a single-stream block"
            )
        for name, dim in (("attention", attn_dim), ("mlp", mlp_dim)):
            if dim % self.tp_size:
                raise ValueError(
                    f"proj_out {name} half {dim} is not divisible by tp_size "
                    f"{self.tp_size}"
                )
        self.attn_shard = attn_dim // self.tp_size
        self.mlp_shard = mlp_dim // self.tp_size
        mlp_shard = self.mlp_shard
        self.out_features = linear.out_features
        # Set by lora.wrap_with_lora when adapters are configured. lora_A covers the
        # attention half, lora_A_mlp the MLP half; lora_B is shared by both.
        self.lora_A = None
        self.lora_A_mlp = None

        attn_start = tp_rank * self.attn_shard
        mlp_start = attn_dim + tp_rank * mlp_shard
        self.weight_attn = nn.Parameter(
            linear.weight.data[:, attn_start : attn_start + self.attn_shard].clone(),
            requires_grad=False,
        )
        self.weight_mlp = nn.Parameter(
            linear.weight.data[:, mlp_start : mlp_start + mlp_shard].clone(),
            requires_grad=False,
        )
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_part = x[..., : self.attn_shard]
        mlp_part = x[..., self.attn_shard :]
        out = F.linear(attn_part, self.weight_attn, None) + F.linear(
            mlp_part, self.weight_mlp, None
        )
        if self.lora_A is not None:
            # One adapter, two halves, one reduce -- see RowParallelLinear.forward
            # for why the delta goes in before it. lora_B is shared by the halves,
            # so their contributions are summed here first.
            from .lora import lora_delta_split

            out = out + lora_delta_split(self, attn_part, mlp_part)
        if self.tp_size > 1:
            out = _all_reduce(out, self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out


def _column(linear: nn.Linear, tp_group=None) -> ColumnParallelLinear:
    """Replace ``linear`` with its column-parallel shard of the same weights."""
    layer = ColumnParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        gather_output=False,
        dtype=linear.weight.dtype,
        tp_group=tp_group,
    )
    # ColumnParallelLinear._load_from_state_dict takes the rank's slice of a
    # full-size weight, so the dense layer's own state dict is what to hand it.
    layer.load_state_dict(linear.state_dict())
    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


def _set(parent: nn.Module, name: str, module: nn.Module) -> None:
    """Assign ``module`` at ``name`` on ``parent``, supporting ``a.b.0`` paths."""
    *path, leaf = name.split(".")
    for part in path:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def shard_flux_transformer(
    transformer: FluxTransformer2DModel, tp_group=None
) -> int:
    """Shard a loaded FLUX transformer across a tensor-parallel group, in place.

    Safe to call with a single-rank group: every layer is then replaced by an
    equivalent unsharded one, which is wasteful but correct, so callers do not
    need a TP=1 special case.

    Args:
        transformer: A loaded diffusers ``FluxTransformer2DModel``. Its dense
            layers are replaced and their weights consumed.
        tp_group: Process group to shard over, or None for the default world.

    Returns:
        Number of layers replaced.

    Raises:
        ValueError: If the group is not a power of two, or if a dimension does not
            divide evenly across it. With 24 heads and a hidden size of 3072, the
            degrees that divide are 1, 2, 4 and 8. Three would divide the heads
            too, but Neuron's replica groups -- and every TP degree the LLM path
            supports -- are powers of two, so it is rejected here rather than left
            to fail deeper in the stack.
    """
    tp_size, _, tp_group = tp_world(tp_group)
    if tp_size & (tp_size - 1):
        raise ValueError(f"tp_size must be a power of two, got {tp_size}")

    attn_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    heads = transformer.config.num_attention_heads
    if heads % tp_size:
        raise ValueError(
            f"{heads} attention heads do not divide across tp_size {tp_size}"
        )

    replaced = 0
    for block in transformer.transformer_blocks:
        for name in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "ff.net.0.proj",
            "ff_context.net.0.proj",
        ):
            _set(block, name, _column(block.get_submodule(name), tp_group))
            replaced += 1
        for name in ("attn.to_out.0", "attn.to_add_out", "ff.net.2", "ff_context.net.2"):
            _set(block, name, RowParallelLinear(block.get_submodule(name), tp_group))
            replaced += 1

    for block in transformer.single_transformer_blocks:
        for name in ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"):
            _set(block, name, _column(block.get_submodule(name), tp_group))
            replaced += 1
        _set(block, "proj_out", _SingleBlockProjOut(block.proj_out, attn_dim, tp_group))
        replaced += 1

    logger.info(
        "FLUX transformer sharded over %d rank(s): %d layers replaced, "
        "%d of %d heads per rank",
        tp_size,
        replaced,
        heads // tp_size,
        heads,
    )
    return replaced


def shard_t5_encoder(encoder: nn.Module, tp_group=None) -> int:
    """Shard a T5 encoder across a tensor-parallel group, in place.

    Same idea as the transformer -- heads and the feed-forward width -- with two
    T5 specifics:

    * ``T5Attention`` reshapes with ``self.n_heads`` and ``self.inner_dim`` rather
      than inferring them from the tensor, so both have to be reduced to this
      rank's share after the projections are sharded.
    * The relative-attention bias is an ``Embedding(buckets, heads)`` that only
      block 0 owns; it is computed once there and threaded through every later
      block as ``position_bias``. Sharding it along the head axis is what keeps
      that consistent with the sharded attention -- every block then sees exactly
      its own heads' biases.

    T5-XXL has no biases anywhere, so nothing needs the after-reduce bias path.

    Args:
        encoder: A loaded ``T5EncoderModel`` (or its ``.encoder``). Its dense
            layers are replaced and their weights consumed.
        tp_group: Process group to shard over, or None for the default world.

    Returns:
        Number of layers replaced.

    Raises:
        ValueError: If the head count or feed-forward width does not divide.
    """
    tp_size, tp_rank, tp_group = tp_world(tp_group)
    if tp_size == 1:
        return 0

    replaced = 0
    for attn in (m for m in encoder.modules() if isinstance(m, T5Attention)):
        if attn.n_heads % tp_size:
            raise ValueError(
                f"{attn.n_heads} T5 heads do not divide across tp_size {tp_size}"
            )
        for name in ("q", "k", "v"):
            _set(attn, name, _column(getattr(attn, name), tp_group))
            replaced += 1
        _set(attn, "o", RowParallelLinear(attn.o, tp_group))
        replaced += 1

        if attn.has_relative_attention_bias:
            # [buckets, heads] -> this rank's heads.
            weight = attn.relative_attention_bias.weight.data
            heads = weight.shape[1] // tp_size
            start = tp_rank * heads
            local = nn.Embedding(weight.shape[0], heads)
            local.weight = nn.Parameter(
                weight[:, start : start + heads].clone(), requires_grad=False
            )
            _set(attn, "relative_attention_bias", local)
            replaced += 1

        attn.n_heads = attn.n_heads // tp_size
        attn.inner_dim = attn.n_heads * attn.key_value_proj_dim

    for block in encoder.block if hasattr(encoder, "block") else encoder.encoder.block:
        ff = block.layer[-1].DenseReluDense
        for name in ("wi_0", "wi_1") if hasattr(ff, "wi_0") else ("wi",):
            _set(ff, name, _column(getattr(ff, name), tp_group))
            replaced += 1
        _set(ff, "wo", RowParallelLinear(ff.wo, tp_group))
        replaced += 1

    logger.info(
        "T5 encoder sharded over %d ranks: %d layers replaced", tp_size, replaced
    )
    return replaced
