# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for DeepSeek-V4 checkpoints.

The DeepSeek-V4 checkpoint stores most linear weights in FP8 (e4m3) with
per-block ``ue8m0`` scales, and the routed MoE experts in FP4 (e2m1, two values
packed per byte) with per-32 ``ue8m0`` scales along the input dim.

Trainium has no FP4 datapath, so every quantized weight is dequantized to bf16
during checkpoint load. That keeps the compute graph uniform bf16 at the cost of
HBM footprint — see the model README for the resulting memory budget.

All loaders here return bf16 and compose with the generic sharding utilities in
:mod:`vllm_neuron.utils.weight_loader`: the dequant happens first (on the
checkpoint slice this rank needs), then the shard is taken.
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

# float4_e2m1 representable values, indexed by the 4-bit pattern.
# Bit layout: [sign][exp:2][mantissa:1].
_E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

# Default block sizes. FP8 weights use a 2-D [128, 128] block grid; FP4 expert
# weights use a 1-D per-32 grid along the input (reduction) dim.
FP8_BLOCK = (128, 128)
FP4_BLOCK_SIZE = 32


def _e2m1_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_E2M1_VALUES, device=device, dtype=dtype)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a byte tensor holding two float4_e2m1 values per byte.

    The low nibble holds the even (lower-index) element along the packed
    dimension, matching ``torch.float4_e2m1fn_x2`` storage order.

    Args:
        packed: Byte tensor of shape ``[..., N // 2]``. Accepts int8, uint8, or
            ``torch.float4_e2m1fn_x2`` (all are bit-reinterpreted as uint8).

    Returns:
        float32 tensor of shape ``[..., N]``.
    """
    as_bytes = packed.view(torch.uint8)
    table = _e2m1_table(as_bytes.device, torch.float32)
    lo = table[(as_bytes & 0x0F).long()]
    hi = table[((as_bytes >> 4) & 0x0F).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _expand_scale(
    scale: torch.Tensor, shape: tuple[int, ...], block: tuple[int, ...]
) -> torch.Tensor:
    """Repeat a per-block scale grid up to ``shape``.

    The final block along each dim may be ragged (the checkpoint rounds the
    scale grid up), so the expanded tensor is cropped rather than assumed exact.
    """
    expanded = scale.to(torch.float32)
    for dim, block_size in enumerate(block):
        expanded = expanded.repeat_interleave(block_size, dim=dim)
    crop = tuple(slice(0, s) for s in shape)
    return expanded[crop]


def dequant_fp8_blockwise(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int] = FP8_BLOCK,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a 2-D FP8 weight with a per-``block`` scale grid.

    Args:
        weight: FP8 (e4m3) tensor of shape ``[out, in]``.
        scale: ``[ceil(out / block[0]), ceil(in / block[1])]`` scale grid.
        block: Scale block size as ``(out_block, in_block)``.
        out_dtype: Compute dtype to return.
    """
    if weight.ndim != 2:
        raise ValueError(f"FP8 blockwise dequant expects a 2-D weight, got {weight.shape}")
    values = weight.to(torch.float32)
    scales = _expand_scale(scale, tuple(weight.shape), block)
    return (values * scales).to(out_dtype)


def dequant_fp4_blockwise(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = FP4_BLOCK_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a packed FP4 expert weight.

    Args:
        weight: Packed tensor of shape ``[out, in // 2]``.
        scale: ``[out, ceil(in / block_size)]`` scale grid.
        block_size: Elements per scale along the input dim.
        out_dtype: Compute dtype to return.

    Returns:
        ``[out, in]`` tensor in ``out_dtype``.
    """
    values = unpack_fp4(weight)
    if values.ndim != 2:
        raise ValueError(f"FP4 blockwise dequant expects a 2-D weight, got {weight.shape}")
    scales = _expand_scale(scale, tuple(values.shape), (1, block_size))
    return (values * scales).to(out_dtype)


def _slice_along(
    slice_obj, dim: int, start: int, stop: int, ndim: int
) -> torch.Tensor:
    """Read ``slice_obj[..., start:stop, ...]`` along ``dim`` from a safetensors slice."""
    key = [slice(None)] * ndim
    key[dim] = slice(start, stop)
    return slice_obj[tuple(key)]


def fp8_dequant_weight_loader(
    shard_dim: int | None = None,
    shard_size: int | None = None,
    num_shards: int = 1,
    block: tuple[int, int] = FP8_BLOCK,
    is_storage_transposed: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> SafetensorsWeightLoader:
    """Loader for an FP8 weight + its ``ue8m0`` scale, dequantized to bf16.

    Expects the mapping to supply exactly two checkpoint keys, in order:
    ``[<name>.weight, <name>.scale]``.

    Sharding is applied *after* dequant, on the parameter's own layout. Only the
    rank's own slice of the checkpoint is read: the weight is sliced along the
    storage dim, and the scale grid along the matching block-aligned dim, so a
    TP rank never materializes the full tensor.

    Args:
        shard_dim: Dim of the *parameter* to shard along, or None for replicated.
        shard_size: Size of each shard along ``shard_dim``.
        num_shards: Number of shards (TP degree).
        block: FP8 scale block size as ``(out_block, in_block)``.
        is_storage_transposed: True when the checkpoint stores ``[in, out]`` but
            the parameter is ``[out, in]`` (or vice versa).
        out_dtype: Compute dtype to return.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) != 2:
            raise ValueError(
                "fp8_dequant_weight_loader expects [weight, scale] slices, got "
                f"{len(slices)}"
            )
        weight_slice, scale_slice = slices

        if shard_dim is None:
            weight = weight_slice[:]
            scale = scale_slice[:]
            return _finish_fp8(weight, scale, block, is_storage_transposed, out_dtype)

        if shard_size is None:
            raise ValueError("shard_size is required when shard_dim is set")

        # Map the parameter's shard dim onto the checkpoint's storage dim.
        storage_dim = 1 - shard_dim if is_storage_transposed else shard_dim
        start = (rank % num_shards) * shard_size
        stop = start + shard_size

        block_size = block[storage_dim]
        if start % block_size or shard_size % block_size:
            # A shard boundary inside a scale block would need a partial scale
            # row; read the whole tensor and shard after dequant instead.
            weight = weight_slice[:]
            scale = scale_slice[:]
            dequantized = _finish_fp8(
                weight, scale, block, is_storage_transposed, out_dtype
            )
            key = [slice(None)] * dequantized.ndim
            key[shard_dim] = slice(start, stop)
            return dequantized[tuple(key)].contiguous()

        ndim = len(weight_slice.get_shape())
        weight = _slice_along(weight_slice, storage_dim, start, stop, ndim)
        scale = _slice_along(
            scale_slice,
            storage_dim,
            start // block_size,
            stop // block_size,
            len(scale_slice.get_shape()),
        )
        return _finish_fp8(weight, scale, block, is_storage_transposed, out_dtype)

    return SafetensorsWeightLoader(transform=transform)


def _finish_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int],
    is_storage_transposed: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    dequantized = dequant_fp8_blockwise(weight, scale, block, out_dtype)
    if is_storage_transposed:
        dequantized = dequantized.T
    return dequantized.contiguous()


def _load_fp4_expert_shard(
    weight_slice,
    scale_slice,
    *,
    shard_dim: int | None,
    shard_size: int | None,
    num_shards: int,
    block_size: int,
    transpose: bool,
    out_dtype: torch.dtype,
    rank: int,
) -> torch.Tensor:
    """Read one expert's FP4 shard, dequantizing only this rank's slice.

    Slicing the checkpoint *before* dequantizing matters at scale: an expert
    weight is 2048x4096, and at TP=64 a rank keeps 1/64 of it. Dequantizing the
    whole tensor first would materialize 64x more bf16 than needed on every rank
    simultaneously, which exhausts host memory on a large TP degree.

    The parameter layout is ``[out, in]`` before the optional transpose, matching
    the checkpoint. ``shard_dim`` indexes that layout:

    * ``shard_dim == 0`` slices the output rows — a plain row range in the
      packed tensor, and the same row range of the scale grid.
    * ``shard_dim == 1`` slices the input (reduction) dim, which is the packed
      one: two FP4 values share a byte, so the byte range is halved and the
      shard boundary must land on an even element. It also selects a
      ``block_size``-aligned range of scale columns.
    """
    if shard_dim is None:
        return _finish_fp4(weight_slice[:], scale_slice[:], block_size, transpose, out_dtype)

    start = (rank % num_shards) * shard_size
    stop = start + shard_size

    if shard_dim == 0:
        weight = weight_slice[start:stop]
        scale = scale_slice[start:stop]
        return _finish_fp4(weight, scale, block_size, transpose, out_dtype)

    if shard_dim != 1:
        raise ValueError(f"FP4 expert shard_dim must be 0 or 1, got {shard_dim}")

    if start % 2 or shard_size % 2:
        raise ValueError(
            f"FP4 packs two values per byte, so an input-dim shard must be "
            f"even-aligned; got start={start}, shard_size={shard_size}"
        )
    if start % block_size or shard_size % block_size:
        # A shard boundary inside a scale block would need a partial scale
        # column; fall back to dequantizing the whole expert and slicing.
        full = _finish_fp4(
            weight_slice[:], scale_slice[:], block_size, False, out_dtype
        )
        shard = full[:, start:stop]
        return (shard.T if transpose else shard).contiguous()

    weight = weight_slice[:, start // 2 : stop // 2]
    scale = scale_slice[:, start // block_size : stop // block_size]
    return _finish_fp4(weight, scale, block_size, transpose, out_dtype)


def _finish_fp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    transpose: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    dequantized = dequant_fp4_blockwise(weight, scale, block_size, out_dtype)
    if transpose:
        dequantized = dequantized.T
    return dequantized.contiguous()


def fp4_expert_dequant_loader(
    num_local_experts: int,
    shard_dim: int | None = None,
    shard_size: int | None = None,
    num_shards: int = 1,
    block_size: int = FP4_BLOCK_SIZE,
    transpose: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> SafetensorsWeightLoader:
    """Loader that stacks per-expert FP4 weights into one bf16 parameter.

    Expects the mapping to enumerate the local experts' checkpoint keys
    interleaved as ``[w_0, s_0, w_1, s_1, ...]`` — one ``(weight, scale)`` pair
    per expert, in expert order.

    Args:
        num_local_experts: Experts this rank owns (validates the slice count).
        shard_dim: Dim of the *per-expert* 2-D weight to shard along (0 = output
            / intermediate, 1 = input / hidden), or None for replicated.
        shard_size: Size of each shard along ``shard_dim``.
        num_shards: Number of shards (TP degree).
        block_size: Elements per FP4 scale along the input dim.
        transpose: Transpose each expert weight after dequant, so the parameter
            is ``[E, in, out]`` instead of ``[E, out, in]``.
        out_dtype: Compute dtype to return.

    Returns:
        Loader producing ``[num_local_experts, ...]`` stacked bf16 weights.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) != 2 * num_local_experts:
            raise ValueError(
                f"fp4_expert_dequant_loader expects {2 * num_local_experts} slices "
                f"([weight, scale] per expert), got {len(slices)}"
            )
        if shard_dim is not None and shard_size is None:
            raise ValueError("shard_size is required when shard_dim is set")

        experts = []
        for i in range(num_local_experts):
            experts.append(
                _load_fp4_expert_shard(
                    slices[2 * i],
                    slices[2 * i + 1],
                    shard_dim=shard_dim,
                    shard_size=shard_size,
                    num_shards=num_shards,
                    block_size=block_size,
                    transpose=transpose,
                    out_dtype=out_dtype,
                    rank=rank,
                )
            )

        return torch.stack(experts, dim=0).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def fp8_expert_dequant_loader(
    num_local_experts: int,
    shard_dim: int | None = None,
    shard_size: int | None = None,
    num_shards: int = 1,
    block: tuple[int, int] = FP8_BLOCK,
    transpose: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> SafetensorsWeightLoader:
    """Same as :func:`fp4_expert_dequant_loader` for FP8-quantized experts.

    The ``-Base`` checkpoints keep the routed experts in FP8 rather than FP4
    (``expert_dtype`` is unset in ``config.json``).
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) != 2 * num_local_experts:
            raise ValueError(
                f"fp8_expert_dequant_loader expects {2 * num_local_experts} slices "
                f"([weight, scale] per expert), got {len(slices)}"
            )
        if shard_dim is not None and shard_size is None:
            raise ValueError("shard_size is required when shard_dim is set")

        # As in the FP4 path, the checkpoint is sliced before dequantizing so a
        # rank never materializes bf16 for experts it does not own.
        experts = []
        for i in range(num_local_experts):
            weight_slice, scale_slice = slices[2 * i], slices[2 * i + 1]
            if shard_dim is None:
                experts.append(
                    _finish_fp8(weight_slice[:], scale_slice[:], block, transpose, out_dtype)
                )
                continue

            start = (rank % num_shards) * shard_size
            stop = start + shard_size
            block_size = block[shard_dim]
            if start % block_size or shard_size % block_size:
                full = _finish_fp8(
                    weight_slice[:], scale_slice[:], block, False, out_dtype
                )
                key = [slice(None)] * full.ndim
                key[shard_dim] = slice(start, stop)
                shard = full[tuple(key)]
                experts.append((shard.T if transpose else shard).contiguous())
                continue

            weight_key = [slice(None), slice(None)]
            weight_key[shard_dim] = slice(start, stop)
            scale_key = [slice(None), slice(None)]
            scale_key[shard_dim] = slice(start // block_size, stop // block_size)
            experts.append(
                _finish_fp8(
                    weight_slice[tuple(weight_key)],
                    scale_slice[tuple(scale_key)],
                    block,
                    transpose,
                    out_dtype,
                )
            )

        return torch.stack(experts, dim=0).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def cast_weight_loader(
    out_dtype: torch.dtype = torch.bfloat16,
    shard_dim: int | None = None,
    shard_size: int | None = None,
    num_shards: int = 1,
    is_storage_transposed: bool = False,
) -> SafetensorsWeightLoader:
    """Loader for an unquantized weight: optional shard, then dtype cast.

    Used for the bf16 compressor / router / norm weights and the fp32
    hyper-connection parameters, whose checkpoint dtype differs from the
    parameter dtype.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) != 1:
            raise ValueError(
                f"cast_weight_loader expects a single slice, got {len(slices)}"
            )
        slice_obj = slices[0]

        if shard_dim is None:
            result = slice_obj[:]
        else:
            if shard_size is None:
                raise ValueError("shard_size is required when shard_dim is set")
            storage_dim = 1 - shard_dim if is_storage_transposed else shard_dim
            start = (rank % num_shards) * shard_size
            result = _slice_along(
                slice_obj, storage_dim, start, start + shard_size,
                len(slice_obj.get_shape()),
            )

        if is_storage_transposed:
            result = result.T
        return result.to(out_dtype).contiguous()

    return SafetensorsWeightLoader(transform=transform)
