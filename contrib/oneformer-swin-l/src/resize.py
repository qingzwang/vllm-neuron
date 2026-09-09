# SPDX-License-Identifier: Apache-2.0
"""Exact bilinear resize for the two ratios OneFormer actually uses.

``F.interpolate(..., mode="bilinear", align_corners=False)`` compiles here, so this is
not a correctness patch -- it is a cost patch. The generic lowering builds a *dense
resample matrix* and does the resize as a matmul. In the profile of the 241 ms forward
one such op, ``%dot.2``, carries ``load_weight_bytes = 1,327,104`` fp32 elements
= ``9216 x 144`` = ``(96*96) x (12*12)``: the mask logits at 96x96 resampled to 12x12,
as a 9216x144 matrix. It costs **10.0 ms and 1.735 GB of spill** -- 24% of the whole
graph's spill traffic -- for what is arithmetic on four strided reads.

The saving grace is that every ratio in this model is an exact power of two, and at
those ratios ``align_corners=False`` puts every destination sample exactly halfway
between source pixels, so the interpolation collapses to fixed weights:

**Downsampling by an even factor f.** Destination pixel ``j`` reads source coordinate
``(j + 0.5) * f - 0.5 = j*f + (f-1)/2``, and for even ``f`` that is a half-integer.
So it always lands between source pixels ``j*f + f/2 - 1`` and ``j*f + f/2`` with
weights 0.5 / 0.5, and in 2D between four of them with weights 0.25. Note this is
*not* an area average: for ``f = 8`` only 4 of each 8x8 block's 64 pixels contribute.
That looks wrong and is nevertheless exactly what bilinear interpolation does --
matching ``F.interpolate`` is the whole point, not improving on it.

**Upsampling by 2.** Destination ``j`` reads ``j/2 - 0.25``, which for even ``j`` is
``k - 0.25`` (weights 0.25 / 0.75 on ``k-1``, ``k``) and for odd ``j`` is ``k + 0.25``
(weights 0.75 / 0.25 on ``k``, ``k+1``). At the borders the out-of-range neighbour is
replaced by the edge pixel, which reproduces ``F.interpolate``'s clamping: the two
weights then land on the same value and sum to one.

:func:`fixed_bilinear_resize` is the dispatcher, and it falls back to ``F.interpolate``
for any ratio it cannot do exactly. That fallback is deliberate: this module is
installed by a source transform into upstream's own code, and a size it has not been
reasoned about must keep working rather than quietly resample differently.

Both formulations come from the OneFormer ConvNeXt-XL Neuron port
(``xniwangaws/NeuronStuff``, branch ``oneformer-convnext-xl-trn2``), whose notes
report fixed resize operators as a material part of its result -- not a rounding error
next to its NKI kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fixed_bilinear_upsample_2x(x: torch.Tensor) -> torch.Tensor:
    """Exactly ``F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)``.

    Args:
        x: ``(N, C, H, W)``.

    Returns:
        ``(N, C, 2H, 2W)``.
    """
    # Shifted copies with the edge repeated, which is what clamping the source
    # coordinate amounts to.
    left = torch.cat((x[..., :1], x[..., :-1]), dim=-1)
    right = torch.cat((x[..., 1:], x[..., -1:]), dim=-1)
    even_x = 0.25 * left + 0.75 * x
    odd_x = 0.75 * x + 0.25 * right
    # Interleave the two half-phases along W: stack then flatten puts them in
    # (even, odd, even, odd, ...) order, which is the destination order.
    horizontal = torch.stack((even_x, odd_x), dim=-1).flatten(-2)

    top = torch.cat((horizontal[:, :, :1], horizontal[:, :, :-1]), dim=2)
    bottom = torch.cat((horizontal[:, :, 1:], horizontal[:, :, -1:]), dim=2)
    even_y = 0.25 * top + 0.75 * horizontal
    odd_y = 0.75 * horizontal + 0.25 * bottom
    return torch.stack((even_y, odd_y), dim=3).flatten(2, 3)


def fixed_bilinear_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Exactly ``F.interpolate(x, scale_factor=1/factor, ...)`` for even ``factor``.

    Args:
        x: ``(N, C, H, W)``, with ``H`` and ``W`` divisible by ``factor``.
        factor: a positive even integer.

    Returns:
        ``(N, C, H // factor, W // factor)``.
    """
    if factor <= 0 or factor % 2 != 0:
        raise ValueError(f"factor must be a positive even integer, got {factor}")
    offset = factor // 2 - 1
    return 0.25 * (
        x[:, :, offset::factor, offset::factor]
        + x[:, :, offset::factor, offset + 1 :: factor]
        + x[:, :, offset + 1 :: factor, offset::factor]
        + x[:, :, offset + 1 :: factor, offset + 1 :: factor]
    )


def fixed_bilinear_resize(x: torch.Tensor, size) -> torch.Tensor:
    """Resize ``(N, C, H, W)`` to ``size``, exactly, whenever the ratio allows it.

    Handles the identity, even-integer reduction, and doubling (applied repeatedly for
    4x, 8x, ...). Anything else -- a non-integer ratio, an odd factor, a factor that
    does not divide the input -- goes to ``F.interpolate`` unchanged.
    """
    height, width = int(x.shape[-2]), int(x.shape[-1])
    target_h, target_w = (int(v) for v in size)

    if (target_h, target_w) == (height, width):
        return x

    if (
        target_h < height
        and height % target_h == 0
        and width % target_w == 0
        and height // target_h == width // target_w
        and (height // target_h) % 2 == 0
    ):
        return fixed_bilinear_downsample(x, height // target_h)

    # Doubling only. Doubling twice is *not* the same function as a single 4x upsample
    # -- destination 2 of a 4x resize reads 0.875/0.125 where two doublings give
    # 0.8125/0.1875 -- so a 4x request has to fall back rather than be composed.
    if (target_h, target_w) == (2 * height, 2 * width):
        return fixed_bilinear_upsample_2x(x)

    return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
