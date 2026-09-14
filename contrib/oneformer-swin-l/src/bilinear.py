# SPDX-License-Identifier: Apache-2.0
"""Bilinear sampling and multi-scale deformable attention, without ``grid_sample``.

``F.grid_sample`` does not survive this compiler: the graph compiles and then the
runtime aborts reading the result back (``Check failed: pjrt_data->buffer != nullptr
PjRt buffer is null in TransferFromDevice``). See ``probe_device_ops.py``. Since
OneFormer's pixel decoder is six layers of multi-scale deformable attention, and
deformable attention *is* bilinear sampling at learned offsets, there is no way
around implementing it.

What is implemented here is ``F.grid_sample(..., mode="bilinear",
padding_mode="zeros", align_corners=False)``, by construction -- the same algorithm, to
2.1e-05 on unit-scale values, which is where the two disagree about the order to multiply
four weights in and nothing more:

* the same coordinate convention -- ``align_corners=False`` puts pixel centres at
  ``(i + 0.5) / size`` in [0, 1], so a normalized ``g`` maps to
  ``((g + 1) * size - 1) / 2``;
* the same four neighbours and the same weights;
* the same zero padding: a neighbour outside the image contributes nothing, which is
  done by zeroing its *weight* rather than by clamping the value, so the gather can
  clamp its indices and stay in range.

Everything is arithmetic and one ``gather`` -- no data-dependent control flow, no
dynamic shapes, so it compiles as part of the surrounding graph.
"""

from __future__ import annotations

import functools

import torch


def bilinear_sample(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """``F.grid_sample(value, grid, "bilinear", "zeros", align_corners=False)``.

    Args:
        value: ``(N, C, H, W)``.
        grid: ``(N, Q, P, 2)`` normalized to [-1, 1], ``(x, y)`` last.

    Returns:
        ``(N, C, Q, P)``.
    """
    n, c, h, w = value.shape
    _, q, p, _ = grid.shape

    # Coordinate arithmetic runs in float32 whatever the model's dtype is. In bfloat16
    # only integers up to 256 are exact, so the flattened index `y * w + x` -- which
    # reaches 2303 for a 48x48 level -- rounds, and the gather then reads the wrong
    # element or goes out of bounds outright ("index 576 is out of bounds for dimension
    # 2 with size 576"). The *values* stay in the model's dtype; only the addressing is
    # promoted.
    coord_dtype = torch.float32
    grid = grid.to(coord_dtype)

    # Normalized -> pixel coordinates, align_corners=False.
    ix = ((grid[..., 0] + 1) * w - 1) / 2  # (N, Q, P)
    iy = ((grid[..., 1] + 1) * h - 1) / 2

    x0 = torch.floor(ix)
    y0 = torch.floor(iy)
    x1 = x0 + 1
    y1 = y0 + 1

    # Interpolation weights, before any clamping.
    wx1 = ix - x0
    wx0 = 1 - wx1
    wy1 = iy - y0
    wy0 = 1 - wy1

    def inside(x, y):
        return ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)).to(coord_dtype)

    corners = (
        (x0, y0, wx0 * wy0),
        (x1, y0, wx1 * wy0),
        (x0, y1, wx0 * wy1),
        (x1, y1, wx1 * wy1),
    )

    flat = value.reshape(n, c, h * w)
    out = value.new_zeros(n, c, q * p)
    for x, y, weight in corners:
        # Zeros padding lives in the weight; the index is clamped so the gather is
        # always in range. Both are needed: clamping alone would replicate edges.
        # Weights are computed in float32 with the coordinates and cast at the end, so
        # a bfloat16 model still gets bilinear weights worth the name.
        weight = (weight * inside(x, y)).reshape(n, 1, q * p).to(value.dtype)
        xc = x.clamp(0, w - 1)
        yc = y.clamp(0, h - 1)
        index = (yc * w + xc).reshape(n, 1, q * p).to(torch.int64).expand(n, c, q * p)
        out = out + weight * torch.gather(flat, 2, index)

    return out.reshape(n, c, q, p)


def bilinear_sample_packed(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """:func:`bilinear_sample`, bit-for-bit, in **one** gather instead of four.

    Same signature, same result -- the difference is entirely in how much DMA it asks
    for. :func:`bilinear_sample` issues four gathers at four computed indices, so one
    bilinear sample is four scattered reads; the profile says those are 3.7 M packets
    and 83 ms, a third of the whole forward, and that the cost is per *packet*.

    Here the 2x2 neighbourhood is materialized into the table instead::

        packed[p] = [ v(p), v(p+1), v(p+W), v(p+W+1) ]

    so one index fetches all four neighbours as four *contiguous* floats. The table is
    4x bigger and is built from four shifted slices of the original -- sequential traffic,
    which is the cheap kind.

    Measured on the whole model at 384x384: **1.34 M packets of 1612 B against 3.70 M of
    552 B, and 30.9 ms of dynamic DMA against 81.2** -- 2.8x fewer gathers rather than the
    4x the arithmetic suggests, because the four-corner version was already being partly
    coalesced. 230.5 ms to 143.1 ms end to end, at bit-identical output.

    At the default 640x640 the same change is worth more, and says the per-packet thing
    more plainly than the 384 numbers do: **5.050 GB in 7.02 M packets becomes 5.066 GB in
    2.79 M packets** -- slightly *more* bytes, 2.5x fewer packets -- and dynamic DMA goes
    from 216.1 ms to 75.4. 614.8 ms to 466.7 end to end, again bit-identical.

    Two details make it exact rather than approximate:

    * **A zero halo, not a clamp.** The obvious version clamps the base index, and it is
      wrong at the boundary: at ``x0 == -1`` the ``x1`` corner has a nonzero weight and
      must read column 0, while ``packed[clamp(x0) == 0]`` slot 1 holds column *1*. Padding
      the image with one ring of zeros first makes that case an honest read of an honest
      zero, which is what ``padding_mode="zeros"`` means anyway.
    * **The weights still carry the padding.** The index is clamped after the halo shift,
      but only where it cannot matter: a sample can reach a wrong element only once it is
      two or more pixels outside, and by then ``inside()`` has zeroed all four of its
      weights. That is the same argument :func:`bilinear_sample` relies on.
    """
    n, c, h, w = value.shape
    _, q, p, _ = grid.shape

    coord_dtype = torch.float32
    grid = grid.to(coord_dtype)
    ix = ((grid[..., 0] + 1) * w - 1) / 2
    iy = ((grid[..., 1] + 1) * h - 1) / 2
    x0 = torch.floor(ix)
    y0 = torch.floor(iy)
    wx1 = ix - x0
    wx0 = 1 - wx1
    wy1 = iy - y0
    wy0 = 1 - wy1

    # One ring of zeros, then a tail long enough that base + wp + 1 is always in range.
    hp, wp = h + 2, w + 2
    haloed = torch.nn.functional.pad(value, (1, 1, 1, 1)).reshape(n, c, hp * wp)
    padded = torch.nn.functional.pad(haloed, (0, wp + 1))
    packed = torch.stack(
        (
            padded[:, :, 0:hp * wp],
            padded[:, :, 1:hp * wp + 1],
            padded[:, :, wp:hp * wp + wp],
            padded[:, :, wp + 1:hp * wp + wp + 1],
        ),
        dim=-1,
    )  # (n, c, hp*wp, 4)

    # inside(x, y) is (x in range) & (y in range), and the four corners use only two
    # distinct x and two distinct y, so folding each axis's test into that axis's weight
    # computes 8 comparisons and 4 ands where the corner-at-a-time form needs 16 and 12.
    # Bit-for-bit equal, signed zeros included: every mask is exactly 0.0 or 1.0, so the
    # regrouping cannot round -- x * 1.0 is exact and any product containing 0.0 is a zero
    # of the same sign whichever order it is taken in.
    def in_range(v, hi):
        return ((v >= 0) & (v <= hi)).to(coord_dtype)

    wx0 = wx0 * in_range(x0, w - 1)
    wx1 = wx1 * in_range(x0 + 1, w - 1)
    wy0 = wy0 * in_range(y0, h - 1)
    wy1 = wy1 * in_range(y0 + 1, h - 1)

    weights = torch.stack(
        (wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1), dim=-1
    ).reshape(n, 1, q * p, 4).to(value.dtype)

    base = (y0 + 1).clamp(0, hp - 1) * wp + (x0 + 1).clamp(0, wp - 1)
    index = base.reshape(n, 1, q * p, 1).to(torch.int64).expand(n, c, q * p, 4)
    out = (torch.gather(packed, 2, index) * weights).sum(-1)
    return out.reshape(n, c, q, p)


SAMPLERS = {"corners": bilinear_sample, "packed": bilinear_sample_packed}


def _split_heads(fn, value, sampling_locations, attention_weights, heads, split):
    """Run ``fn`` on ``split`` groups of heads and concatenate, instead of all at once.

    Bit-for-bit identical -- not "numerically equivalent", identical. Every tensor here
    has a head axis, nothing in deformable attention reduces across it (the weighted sum
    is over levels and points, both inside a head), and the returned
    ``(B, Q, heads * head_dim)`` lays its heads out contiguously along the last axis, so
    ``cat`` puts the pieces back element for element. There is no reduction to
    reassociate, which is what separates this from
    ``patches._cross_attention_per_head``.

    What changes is how much is live. At 640x640 the 80x80 level's packed table is
    ``(8, 32, 82^2, 4)`` float32 -- **26.3 MiB** -- and the gather's result before the
    weighted sum is ``(8, 32, 8400 * 4, 4)``, **32.8 MiB**, both against a 24 MiB SBUF and
    both live together. Two groups halve them, eight groups leave 3.3 and 4.1 MiB.

    Splitting *here* rather than around the sampler is the whole design. The sampler runs
    18 times (six layers x three levels), so concatenating its output would add 18 copies
    of a 32.8 MiB tensor; splitting the whole attention concatenates once, on the
    ``(1, 8400, 256)`` result -- 8.6 MiB. Measured at ``split=2``, that placement is worth
    about 8 ms: +24.1 ms over its own baseline here against +32.3 for the per-sampler
    version.

    And it is still a loss. ``split=2`` is 451.77 ms against 427.72 at ``split=1``, and
    ``split=4`` and ``split=8`` do not compile at all -- ``neuronx-cc`` runs the 32 GB
    compile host out of memory on 72 and 144 gather sites. Kept, at a default of 1, so the
    negative result can be re-run at another size or on a part with more SBUF; see the
    README's "The same trick on the gather, measured and rejected".
    """
    if split <= 1 or heads <= 1:
        return None
    step = -(-heads // split)  # ceil, so any head count works; the last group is short
    return torch.cat(
        [
            fn(
                value=value[:, :, i:i + step],
                sampling_locations=sampling_locations[:, :, i:i + step],
                attention_weights=attention_weights[:, :, i:i + step],
            )
            for i in range(0, heads, step)
        ],
        dim=-1,
    )


def multi_scale_deformable_attention(
    value: torch.Tensor,
    spatial_shapes: list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    gather: str = "packed",
    split: int = 1,
) -> torch.Tensor:
    """The pixel decoder's attention, one level at a time.

    Mirrors ``transformers.models.oneformer.modeling_oneformer``'s pure-PyTorch
    version, with :func:`bilinear_sample` in place of ``F.grid_sample`` and the level
    sizes as Python ints so every shape is static.

    Args:
        value: ``(B, sum(H_l * W_l), heads, head_dim)``.
        spatial_shapes: ``[(H_l, W_l), ...]``, Python ints.
        sampling_locations: ``(B, Q, heads, levels, points, 2)`` in [0, 1].
        attention_weights: ``(B, Q, heads, levels, points)``.
        gather: ``"packed"`` for one gather per sample (:func:`bilinear_sample_packed`),
            ``"corners"`` for the original four. Identical results, bit for bit; the
            difference is 75.4 ms of dynamic DMA against 216.1 at 640x640, and 30.9
            against 81.2 at 384. Defaults to packed.
        split: how many groups of heads to run at a time rather than all eight at once.
            Bit-for-bit identical at any value, and it does shrink the live set -- at 640
            the packed table and its gathered result are 26.3 and 32.8 MiB against a 24 MiB
            SBUF -- but measured it is slower, and above 2 it does not compile. Defaults to
            1, all heads in one pass. See :func:`_split_heads` for the numbers.

    Returns:
        ``(B, Q, heads * head_dim)``.
    """
    sample = SAMPLERS[gather]
    batch, _, heads, head_dim = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    assert levels == len(spatial_shapes)

    grouped = _split_heads(
        functools.partial(
            multi_scale_deformable_attention, spatial_shapes=spatial_shapes,
            gather=gather, split=1,
        ),
        value, sampling_locations, attention_weights, heads, split,
    )
    if grouped is not None:
        return grouped

    splits = [h * w for h, w in spatial_shapes]
    value_levels = value.split(splits, dim=1)
    # grid_sample's convention is [-1, 1]; the caller's locations are in [0, 1].
    grids = sampling_locations * 2 - 1

    sampled = []
    for level, (h, w) in enumerate(spatial_shapes):
        # (B, H*W, heads, head_dim) -> (B*heads, head_dim, H, W)
        v = (
            value_levels[level]
            .permute(0, 2, 3, 1)
            .reshape(batch * heads, head_dim, h, w)
        )
        g = grids[:, :, :, level].transpose(1, 2).reshape(batch * heads, queries, points, 2)
        sampled.append(sample(v, g))

    # (B*heads, head_dim, Q, levels, points) weighted by (B*heads, 1, Q, levels, points)
    stacked = torch.stack(sampled, dim=-2)
    weights = (
        attention_weights.permute(0, 2, 1, 3, 4)
        .reshape(batch * heads, 1, queries, levels, points)
    )
    out = (stacked * weights).sum(dim=(-2, -1))  # (B*heads, head_dim, Q)
    return out.reshape(batch, heads * head_dim, queries).transpose(1, 2)
