# SPDX-License-Identifier: Apache-2.0
"""Multi-scale deformable attention through the NKI library kernel.

``src/bilinear.py`` does this attention in plain PyTorch, and it works: the whole model
runs and matches CPU. It is also where the time goes. In the profile of the 241 ms
forward, **83 ms is one gather** -- the four-corner gather in :func:`bilinear.bilinear_sample`,
executed 6 layers x 3 levels x 4 corners = 72 times, each one reading 24 non-contiguous
elements per output element. No compiler flag moves it: across the whole ``--optlevel``
and DGE sweep the gather stayed at 83.0-84.0 ms, which is what says the fix has to be a
different *algorithm*, not a different build.

The algorithm exists and ships in this venv:
``nkilib.experimental.deformable_attention.ms_deformable_attention`` is a hand-written
NKI kernel that does the whole thing -- coordinate scaling, the four corners, zeros
padding, the attention-weighted sum -- with indirect DMA gathers into SBUF and a PSUM
accumulation, in one operator. The OneFormer ConvNeXt-XL Neuron port
(``xniwangaws/NeuronStuff``, branch ``oneformer-convnext-xl-trn2``) uses exactly this
kernel, and its pixel decoder -- six of these layers, same as here -- comes in at 70 ms
for the whole decoder.

Two things had to be established before trusting it, because **every level in this model
is square, so a row/column mix-up would pass silently**:

* *Coordinate convention.* The kernel's docstring pseudocode names its axes so that
  ``sampling_locations[..., 0]`` is scaled by ``H_l``. The code does not: step 2 of
  ``ms_deformable_attention.py`` multiplies ``[..., 0]`` by ``float(W_l)`` and ``[..., 1]``
  by ``float(H_l)``, and step 6 forms the flat index as ``y * W_l + x``. So ``[..., 0]``
  is the column and ``[..., 1]`` is the row -- HuggingFace's convention exactly, and the
  same as the package's own ``ms_deformable_attention_torch.py`` reference. No swap.
* *Output layout.* ``(B, N_q, N_h * C_h)``, head-major, which is what
  :func:`bilinear.multi_scale_deformable_attention` returns. A drop-in.

``spatial_shapes`` is still ``[(H_l, W_l), ...]``, so it is only the *coordinates* that
read reversed. ``probe_device_ops.py`` has a deliberately non-square probe
(``ms_deform_attn_nki_hw``) that fails if either of those two conclusions is wrong.
"""

from __future__ import annotations

import os

import torch

# LNC ("logical NeuronCore") sharding. The kernel shards its query loop across the
# logical cores of one NeuronCore; trn1's NeuronCore-v2 is LNC1, trn2 is LNC2. The
# runtime's own variable is the honest source, and 1 is the right default here.
LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))

_CALLER = None


def _caller():
    """The kernel, wrapped as a torch higher-order op. Cached; safe to call per layer.

    Two wrappers exist for this. ``nki``'s own ``Kernel.__call__`` picks one
    automatically, but its torch path imports ``torch_neuronx.nki_hop``, and this venv
    has ``libtorch-neuronx-lite`` rather than ``torch-neuronx`` -- so the automatic path
    raises ``ModuleNotFoundError``. Going through ``libtorch_neuronx_lite`` directly is
    the same registration by hand.
    """
    global _CALLER
    if _CALLER is None:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        from nkilib.experimental.deformable_attention.ms_deformable_attention import (
            ms_deformable_attention,
        )

        _CALLER = wrap_nki(ms_deformable_attention)
    return _CALLER


def available() -> bool:
    """Whether the kernel and its torch wrapper can be imported at all."""
    try:
        _caller()
    except ImportError:
        return False
    return True


def sanitize_far_oob_samples(
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    spatial_shapes: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move samples that are far outside the feature map, without changing the result.

    A sample contributes nothing under zeros padding once *all four* of its bilinear
    corners are out of range, which for ``align_corners=False`` happens exactly when the
    pixel coordinate ``x * W_l - 0.5`` leaves ``(-1, W_l)`` -- i.e. when the normalized
    ``x`` leaves ``(-0.5/W_l, 1 + 0.5/W_l)``. Zeroing such a sample's attention weight
    and parking its coordinate at 0.5 is therefore an identity on the output, and it
    keeps the coordinate inside the range the kernel's indirect-DMA addressing can
    express. (At the two endpoints the bilinear weight is already 0, so the open
    interval is the right one.)

    Args:
        sampling_locations: ``(B, Q, heads, levels, points, 2)`` in [0, 1], ``(x, y)``.
        attention_weights: ``(B, Q, heads, levels, points)``.
        spatial_shapes: ``[(H_l, W_l), ...]``, Python ints.

    Returns:
        The same two tensors, with far-outside samples neutralized.
    """
    safe_locations = []
    safe_weights = []
    for level, (height, width) in enumerate(spatial_shapes):
        locations = sampling_locations[:, :, :, level]
        weights = attention_weights[:, :, :, level]
        x = locations[..., 0]
        y = locations[..., 1]
        valid = (
            (x > -0.5 / width)
            & (x < 1.0 + 0.5 / width)
            & (y > -0.5 / height)
            & (y < 1.0 + 0.5 / height)
        )
        safe_locations.append(
            torch.where(valid.unsqueeze(-1), locations, torch.full_like(locations, 0.5))
        )
        safe_weights.append(weights * valid.to(weights.dtype))
    return torch.stack(safe_locations, dim=3), torch.stack(safe_weights, dim=3)


def multi_scale_deformable_attention_nki(
    value: torch.Tensor,
    spatial_shapes: list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Same signature and same result as :func:`bilinear.multi_scale_deformable_attention`.

    Args:
        value: ``(B, sum(H_l * W_l), heads, head_dim)``.
        spatial_shapes: ``[(H_l, W_l), ...]``, Python ints.
        sampling_locations: ``(B, Q, heads, levels, points, 2)`` in [0, 1].
        attention_weights: ``(B, Q, heads, levels, points)``.

    Returns:
        ``(B, Q, heads * head_dim)``.
    """
    shapes = tuple((int(h), int(w)) for h, w in spatial_shapes)
    starts, offset = [], 0
    for height, width in shapes:
        starts.append(offset)
        offset += height * width

    safe_locations, safe_weights = sanitize_far_oob_samples(
        sampling_locations, attention_weights, shapes
    )
    # The kernel reads the coordinates in float32 whatever the values are: it floors them
    # to an index, and bfloat16 is exact only to 256, well under a 48x48 level's 2303.
    # Weights share the value dtype -- that is the buffer the kernel allocates for them.
    return _caller()[LNC](
        value,
        shapes,
        tuple(starts),
        safe_locations.to(torch.float32),
        safe_weights.to(value.dtype),
        value_layout="BLNC",
        sampling_locations_layout="BQHLP2",
        align_corners=False,
        padding_mode="zeros",
    )
