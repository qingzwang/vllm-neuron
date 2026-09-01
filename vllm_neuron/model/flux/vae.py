# SPDX-License-Identifier: Apache-2.0
"""Neuron adaptations for the FLUX VAE decoder.

Two changes are needed to run ``AutoencoderKL``'s decoder on Neuron; both are
about how it lowers, not about the math.

**Staging.** The full 1024x1024 decode lowers to ~10M machine instructions,
against ``neuronx-cc``'s ~5M budget, and the compiler's own advice is to shrink
the graph. Decode is therefore compiled as one graph per resolution level:
denorm + ``conv_in`` + mid block, then one graph per upsampling block. Stages
hand activations to each other on-device, so staging costs nothing at runtime.

**Nearest-neighbour upsampling.** ``F.interpolate(..., mode="nearest")`` lowers
to an indirect (scatter/gather) memory copy that the runtime rejects at these
sizes with an out-of-bounds access. For an exact 2x factor the same result is a
broadcast plus a reshape, which lowers to straight copies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.upsampling import Upsample2D


@torch.compiler.allow_in_graph
def upsample_nearest_2x(x: torch.Tensor) -> torch.Tensor:
    """Exact 2x nearest-neighbour upsample of an NCHW tensor.

    Equivalent to ``F.interpolate(x, scale_factor=2.0, mode="nearest")``,
    expressed as broadcast + reshape so it lowers to copies instead of an
    indirect gather.
    """
    batch, channels, height, width = x.shape
    return (
        x.reshape(batch, channels, height, 1, width, 1)
        .expand(batch, channels, height, 2, width, 2)
        .reshape(batch, channels, height * 2, width * 2)
    )


class NeuronUpsample2D(nn.Module):
    """``Upsample2D`` with the interpolation swapped for :func:`upsample_nearest_2x`.

    Wraps the original module rather than replacing it so the convolution
    weights keep their identity and their place in the parameter tree.

    Args:
        original: The ``Upsample2D`` being replaced.

    Raises:
        NotImplementedError: If the original uses a transposed convolution, a
            norm, or no interpolation at all. The FLUX VAE decoder uses none of
            those; anything else needs its own lowering check.
    """

    def __init__(self, original: Upsample2D) -> None:
        super().__init__()
        if original.use_conv_transpose or original.norm is not None:
            raise NotImplementedError(
                "NeuronUpsample2D only handles the interpolate+conv form of "
                "Upsample2D used by the FLUX VAE decoder."
            )
        if not original.interpolate:
            raise NotImplementedError(
                "NeuronUpsample2D expects an interpolating Upsample2D."
            )
        self.original = original

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if output_size is not None:
            raise NotImplementedError(
                "NeuronUpsample2D only supports the default 2x scale factor, "
                f"got output_size={output_size}."
            )
        original = self.original
        hidden_states = upsample_nearest_2x(hidden_states)
        if original.use_conv:
            conv = original.conv if original.name == "conv" else original.Conv2d_0
            hidden_states = conv(hidden_states)
        return hidden_states


def patch_upsampling(module: nn.Module) -> int:
    """Replace every ``Upsample2D`` under ``module`` in place.

    Args:
        module: Subtree to rewrite, typically the VAE.

    Returns:
        Number of modules replaced.
    """
    # Snapshot before mutating: the replacement keeps the original as a child,
    # so walking and rewriting at the same time would re-wrap its own output.
    targets = [
        (parent, name)
        for parent in list(module.modules())
        for name, child in parent.named_children()
        if isinstance(child, Upsample2D)
    ]
    for parent, name in targets:
        setattr(parent, name, NeuronUpsample2D(getattr(parent, name)))
    return len(targets)


class VaeDecodeHead(nn.Module):
    """Stage 0: latent denormalization, ``conv_in``, and the decoder mid block.

    The mid block carries a spatial self-attention over the whole latent grid --
    16384 positions for a 1024x1024 image -- which makes this the most expensive
    decode stage.
    """

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae
        self.scaling_factor = vae.config.scaling_factor
        self.shift_factor = vae.config.shift_factor

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = latents / self.scaling_factor + self.shift_factor
        if getattr(self.vae, "post_quant_conv", None) is not None:
            z = self.vae.post_quant_conv(z)
        decoder = self.vae.decoder
        out = decoder.mid_block(decoder.conv_in(z), None)
        return out, out.reshape(-1)[:1]


class VaeDecodeStage(nn.Module):
    """One upsampling block of the decoder, plus the output head if it is last.

    Args:
        vae: The ``AutoencoderKL``.
        block_index: Index into ``vae.decoder.up_blocks``.
        is_last: Whether to append ``conv_norm_out`` / activation / ``conv_out``.
    """

    def __init__(self, vae, block_index: int, is_last: bool) -> None:
        super().__init__()
        self.block = vae.decoder.up_blocks[block_index]
        self.is_last = is_last
        self.decoder = vae.decoder if is_last else None

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.block(sample, None)
        if self.is_last:
            decoder = self.decoder
            out = decoder.conv_out(decoder.conv_act(decoder.conv_norm_out(out)))
        return out, out.reshape(-1)[:1]


def build_decode_stages(vae) -> list[nn.Module]:
    """Split the VAE decoder into per-resolution-level compilable stages."""
    num_up_blocks = len(vae.decoder.up_blocks)
    stages: list[nn.Module] = [VaeDecodeHead(vae)]
    stages += [
        VaeDecodeStage(vae, i, is_last=i == num_up_blocks - 1)
        for i in range(num_up_blocks)
    ]
    return stages
