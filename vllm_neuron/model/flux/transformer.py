# SPDX-License-Identifier: Apache-2.0
"""Compiled denoising step for the FLUX transformer on Neuron.

``NeuronFluxTransformer`` wraps a diffusers ``FluxTransformer2DModel`` in a
module whose forward is a single static graph, suitable for
``torch.compile(backend="neuron_libtorch", fullgraph=True)``. It reproduces
upstream's ``forward`` block-for-block and changes only what the Neuron
compiler cannot take:

* **RoPE tables are inputs, not computed inline.** Upstream's ``FluxPosEmbed``
  builds them in float64, which Neuron has no lowering for. They depend solely
  on image resolution and prompt length, both fixed by ``FluxNeuronConfig``, so
  ``build_rotary_embedding`` computes them once on the host in float64 (bit-exact
  with upstream) and the result is passed in as a plain tensor pair.
* **GELU is recomputed from tanh.** ``libtorch_neuronx_lite`` replaces
  ``F.gelu`` with a C extension Dynamo cannot trace, so the equivalent tanh
  expression is substituted at load time. Same workaround as the Qwen3-VL vision
  encoder.
* **The scheduler update is folded in** (optional). Denoising is a sequential
  chain of ~30 steps whose only cross-step state is the latent tensor. Returning
  a velocity would force a device->host->device round trip per step just to
  apply a two-term Euler update; folding it in keeps latents resident on device
  for the whole loop.

Everything else -- attention, norms, the double/single block stacks -- runs
upstream diffusers code, with attention processors swapped for
``NeuronFluxAttnProcessor``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from diffusers.models.activations import GELU as DiffusersGELU
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

from .attention import NeuronFluxAttnProcessor
from .config import FluxNeuronConfig


@torch.compiler.allow_in_graph
def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """tanh-approximate GELU, traceable by torch.compile on Neuron.

    Matches ``F.gelu(x, approximate="tanh")``, which FLUX uses in both the
    single-block MLP and the double-block feed-forwards.
    """
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


class _GeluTanh(nn.Module):
    """Traceable stand-in for ``nn.GELU(approximate="tanh")``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu_tanh(x)


def patch_untraceable_activations(module: nn.Module) -> int:
    """Replace GELU activations in ``module`` with traceable equivalents.

    Handles both shapes FLUX uses: bare ``nn.GELU`` submodules (the single
    block's ``act_mlp``) and diffusers' fused ``GELU`` projection layer, whose
    ``gelu`` method calls ``F.gelu`` directly.

    Args:
        module: Subtree to rewrite in place.

    Returns:
        Number of activations replaced.
    """
    replaced = 0
    # Snapshot the walk before mutating the tree.
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.GELU):
                setattr(parent, name, _GeluTanh())
                replaced += 1
            elif isinstance(child, DiffusersGELU):
                if child.approximate != "tanh":
                    raise NotImplementedError(
                        f"only tanh-approximate GELU is patched, got "
                        f"approximate={child.approximate!r}"
                    )
                # Keep the module (it owns `proj`), swap only the activation.
                child.gelu = gelu_tanh
                replaced += 1
    return replaced


def build_rotary_embedding(
    config: FluxNeuronConfig,
    transformer_config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the joint RoPE tables on the host.

    Reproduces ``FluxPipeline._prepare_latent_image_ids`` followed by
    ``FluxPosEmbed.forward``: text tokens get all-zero position ids and are
    concatenated ahead of the image tokens, whose ids carry (row, column) over
    the patchified latent grid. The three id axes are embedded with the per-axis
    dims from ``axes_dims_rope`` and concatenated on the feature dim.

    Computed in float64 to match upstream exactly, then cast to float32 for the
    device (Neuron has no float64).

    Args:
        config: Resolution and prompt length; determines the table shape.
        transformer_config: The diffusers transformer config, for
            ``axes_dims_rope``.

    Returns:
        ``(cos, sin)``, each ``[joint_seq_len, sum(axes_dims_rope)]`` float32.
    """
    latent_rows = config.latent_height // 2
    latent_cols = config.latent_width // 2

    image_ids = torch.zeros(latent_rows, latent_cols, 3, dtype=torch.float64)
    image_ids[..., 1] += torch.arange(latent_rows, dtype=torch.float64)[:, None]
    image_ids[..., 2] += torch.arange(latent_cols, dtype=torch.float64)[None, :]
    image_ids = image_ids.reshape(latent_rows * latent_cols, 3)

    text_ids = torch.zeros(config.max_sequence_length, 3, dtype=torch.float64)
    ids = torch.cat([text_ids, image_ids], dim=0)

    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(transformer_config.axes_dims_rope):
        cos, sin = get_1d_rotary_pos_embed(
            dim,
            ids[:, axis],
            theta=10000,
            repeat_interleave_real=True,
            use_real=True,
            freqs_dtype=torch.float64,
        )
        cos_parts.append(cos)
        sin_parts.append(sin)

    freqs_cos = torch.cat(cos_parts, dim=-1).to(torch.float32)
    freqs_sin = torch.cat(sin_parts, dim=-1).to(torch.float32)
    return freqs_cos, freqs_sin


class NeuronFluxTransformer(nn.Module):
    """One denoising step of FLUX as a single compilable graph.

    Args:
        transformer: A loaded diffusers ``FluxTransformer2DModel``.
        config: Neuron placement / shape config.

    Raises:
        NotImplementedError: If the checkpoint is a guidance-free FLUX variant
            (``guidance_embeds=False``, e.g. FLUX.1-schnell). Only the
            guidance-distilled variants are wired up.
    """

    def __init__(
        self, transformer: FluxTransformer2DModel, config: FluxNeuronConfig
    ) -> None:
        super().__init__()
        if not transformer.config.guidance_embeds:
            raise NotImplementedError(
                "only guidance-distilled FLUX checkpoints are supported "
                "(transformer config needs guidance_embeds=True); got a "
                "guidance-free variant."
            )
        self.transformer = transformer
        self.fuse_scheduler_step = config.fuse_scheduler_step

        if config.use_nki_attention:
            for block in (
                *transformer.transformer_blocks,
                *transformer.single_transformer_blocks,
            ):
                block.attn.set_processor(NeuronFluxAttnProcessor())

    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity at ``timestep`` and optionally advance the latents.

        Args:
            latents: Packed latents ``[batch, image_seq_len, in_channels]``.
            encoder_hidden_states: T5 prompt embeddings
                ``[batch, max_sequence_length, joint_attention_dim]``.
            pooled_projections: Pooled CLIP embedding
                ``[batch, pooled_projection_dim]``.
            timestep: Current timestep in [0, 1] (upstream's ``t / 1000``),
                shape ``[batch]``.
            guidance: Distilled guidance scale, shape ``[batch]``.
            freqs_cos: RoPE cosine table from ``build_rotary_embedding``.
            freqs_sin: RoPE sine table from ``build_rotary_embedding``.
            sigma: Noise level at the current step, shape ``[1]``.
            sigma_next: Noise level at the next step, shape ``[1]``.

        Returns:
            ``(out, sync_tag)``. ``out`` is the advanced latents when
            ``fuse_scheduler_step`` is set, otherwise the raw velocity
            prediction. ``sync_tag`` is a one-element view of ``out``: reading
            it on the host is the cheapest way to wait for this step, since
            NEFF execution is asynchronous and the runtime's queue is bounded.
        """
        model = self.transformer

        hidden_states = model.x_embedder(latents)
        # Upstream scales both back up by 1000 inside forward; keep the same
        # convention so callers can pass the pipeline's t/1000 unchanged.
        temb = model.time_text_embed(
            timestep.to(hidden_states.dtype) * 1000,
            guidance.to(hidden_states.dtype) * 1000,
            pooled_projections,
        )
        encoder_hidden_states = model.context_embedder(encoder_hidden_states)
        image_rotary_emb = (freqs_cos, freqs_sin)

        for block in model.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        for block in model.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = model.norm_out(hidden_states, temb)
        velocity = model.proj_out(hidden_states)

        if self.fuse_scheduler_step:
            # FlowMatchEulerDiscreteScheduler.step for this scheduler is exactly
            # x + (sigma_next - sigma) * v. Done in fp32 to keep the accumulated
            # latent from drifting over ~30 steps of BF16 addition.
            out = (latents.float() + (sigma_next - sigma) * velocity.float()).to(
                latents.dtype
            )
        else:
            out = velocity

        return out, out.reshape(-1)[:1]
