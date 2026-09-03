# SPDX-License-Identifier: Apache-2.0
"""What one tensor-parallel rank of the FLUX pipeline runs.

Every network lives here, on this rank's NeuronCore, because a process drives
exactly one core -- the compile backend loads every NEFF onto the process's own
core -- and a core belongs to one process. So `tp_degree` cores means
`tp_degree` processes, each holding a share of the model, and the pipeline process
itself never touches the device.

How each component is divided, following NxD Inference's arrangement for the same
model:

| Component | Across ranks | Why |
|---|---|---|
| `transformer` | sharded | 15.2 GiB and 28 invocations per request: the whole cost |
| `text_encoder_2` (T5-XXL) | sharded | 8.9 GiB, and it divides cleanly on 64 heads |
| `text_encoder` (CLIP-L) | replicated | 0.22 GiB; sharding it would cost collectives to save nothing |
| `vae` (decoder) | replicated | 0.15 GiB, convolutional, runs once per request |

Replicated components carry no collectives, so a rank runs them alone; sharded ones
do, so every rank has to execute them together. The command protocol keeps that
true by broadcasting every command to every rank.

## Protocol

`MPExecutor` sends one broadcast per call and gets one tensor back per rank, so the
first argument selects what to do and the request's state stays here between calls:

| `mode` | Sends | Does |
|---|---|---|
| `"encode"` | CLIP and T5 token ids | both encoders; keeps the embeddings |
| `"context"` | guidance, packed latents | keeps them, with the RoPE tables built here |
| `"step"` | timestep, sigma, sigma_next | one denoising step; latents updated in place |
| `"latents"` | nothing | returns the latents, for accuracy checks |
| `"decode"` | nothing | unpacks and VAE-decodes; returns the image |

Nothing large crosses the boundary per step: the embeddings never leave, and a step
sends three scalars and gets back a one-element fence. Reading that fence is what
waits for the device, since NEFF execution is queued rather than synchronous.

## Host memory at load

Each rank materializes the whole checkpoint before keeping its shares of it, so
they take a file lock and do it one at a time. That is the dominant startup cost.
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from contextlib import contextmanager

import torch
import torch.nn as nn

from .config import FluxNeuronConfig
from .parallel import shard_flux_transformer, shard_t5_encoder
from .transformer import (
    NeuronFluxTransformer,
    build_rotary_embedding,
    patch_untraceable_activations,
)
from .vae import build_decode_stages, patch_upsampling

logger = logging.getLogger(__name__)

ENCODE, CONTEXT, STEP, LATENTS, DECODE = "encode", "context", "step", "latents", "decode"

# CLIP's fixed context length; the pooled branch always sees exactly this many
# tokens, independent of max_sequence_length (which governs T5 only).
CLIP_SEQ_LEN = 77

DEVICE = torch.device("neuron", 0)


@contextmanager
def _load_serialized():
    """Hold a lock so only one rank materializes the checkpoint at a time."""
    path = os.path.join(tempfile.gettempdir(), "flux_tp_load.lock")
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _TextEncoder(nn.Module):
    """Static-shape wrapper so a HF text encoder compiles as one graph.

    FLUX pads both prompts to a fixed length and passes no attention mask, so the
    only input is a token-id tensor of constant shape.
    """

    def __init__(self, encoder, pooled: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooled = pooled

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids, output_hidden_states=False)
        embeds = out.pooler_output if self.pooled else out[0]
        return embeds, embeds.reshape(-1)[:1]


class FluxRank(nn.Module):
    """One rank's models and the state of the request in flight.

    Not itself compiled: it dispatches on ``mode`` and calls the compiled graphs,
    so each graph stays branch-free and static.

    Args:
        pipe: A loaded diffusers ``FluxPipeline`` whose components this rank takes
            over. Sharded in place.
        config: Shape config, shared by every rank.
    """

    def __init__(self, pipe, config: FluxNeuronConfig) -> None:
        super().__init__()
        self.config = config
        self.pipe = pipe

        for component in (
            pipe.transformer,
            pipe.vae,
            pipe.text_encoder,
            pipe.text_encoder_2,
        ):
            component.requires_grad_(False).eval()

        # Ops that do not lower, rewritten before anything captures references into
        # the module tree.
        patch_untraceable_activations(pipe.transformer)
        patch_upsampling(pipe.vae)

        shard_flux_transformer(pipe.transformer)
        shard_t5_encoder(pipe.text_encoder_2.encoder)

        self._step = NeuronFluxTransformer(pipe.transformer, config)
        self._clip = _TextEncoder(pipe.text_encoder, pooled=True)
        self._t5 = _TextEncoder(pipe.text_encoder_2, pooled=False)
        self._decode_stages = build_decode_stages(pipe.vae)

        freqs_cos, freqs_sin = build_rotary_embedding(config, pipe.transformer.config)

        for module in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
            module.to(DEVICE)
        # Graph inputs, so they live where the step graph runs. Uploaded once,
        # reused by every step of every request.
        self.freqs_cos = freqs_cos.to(DEVICE)
        self.freqs_sin = freqs_sin.to(DEVICE)

        self.step_compiled = self._compile(self._step)
        self.clip_compiled = self._compile(self._clip)
        self.t5_compiled = self._compile(self._t5)
        self.decode_compiled = [self._compile(stage) for stage in self._decode_stages]

        self.embeds: torch.Tensor | None = None
        self.pooled: torch.Tensor | None = None
        self.guidance: torch.Tensor | None = None
        self.latents: torch.Tensor | None = None

    def _compile(self, module: nn.Module) -> nn.Module:
        from vllm_neuron.envs import get_compile_backend_name

        # Lift Dynamo's recompile cap: the VAE decode stages share one code object
        # across five resolution levels, which blows through the default of 8.
        torch._dynamo.config.cache_size_limit = 2**62
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 2**62

        return torch.compile(
            module,
            backend=get_compile_backend_name(),
            fullgraph=True,
            options={
                "alias_meta_to_neuron": True,
                "compiler_args": self.config.neuronx_cc_args(),
            },
        )

    def warmup(self) -> None:
        """Compile every graph, with synthetic inputs at the runtime shapes.

        Driven by a call rather than at trace time so a failure surfaces here,
        at load, rather than inside the first request.
        """
        cfg = self.config
        # On device, because the executor is what moves a request's inputs and
        # this call bypasses it.
        def on_device(*tensors):
            return [t.to(DEVICE) for t in tensors]

        self(
            ENCODE,
            *on_device(
                torch.zeros(1, CLIP_SEQ_LEN, dtype=torch.long),
                torch.zeros(1, cfg.max_sequence_length, dtype=torch.long),
            ),
        )
        self(
            CONTEXT,
            *on_device(
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(
                    1, cfg.image_seq_len, self.pipe.transformer.config.in_channels,
                    dtype=cfg.dtype,
                ),
            ),
        )
        self(
            STEP,
            *on_device(
                torch.ones(1, dtype=torch.float32),
                torch.ones(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
            ),
        )
        self(DECODE)

    def forward(self, mode: str, *args: torch.Tensor) -> torch.Tensor:
        if mode == ENCODE:
            clip_ids, t5_ids = args
            pooled, pooled_tag = self.clip_compiled(clip_ids)
            embeds, embeds_tag = self.t5_compiled(t5_ids)
            self.pooled = pooled.to(self.config.dtype)
            self.embeds = embeds.to(self.config.dtype)
            pooled_tag.cpu()
            return embeds_tag

        if mode == CONTEXT:
            guidance, latents = args
            self.guidance = guidance
            self.latents = latents
            return latents.reshape(-1)[:1]

        if mode == STEP:
            timestep, sigma, sigma_next = args
            out, tag = self.step_compiled(
                self.latents,
                self.embeds,
                self.pooled,
                timestep,
                self.guidance,
                self.freqs_cos,
                self.freqs_sin,
                sigma,
                sigma_next,
            )
            self.latents = out
            return tag

        if mode == LATENTS:
            return self.latents

        if mode == DECODE:
            cfg = self.config
            # Unpacking is a permute+reshape needing a contiguity-fixing copy that
            # privateuse1 does not implement. It runs on 512 KiB, so the hop
            # through the host is free next to the decode itself.
            unpacked = self.pipe._unpack_latents(
                self.latents.cpu(), cfg.height, cfg.width, self.pipe.vae_scale_factor
            )
            stage_input = unpacked.to(DEVICE)
            for stage in self.decode_compiled:
                stage_input, tag = stage(stage_input)
                tag.cpu()
            return stage_input

        raise ValueError(f"unknown mode {mode!r}")


def load_rank(model_path: str, config: FluxNeuronConfig) -> FluxRank:
    """Build this rank. Runs inside a worker process; used as ``model_load``.

    Args:
        model_path: HF repo id or local path of the ``FluxPipeline`` folder.
        config: Shape config, shared by every rank.

    Returns:
        The rank, on device with every graph compiled.
    """
    from diffusers import FluxPipeline

    with _load_serialized():
        pipe = FluxPipeline.from_pretrained(model_path, dtype=config.dtype)
        rank = FluxRank(pipe, config)

    # Warmup executes the sharded graphs, so it needs every rank present: the
    # first device all-reduce would otherwise sit waiting for ranks still queued
    # behind the load lock, long enough for the runtime to give up on it.
    #
    # An all-reduce on a host tensor rather than dist.barrier(), which picks a
    # device for itself and lands on privateuse1: "You should register
    # PrivateUse1HooksInterface ... and implement isAvailable".
    torch.distributed.all_reduce(torch.zeros(1))
    rank.warmup()
    return rank
