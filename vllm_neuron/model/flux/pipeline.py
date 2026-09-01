# SPDX-License-Identifier: Apache-2.0
"""FLUX.1 text-to-image pipeline on Neuron.

Developed against ``Freepik/flux.1-lite-8B`` (8 double + 38 single blocks,
distilled from FLUX.1-dev), but nothing here is specific to that checkpoint: any
guidance-distilled ``FluxPipeline`` in diffusers format loads through the same
path.

Why this lives outside the vLLM engine
--------------------------------------
vLLM 0.24 has no text-to-image request path -- its ``DiffusionConfig`` is for
discrete diffusion *language* models (dLLM), which reuse the token/KV-cache data
plane. FLUX has no KV cache, no tokens on the output side, and no autoregressive
loop, so it does not fit ``NeuronModelRunner``. What it does share with the rest
of this package is the compilation stack, so this module reuses that directly:
``torch.compile`` with the ``neuron_libtorch`` backend, the same ``neuronx-cc``
flags, and the package's NKI flash-attention kernel.

Execution model
---------------
Every component is a separate NEFF. Shapes are fully static (fixed resolution,
prompt padded to ``max_sequence_length``), so one compilation serves every
request and there is no bucketing. Per request:

1. tokenize on host; CLIP + T5 encoders produce the pooled and sequence
   embeddings (device or CPU per config)
2. sample initial latents on the host from a seeded generator, pack, move to
   device once
3. run ``num_inference_steps`` iterations of the compiled step graph; latents
   stay on device the whole time
4. unpack, VAE-decode, postprocess to PIL
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.utils.torch_utils import randn_tensor

from .config import FluxNeuronConfig
from .transformer import (
    NeuronFluxTransformer,
    build_rotary_embedding,
    patch_untraceable_activations,
)
from .vae import build_decode_stages, patch_upsampling

logger = logging.getLogger(__name__)

# CLIP's fixed context length; the pooled branch always sees exactly this many
# tokens, independent of max_sequence_length (which governs T5 only).
CLIP_SEQ_LEN = 77


@dataclass
class GenerationTiming:
    """Wall-clock breakdown of one ``generate`` call, in milliseconds.

    Every measurement is taken after the device has been synchronized (NEFF
    execution is asynchronous), so the parts sum to ``total_ms`` up to host
    overhead.
    """

    encode_ms: float = 0.0
    latent_init_ms: float = 0.0
    denoise_ms: float = 0.0
    decode_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    step_ms: list[float] = field(default_factory=list)

    @property
    def mean_step_ms(self) -> float:
        return float(np.mean(self.step_ms)) if self.step_ms else 0.0

    @property
    def median_step_ms(self) -> float:
        return float(np.median(self.step_ms)) if self.step_ms else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "encode_ms": round(self.encode_ms, 2),
            "latent_init_ms": round(self.latent_init_ms, 2),
            "denoise_ms": round(self.denoise_ms, 2),
            "decode_ms": round(self.decode_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "mean_step_ms": round(self.mean_step_ms, 2),
            "median_step_ms": round(self.median_step_ms, 2),
            "step_ms": [round(s, 2) for s in self.step_ms],
        }


class _TextEncoder(nn.Module):
    """Static-shape wrapper so a HF text encoder compiles as one graph.

    FLUX pads both prompts to a fixed length and passes no attention mask, so
    the only input is a token-id tensor of constant shape.
    """

    def __init__(self, encoder, pooled: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooled = pooled

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids, output_hidden_states=False)
        embeds = out.pooler_output if self.pooled else out[0]
        return embeds, embeds.reshape(-1)[:1]


class NeuronFluxPipeline:
    """FLUX.1 text-to-image on a single logical NeuronCore.

    Use :meth:`from_pretrained` to build one; the constructor takes an
    already-loaded diffusers pipeline.

    Attributes:
        placement: Where each component ended up (``"neuron"`` or ``"cpu"``).
            Components requested on device that failed to compile are recorded
            here as ``"cpu"``.
        compile_ms: Wall-clock compilation time per component.
    """

    def __init__(self, pipe: FluxPipeline, config: FluxNeuronConfig) -> None:
        self.config = config
        self.pipe = pipe
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.image_processor = pipe.image_processor
        self.vae_scale_factor = pipe.vae_scale_factor

        self.placement: dict[str, str] = {}
        self.compile_ms: dict[str, float] = {}

        # Inference only: without this, components left on CPU run in eager mode
        # and build an autograd graph, which then blocks the numpy conversion in
        # postprocessing.
        for component in (
            pipe.transformer,
            pipe.vae,
            pipe.text_encoder,
            pipe.text_encoder_2,
        ):
            component.requires_grad_(False).eval()

        # Rewrite the ops that do not lower before anything captures references
        # into the module tree.
        logger.info(
            "Rewrote %d GELU activations in the transformer and %d upsamplers "
            "in the VAE for Neuron lowering",
            patch_untraceable_activations(pipe.transformer),
            patch_upsampling(pipe.vae),
        )

        self._step = NeuronFluxTransformer(pipe.transformer, config)
        self._clip = _TextEncoder(pipe.text_encoder, pooled=True)
        self._t5 = _TextEncoder(pipe.text_encoder_2, pooled=False)
        self._decode_stages = build_decode_stages(pipe.vae)

        freqs_cos, freqs_sin = build_rotary_embedding(config, pipe.transformer.config)
        self._freqs_cos_host = freqs_cos
        self._freqs_sin_host = freqs_sin
        self._freqs_cos = freqs_cos
        self._freqs_sin = freqs_sin

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: FluxNeuronConfig | None = None,
        **kwargs: Any,
    ) -> NeuronFluxPipeline:
        """Load a diffusers FLUX checkpoint and prepare it for Neuron.

        Args:
            model_path: HF repo id or local path to a ``FluxPipeline`` folder.
            config: Neuron shape/placement config; defaults to 1024x1024 with
                the default component placement (see ``FluxNeuronConfig``).
            **kwargs: Forwarded to ``FluxPipeline.from_pretrained``.

        Returns:
            A pipeline that still needs :meth:`compile` before the first
            request (``generate`` calls it lazily otherwise).
        """
        config = config or FluxNeuronConfig()
        logger.info("Loading FLUX checkpoint from %s", model_path)
        pipe = FluxPipeline.from_pretrained(model_path, dtype=config.dtype, **kwargs)
        return cls(pipe, config)

    def _compile(self, module: nn.Module) -> nn.Module:
        from vllm_neuron.envs import get_compile_backend_name

        # Lift Dynamo's recompile cap, as NeuronModelRunner does. Shapes here
        # are static, so a recompile is never silent overhead -- but the VAE
        # decode stages share one code object across five resolution levels, and
        # a process that builds pipelines for more than one output size blows
        # through the default limit of 8 and gets a component pushed to CPU.
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

    @contextmanager
    def _timed(self, name: str):
        start = time.perf_counter()
        yield
        self.compile_ms[name] = (time.perf_counter() - start) * 1e3

    def _place(
        self,
        name: str,
        wrappers: list[nn.Module],
        inner: nn.Module,
        warmup_args: tuple[torch.Tensor, ...],
    ) -> list[nn.Module]:
        """Move a component to Neuron and compile its graphs, or leave it on CPU.

        Compilation is driven by a warmup call with synthetic inputs of the exact
        runtime shapes -- same pattern as the LLM path, and required here because
        a failure has to surface at load time rather than mid-request.

        A component may be more than one graph (the VAE decoder is five). Stages
        are warmed up as a chain: each one's output feeds the next, so only the
        first stage's shapes have to be stated.

        Args:
            name: Component key in ``FLUX_COMPONENTS``.
            wrappers: Modules to compile, in execution order.
            inner: The weight-owning module to move between devices.
            warmup_args: Synthetic inputs to the first stage, on the host.

        Returns:
            The compiled modules, or ``wrappers`` unchanged if the component
            stays on CPU.
        """
        if not self.config.runs_on_device(name):
            self.placement[name] = "cpu"
            logger.info("Component %s: staying on CPU (per config)", name)
            return wrappers

        device = self.config.device_for(name)
        logger.info(
            "Component %s: moving to %s and compiling %d graph(s)",
            name,
            device,
            len(wrappers),
        )
        try:
            # The weight upload is inside the try because it is the step that
            # runs out of HBM: a logical core holds 24 GB, and the transformer
            # plus T5 do not both fit. Falling back keeps the pipeline usable.
            inner.to(device)
            compiled = [self._compile(wrapper) for wrapper in wrappers]
            with self._timed(name):
                args = tuple(
                    a.to(device) if isinstance(a, torch.Tensor) else a
                    for a in warmup_args
                )
                for stage in compiled:
                    out, tag = stage(*args)
                    tag.cpu()
                    args = (out,)
        except Exception:
            logger.warning(
                "Component %s failed to load or compile for Neuron; falling "
                "back to CPU. Latency for this component will be much worse.",
                name,
                exc_info=True,
            )
            inner.to("cpu")
            self.placement[name] = "cpu"
            self.compile_ms.pop(name, None)
            return wrappers

        self.placement[name] = "neuron"
        logger.info(
            "Component %s: compiled in %.1f s", name, self.compile_ms[name] / 1e3
        )
        return compiled

    def compile(self) -> None:
        """Place and compile every component. Idempotent."""
        if self.placement:
            return

        cfg = self.config
        batch = 1
        tcfg = self.pipe.transformer.config
        dtype = cfg.dtype

        # Transformer first: it is both the largest component and the one whose
        # placement actually decides end-to-end latency (~30 invocations per
        # request against one for everything else), so if HBM runs short it must
        # not be the component that loses its seat.
        (self._step_compiled,) = self._place(
            "transformer",
            [self._step],
            self.pipe.transformer,
            (
                torch.zeros(batch, cfg.image_seq_len, tcfg.in_channels, dtype=dtype),
                torch.zeros(
                    batch,
                    cfg.max_sequence_length,
                    tcfg.joint_attention_dim,
                    dtype=dtype,
                ),
                torch.zeros(batch, tcfg.pooled_projection_dim, dtype=dtype),
                torch.zeros(batch, dtype=torch.float32),
                torch.zeros(batch, dtype=torch.float32),
                self._freqs_cos_host,
                self._freqs_sin_host,
                torch.ones(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
            ),
        )
        self._decode_compiled = self._place(
            "vae",
            self._decode_stages,
            self.pipe.vae,
            (
                torch.zeros(
                    batch,
                    self.pipe.vae.config.latent_channels,
                    cfg.latent_height,
                    cfg.latent_width,
                    dtype=dtype,
                ),
            ),
        )
        (self._clip_compiled,) = self._place(
            "text_encoder",
            [self._clip],
            self.pipe.text_encoder,
            (torch.zeros(batch, CLIP_SEQ_LEN, dtype=torch.long),),
        )
        (self._t5_compiled,) = self._place(
            "text_encoder_2",
            [self._t5],
            self.pipe.text_encoder_2,
            (torch.zeros(batch, cfg.max_sequence_length, dtype=torch.long),),
        )

        # The RoPE tables are graph inputs, so they live wherever the step graph
        # runs. Uploaded once here, reused by every step of every request.
        if self.placement["transformer"] == "neuron":
            device = cfg.device_for("transformer")
            self._freqs_cos = self._freqs_cos_host.to(device)
            self._freqs_sin = self._freqs_sin_host.to(device)

        logger.info(
            "FLUX pipeline ready. placement=%s compile_time_s=%s",
            self.placement,
            {k: round(v / 1e3, 1) for k, v in self.compile_ms.items()},
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _device_for(self, component: str) -> torch.device:
        return (
            self.config.device_for(component)
            if self.placement.get(component) == "neuron"
            else torch.device("cpu")
        )

    @staticmethod
    def _handoff(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move a tensor between components, hopping via the host if needed.

        There is no direct copy between two logical NeuronCores, so a
        cross-core handoff has to land on the CPU in between. Only small
        tensors cross component boundaries (prompt embeddings, packed latents),
        so the extra hop does not show up in the timings.
        """
        if tensor.device == device:
            return tensor
        if tensor.device.type == "neuron" and device.type == "neuron":
            return tensor.cpu().to(device)
        return tensor.to(device)

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Run both text encoders.

        Args:
            prompt: The text prompt. Truncated to CLIP's 77 tokens for the
                pooled branch and to ``max_sequence_length`` for T5, matching
                upstream.

        Returns:
            ``(prompt_embeds, pooled_prompt_embeds)`` --
            ``[1, max_sequence_length, joint_attention_dim]`` and
            ``[1, pooled_projection_dim]``, on the transformer's device.
        """
        clip_ids = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=CLIP_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        t5_ids = self.tokenizer_2(
            [prompt],
            padding="max_length",
            max_length=self.config.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        pooled, pooled_tag = self._clip_compiled(
            clip_ids.to(self._device_for("text_encoder"))
        )
        embeds, embeds_tag = self._t5_compiled(
            t5_ids.to(self._device_for("text_encoder_2"))
        )
        # Fence both encoders, so the caller's timing attributes their cost here
        # rather than to whichever later stage first waits on the queue.
        pooled_tag.cpu()
        embeds_tag.cpu()

        target = self._device_for("transformer")
        return (
            self._handoff(embeds.to(self.config.dtype), target),
            self._handoff(pooled.to(self.config.dtype), target),
        )

    def _timesteps_and_sigmas(
        self, num_inference_steps: int
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Host-side schedule for the flow-matching sampler.

        Returns:
            ``(timesteps, sigmas)`` where ``sigmas`` has
            ``num_inference_steps + 1`` entries (trailing 0.0), so step ``i``
            consumes ``sigmas[i]`` and ``sigmas[i + 1]``.
        """
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            self.config.image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
        self.scheduler.set_begin_index(0)
        return self.scheduler.timesteps, self.scheduler.sigmas.numpy()

    def _init_latents(self, seed: int | None) -> torch.Tensor:
        """Sample and pack the initial latents.

        Sampled on the host so a seed reproduces the same image regardless of
        component placement, then packed into the transformer's
        ``[1, image_seq_len, in_channels]`` layout.
        """
        cfg = self.config
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        channels = self.pipe.transformer.config.in_channels // 4
        latents = randn_tensor(
            (1, channels, cfg.latent_height, cfg.latent_width),
            generator=generator,
            device=torch.device("cpu"),
            dtype=cfg.dtype,
        )
        return self.pipe._pack_latents(
            latents, 1, channels, cfg.latent_height, cfg.latent_width
        )

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int | None = None,
        output_type: str = "pil",
    ) -> tuple[Any, GenerationTiming]:
        """Generate one image.

        Args:
            prompt: Text prompt.
            num_inference_steps: Denoising steps. FLUX.1-lite is tuned for the
                same 20-30 range as FLUX.1-dev.
            guidance_scale: Distilled guidance embedding value. This is *not*
                classifier-free guidance -- there is no negative pass, so cost
                is independent of this value.
            seed: Host RNG seed for the initial latents. ``None`` is
                nondeterministic.
            output_type: ``"pil"``, ``"np"``, or ``"latent"``.

        Returns:
            ``(image, timing)``.
        """
        self.compile()
        with torch.no_grad():
            return self._generate(
                prompt, num_inference_steps, guidance_scale, seed, output_type
            )

    def _generate(
        self,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        output_type: str,
    ) -> tuple[Any, GenerationTiming]:
        cfg = self.config
        timing = GenerationTiming()
        total_start = time.perf_counter()

        start = time.perf_counter()
        prompt_embeds, pooled_embeds = self.encode_prompt(prompt)
        timing.encode_ms = (time.perf_counter() - start) * 1e3

        start = time.perf_counter()
        timesteps, sigmas = self._timesteps_and_sigmas(num_inference_steps)
        device = self._device_for("transformer")
        latents = self._init_latents(seed).to(device)
        guidance = torch.full((1,), guidance_scale, dtype=torch.float32).to(device)
        timing.latent_init_ms = (time.perf_counter() - start) * 1e3

        for i, t in enumerate(timesteps):
            step_start = time.perf_counter()
            # Cast before the device move, never during it (see the scheduler
            # branch below).
            timestep = (t / 1000).reshape(1).to(torch.float32).to(device)
            sigma = torch.full((1,), float(sigmas[i]), dtype=torch.float32).to(device)
            sigma_next = torch.full((1,), float(sigmas[i + 1]), dtype=torch.float32).to(
                device
            )

            out, tag = self._step_compiled(
                latents,
                prompt_embeds,
                pooled_embeds,
                timestep,
                guidance,
                self._freqs_cos,
                self._freqs_sin,
                sigma,
                sigma_next,
            )
            if cfg.fuse_scheduler_step:
                latents = out
            else:
                # Cast on the host and move as a second step: privateuse1 cannot
                # convert dtype as part of a host-to-device copy.
                stepped = self.scheduler.step(
                    out.cpu().float(), t, latents.cpu().float(), return_dict=False
                )[0]
                latents = stepped.to(cfg.dtype).to(device)

            # NEFF execution is queued, not synchronous: without this the loop
            # would race ahead of the device and overrun the runtime's execution
            # queue. Reading a one-element view is the cheapest available fence.
            tag.cpu()
            timing.step_ms.append((time.perf_counter() - step_start) * 1e3)

        timing.denoise_ms = sum(timing.step_ms)

        if output_type == "latent":
            timing.total_ms = (time.perf_counter() - total_start) * 1e3
            return latents, timing

        start = time.perf_counter()
        # Unpacking is a permute+reshape, which needs a contiguity-fixing copy
        # that privateuse1 does not implement. It runs on 512 KiB of latents, so
        # the round trip through the host is free next to the decode itself.
        unpacked = self.pipe._unpack_latents(
            latents.cpu(), cfg.height, cfg.width, self.vae_scale_factor
        )
        stage_input = unpacked.to(self._device_for("vae"))
        for stage in self._decode_compiled:
            stage_input, tag = stage(stage_input)
            tag.cpu()
        image = stage_input.cpu()
        timing.decode_ms = (time.perf_counter() - start) * 1e3

        start = time.perf_counter()
        image = self.image_processor.postprocess(
            image.to(torch.float32), output_type=output_type
        )[0]
        timing.postprocess_ms = (time.perf_counter() - start) * 1e3

        timing.total_ms = (time.perf_counter() - total_start) * 1e3
        return image, timing
