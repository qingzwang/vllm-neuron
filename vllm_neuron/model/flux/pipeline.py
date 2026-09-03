# SPDX-License-Identifier: Apache-2.0
"""FLUX.1 text-to-image pipeline on Neuron.

Developed against ``black-forest-labs/FLUX.1-dev`` (19 double + 38 single blocks),
but nothing here is specific to that checkpoint: any guidance-distilled
``FluxPipeline`` in diffusers format loads through the same path, with block counts,
head counts and dimensions read from the checkpoint.

Why this lives outside the vLLM engine
--------------------------------------
vLLM 0.24 has no text-to-image request path -- its ``DiffusionConfig`` is for
discrete diffusion *language* models (dLLM), which reuse the token/KV-cache data
plane. FLUX has no KV cache, no tokens on the output side, and no autoregressive
loop, so it does not fit ``NeuronModelRunner``. What it does share with the rest
of this package is the compilation stack, so this module reuses that directly:
``torch.compile`` with the ``neuron_libtorch`` backend, the same ``neuronx-cc``
flags, the package's NKI flash-attention kernel, and its tensor-parallel layers.

Execution model
---------------
The model is tensor-parallel across ``tp_degree`` NeuronCores, one process per
core, because the compile backend binds NEFF execution to the process's own core
and a core belongs to one process. Those processes hold every network; **this
process never touches the device**. It tokenizes, samples the initial latents,
drives the denoising loop, and turns the returned tensor into an image. See
``worker.py`` for what a rank runs and how the four components are divided.

Shapes are fully static (fixed resolution, prompt padded to
``max_sequence_length``), so one compilation serves every request and there is no
bucketing. Per request:

1. tokenize on host, then both text encoders run on the ranks and the embeddings
   stay there
2. sample initial latents on the host from a seeded generator, pack, send once
3. ``num_inference_steps`` iterations of the compiled step graph; the latents stay
   in the ranks the whole time
4. unpack and VAE-decode on the ranks, postprocess to PIL here
"""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.utils.torch_utils import randn_tensor

from .config import FluxNeuronConfig
from .worker import (
    CLIP_SEQ_LEN,
    CONTEXT,
    DECODE,
    ENCODE,
    LATENTS,
    LOAD_LORA,
    SELECT_LORA,
    STEP,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationTiming:
    """Wall-clock breakdown of one ``generate`` call, in milliseconds.

    Every measurement is taken after the ranks have answered, and answering means
    reading a device tensor, so the parts sum to ``total_ms`` up to host overhead.
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


class NeuronFluxPipeline:
    """FLUX.1 text-to-image on Neuron, tensor-parallel across NeuronCores.

    Occupies ``config.tp_degree`` logical cores, held by child processes for the
    pipeline's lifetime -- so :meth:`close` (or ``with``) matters. This process
    holds none.

    Use :meth:`from_pretrained` to build one; the constructor takes an
    already-loaded diffusers pipeline, from which it keeps only the tokenizers,
    the scheduler and the image processor. The networks are dropped here and
    loaded again inside each rank, which is also what lets a rank keep just its
    shard.

    Args:
        pipe: A loaded diffusers ``FluxPipeline``.
        config: Neuron shape/parallelism config.
        model_path: Where ``pipe`` was loaded from. The ranks load from it too, so
            it is required.

    Attributes:
        compile_ms: Wall-clock time to bring the ranks up, weight loading and
            compilation included.

    Raises:
        ValueError: If ``model_path`` is None.
    """

    def __init__(
        self,
        pipe: FluxPipeline,
        config: FluxNeuronConfig,
        model_path: str | None = None,
    ) -> None:
        if model_path is None:
            raise ValueError(
                "model_path is required: each rank loads its own shard of the "
                "checkpoint, so it has to know where the checkpoint is."
            )
        self.config = config
        self._model_path = model_path
        self.compile_ms: dict[str, float] = {}
        self._tp = None
        # name -> device slot. Slot 0 is the unmodified model and is never handed
        # out, so an adapter's slot is always >= 1.
        self._lora_slots: dict[str, int] = {}
        self._active_lora: str | None = None

        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.image_processor = pipe.image_processor
        self.vae_scale_factor = pipe.vae_scale_factor
        self._in_channels = pipe.transformer.config.in_channels
        self._pack = pipe._pack_latents

        # The ranks own every network. Keeping a second copy here would cost
        # ~25 GiB of host memory that nothing reads -- and the ranks are forked
        # from this process, so it would be copy-on-write pressure too.
        pipe.transformer = None
        pipe.text_encoder = None
        pipe.text_encoder_2 = None
        pipe.vae = None
        gc.collect()

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
            config: Neuron shape/parallelism config; defaults to 1024x1024 at
                ``tp_degree=2`` (see ``FluxNeuronConfig``).
            **kwargs: Forwarded to ``FluxPipeline.from_pretrained``.

        Returns:
            A pipeline that still needs :meth:`compile` before the first request
            (``generate`` calls it lazily otherwise).
        """
        config = config or FluxNeuronConfig()
        logger.info("Loading FLUX checkpoint from %s", model_path)
        pipe = FluxPipeline.from_pretrained(model_path, dtype=config.dtype, **kwargs)
        return cls(pipe, config, model_path=model_path)

    def compile(self) -> None:
        """Start the ranks, which load, shard and compile. Idempotent."""
        if self._tp is not None:
            return
        from .tp import TensorParallelFlux

        start = time.perf_counter()
        self._tp = TensorParallelFlux(self._model_path, self.config)
        self.compile_ms["ranks"] = (time.perf_counter() - start) * 1e3
        logger.info(
            "FLUX pipeline ready: tp_degree=%d on cores %s, up in %.1f s",
            self.config.tp_degree,
            list(self.config.tp_core_ids),
            self.compile_ms["ranks"] / 1e3,
        )

    def close(self) -> None:
        """Stop the ranks and release their cores. Idempotent."""
        tp, self._tp = self._tp, None
        if tp is not None:
            tp.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best effort: a pipeline that goes out of scope must not leave children
        # holding NeuronCores. Nothing useful to do if this fails during
        # interpreter teardown.
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def load_lora(self, name: str, path: str, slot: int | None = None) -> int:
        """Load an adapter into a device slot. Nothing is recompiled.

        Every rank reads the adapter and keeps its own shard of it, so this costs a
        pass over the file plus a host-to-device copy -- seconds for a large
        adapter. Switching to an already-loaded one afterwards is a 4-byte copy, so
        load the adapters you expect to alternate between and switch freely.

        Args:
            name: How to refer to it later.
            path: Directory or file holding the adapter. diffusers/PEFT, kohya and
                XLabs layouts all work.
            slot: Which slot to use, 1..lora_slots. Defaults to the next free one,
                or to the slot ``name`` already occupies.

        Returns:
            The slot the adapter now occupies.

        Raises:
            RuntimeError: If this pipeline was built without LoRA slots
                (``lora_slots=0``).
            ValueError: If every slot is taken and no slot was named, or the slot
                given is out of range.
        """
        cfg = self.config
        if not cfg.lora_enabled:
            raise RuntimeError(
                "This pipeline was built without LoRA slots; rebuild with "
                "FluxNeuronConfig(lora_slots=N) to load adapters."
            )
        self.compile()

        if slot is None:
            slot = self._lora_slots.get(name)
        if slot is None:
            taken = set(self._lora_slots.values())
            free = [i for i in range(1, cfg.lora_total_slots) if i not in taken]
            if not free:
                raise ValueError(
                    f"all {cfg.lora_slots} LoRA slots are taken by "
                    f"{sorted(self._lora_slots)}; pass slot= to overwrite one, or "
                    "rebuild with more lora_slots."
                )
            slot = free[0]
        if not 1 <= slot < cfg.lora_total_slots:
            raise ValueError(
                f"slot {slot} is out of range 1..{cfg.lora_slots}; slot 0 is the "
                "unmodified model."
            )

        start = time.perf_counter()
        self._tp.run(LOAD_LORA, slot, path)
        # Whatever used to be in this slot is gone.
        self._lora_slots = {
            n: s for n, s in self._lora_slots.items() if s != slot or n == name
        }
        self._lora_slots[name] = slot
        if self._active_lora is not None and self._lora_slots.get(self._active_lora) == slot:
            self._active_lora = name
        logger.info(
            "Adapter %r loaded into slot %d in %.2f s",
            name,
            slot,
            time.perf_counter() - start,
        )
        return slot

    def set_lora(self, name: str | None) -> None:
        """Choose the adapter later requests use. Sticky until changed.

        Args:
            name: A name passed to :meth:`load_lora`, or None for the unmodified
                model.

        Raises:
            KeyError: If ``name`` was never loaded.
            RuntimeError: If this pipeline was built without LoRA slots and ``name``
                is not None.
        """
        if name is None:
            if self.config.lora_enabled and self._tp is not None:
                self._tp.run(SELECT_LORA, 0)
            self._active_lora = None
            return
        if not self.config.lora_enabled:
            raise RuntimeError(
                "This pipeline was built without LoRA slots; rebuild with "
                "FluxNeuronConfig(lora_slots=N)."
            )
        self.compile()
        if name not in self._lora_slots:
            raise KeyError(
                f"adapter {name!r} is not loaded; loaded adapters are "
                f"{sorted(self._lora_slots)}"
            )
        self._tp.run(SELECT_LORA, self._lora_slots[name])
        self._active_lora = name

    def list_loras(self) -> dict[str, int]:
        """Loaded adapters and the slots they occupy."""
        return dict(self._lora_slots)

    @property
    def active_lora(self) -> str | None:
        """The adapter later requests will use, or None for the base model."""
        return self._active_lora

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _timesteps_and_sigmas(
        self, num_inference_steps: int
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Upstream's flow-matching schedule, on the host.

        Reproduces ``FluxPipeline.__call__``'s schedule setup: sigmas linear in
        [1, 1/n], shifted by the resolution-dependent mu that ``calculate_shift``
        derives from the image sequence length.
        """
        cfg = self.config
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            cfg.image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
        return self.scheduler.timesteps, self.scheduler.sigmas.numpy()

    def _init_latents(self, seed: int | None) -> torch.Tensor:
        """Sample and pack the initial latents.

        Sampled on the host so a seed reproduces the same image at any
        ``tp_degree``, then packed into the transformer's
        ``[1, image_seq_len, in_channels]`` layout.
        """
        cfg = self.config
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        channels = self._in_channels // 4
        latents = randn_tensor(
            (1, channels, cfg.latent_height, cfg.latent_width),
            generator=generator,
            device=torch.device("cpu"),
            dtype=cfg.dtype,
        )
        return self._pack(latents, 1, channels, cfg.latent_height, cfg.latent_width)

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Token ids for both encoders, truncated the way upstream truncates."""
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
        return clip_ids, t5_ids

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
            prompt: Text prompt. Truncated to CLIP's 77 tokens for the pooled
                branch and to ``max_sequence_length`` for T5.
            num_inference_steps: Denoising steps. FLUX.1-dev is tuned for 20-30.
            guidance_scale: Distilled guidance embedding value. This is *not*
                classifier-free guidance -- there is no negative pass, so cost is
                independent of it.
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
        timing = GenerationTiming()
        total_start = time.perf_counter()

        start = time.perf_counter()
        clip_ids, t5_ids = self._tokenize(prompt)
        self._tp.run(ENCODE, clip_ids, t5_ids)
        timing.encode_ms = (time.perf_counter() - start) * 1e3

        start = time.perf_counter()
        timesteps, sigmas = self._timesteps_and_sigmas(num_inference_steps)
        self._tp.run(
            CONTEXT,
            torch.full((1,), guidance_scale, dtype=torch.float32),
            self._init_latents(seed),
        )
        timing.latent_init_ms = (time.perf_counter() - start) * 1e3

        for i, t in enumerate(timesteps):
            step_start = time.perf_counter()
            self._tp.run(
                STEP,
                (t / 1000).reshape(1).to(torch.float32),
                torch.full((1,), float(sigmas[i]), dtype=torch.float32),
                torch.full((1,), float(sigmas[i + 1]), dtype=torch.float32),
            )
            timing.step_ms.append((time.perf_counter() - step_start) * 1e3)
        timing.denoise_ms = sum(timing.step_ms)

        if output_type == "latent":
            latents = self._tp.run(LATENTS)[0].to(self.config.dtype)
            timing.total_ms = (time.perf_counter() - total_start) * 1e3
            return latents, timing

        start = time.perf_counter()
        # Every rank decodes the same image, the VAE being replicated; rank 0's is
        # as good as any.
        image = self._tp.run(DECODE)[0]
        timing.decode_ms = (time.perf_counter() - start) * 1e3

        start = time.perf_counter()
        image = self.image_processor.postprocess(
            image.to(torch.float32), output_type=output_type
        )[0]
        timing.postprocess_ms = (time.perf_counter() - start) * 1e3

        timing.total_ms = (time.perf_counter() - total_start) * 1e3
        return image, timing
