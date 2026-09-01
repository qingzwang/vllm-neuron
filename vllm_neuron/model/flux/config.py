# SPDX-License-Identifier: Apache-2.0
"""Neuron configuration for the FLUX.1 diffusion transformer pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

# Components of a diffusers FluxPipeline that can be placed on Neuron.
FLUX_COMPONENTS = ("transformer", "vae", "text_encoder", "text_encoder_2")

# Every component runs on Neuron by default. T5-XXL cannot share a core with the
# transformer, so it runs on a second one via a worker process; see
# FluxNeuronConfig.text_encoder_worker.
DEFAULT_ON_DEVICE = FLUX_COMPONENTS

# VAE spatial compression for the FLUX AutoencoderKL (8x), and the 2x2 patchify
# the pipeline applies on top of it before feeding the transformer.
VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2


def lnc_compiler_arg() -> list[str]:
    """``--lnc`` for ``neuronx-cc``, matching this process's runtime setting.

    A NEFF records the logical-core config it was built for, and the runtime
    refuses to load one that disagrees: "Runtime currently configured with
    `NEURON_LOGICAL_NC_CONFIG=1` but NEFF ... was compiled with `--lnc=2`".

    Passing the flag explicitly is what keeps that from happening silently.
    ``NEURON_LOGICAL_NC_CONFIG`` is not part of the compilation cache key, but
    the compiler arguments are -- so mirroring the runtime setting into a flag
    both compiles the right NEFF and gives each config its own cache entry.
    Without it, switching to LNC=1 replays the LNC=2 NEFFs from cache and every
    component falls back to CPU.

    Returns:
        ``["--lnc=<n>"]`` when the environment sets a logical-core config, else
        an empty list, which leaves the compiler on its default.
    """
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    return [f"--lnc={lnc}"] if lnc else []


@dataclass
class FluxNeuronConfig:
    """Static shape and placement config for a Neuron FLUX pipeline.

    Unlike the autoregressive models in this package, a diffusion transformer has
    no KV cache and no sequence-length bucketing: image resolution and prompt
    length are fixed for the lifetime of the pipeline, so every denoising step
    replays the exact same graph. That makes one NEFF per (resolution, prompt
    length) pair sufficient — which is why those two live here and are frozen at
    load time.

    Attributes:
        height: Output image height in pixels. Must be a multiple of 16.
        width: Output image width in pixels. Must be a multiple of 16.
        max_sequence_length: T5 prompt length, padded to this exact value so the
            joint attention sequence stays static. FLUX caps this at 512.
        dtype: Compute dtype for all components. Only bfloat16 is supported.
        device_index: Which NeuronCore this process runs on, as an index into the
            cores it can *see*. A process that was pinned with
            ``NEURON_RT_VISIBLE_CORES`` sees only those, renumbered from 0, so
            pinning to core 2 means ``device_index=0``. Leave it at 0 unless this
            process deliberately holds several cores.
        on_device: Which pipeline components to move to Neuron and compile.
            Anything left out stays on CPU in eager PyTorch, which is useful for
            bringing up a component or for A/B latency comparisons. Defaults to
            all of them.
        text_encoder_worker: Run T5-XXL on a second logical NeuronCore in a
            child process (see ``text_encoder_worker.py``). It cannot share the
            pipeline's HBM partition: that holds ~22 GiB and BF16 FLUX.1-lite is
            15.2 GiB of transformer plus 8.9 GiB of T5. And it cannot use a
            second core in this process either, because the compile backend binds
            NEFF execution to the process's own core.

            Set this to False to keep T5 in-process, which means CPU: it will
            try the pipeline's core, fail to allocate, and fall back with a
            warning. That costs ~1.5 s per request.
        worker_device_index: Which core the child process runs on, as a
            *physical* logical-core index -- it becomes the child's
            ``NEURON_RT_VISIBLE_CORES``. Note this is a different namespace from
            ``device_index`` above, and the two cannot be compared: pin this
            process to physical core 3 and its ``device_index`` is still 0.

            The core must be in a different HBM partition from this process's,
            not merely a different core. On trn2 a partition is ~22 GiB shared by
            one physical core pair, so at ``logical-neuroncore-config: 1`` cores
            2k and 2k+1 share a budget and 0/1 would still not fit; 0 and 2
            would.
        fuse_scheduler_step: Fold the FlowMatchEulerDiscreteScheduler update
            into the compiled step graph so latents never leave the device
            during denoising. See NeuronFluxTransformer for the exact update.
            Turning it off runs the upstream scheduler on the host instead, at
            the cost of a round trip per step (~0.5 ms at 512x512, and it grows
            with resolution). Useful for cross-checking the fused update.
        use_nki_attention: Route joint attention through this package's NKI
            flash-attention kernel instead of ``F.scaled_dot_product_attention``.
        optimization_level: ``neuronx-cc`` ``-O`` level (1-3). Lower compiles
            faster, higher may run faster.
    """

    height: int = 1024
    width: int = 1024
    max_sequence_length: int = 512
    dtype: torch.dtype = torch.bfloat16
    device_index: int = 0
    on_device: tuple[str, ...] = DEFAULT_ON_DEVICE
    text_encoder_worker: bool = True
    worker_device_index: int = 1
    fuse_scheduler_step: bool = True
    use_nki_attention: bool = True
    optimization_level: int = 1
    compiler_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dtype != torch.bfloat16:
            raise ValueError(
                f"dtype={self.dtype} is not supported; FLUX on Neuron is BF16 only."
            )
        for name, value in (("height", self.height), ("width", self.width)):
            if value % (VAE_SCALE_FACTOR * PATCH_SIZE) != 0:
                raise ValueError(
                    f"{name}={value} must be a multiple of "
                    f"{VAE_SCALE_FACTOR * PATCH_SIZE} (VAE 8x compression then "
                    "2x2 patchify)."
                )
        if not 1 <= self.max_sequence_length <= 512:
            raise ValueError(
                f"max_sequence_length={self.max_sequence_length} must be in "
                "[1, 512]; FLUX's T5 branch is trained up to 512 tokens."
            )
        unknown = set(self.on_device) - set(FLUX_COMPONENTS)
        if unknown:
            raise ValueError(
                f"on_device contains unknown components {sorted(unknown)}; "
                f"valid components are {list(FLUX_COMPONENTS)}."
            )
        # No check that worker_device_index differs from device_index: they are
        # indices into different things (physical cores vs. this process's
        # visible cores), so comparing them would reject valid setups and accept
        # broken ones. Whether the worker's core is actually free is decided
        # against the environment at start time, in explain_core_conflict.
        if not 1 <= self.optimization_level <= 3:
            raise ValueError(
                f"optimization_level={self.optimization_level} must be in [1, 3]."
            )

    @property
    def latent_height(self) -> int:
        """Latent rows after VAE compression (before patchify)."""
        return self.height // VAE_SCALE_FACTOR

    @property
    def latent_width(self) -> int:
        """Latent columns after VAE compression (before patchify)."""
        return self.width // VAE_SCALE_FACTOR

    @property
    def image_seq_len(self) -> int:
        """Number of image tokens the transformer sees (post 2x2 patchify)."""
        return (self.latent_height // PATCH_SIZE) * (self.latent_width // PATCH_SIZE)

    @property
    def joint_seq_len(self) -> int:
        """Total attention sequence: text tokens prepended to image tokens."""
        return self.max_sequence_length + self.image_seq_len

    def runs_on_device(self, component: str) -> bool:
        if component not in FLUX_COMPONENTS:
            raise ValueError(f"unknown component {component!r}")
        return component in self.on_device

    @property
    def device(self) -> torch.device:
        """The Neuron device on-device components are placed on.

        The index is explicit: privateuse1 rejects an index-less "neuron"
        device, so ``torch.device("neuron")`` alone fails on ``.to()``.
        """
        return torch.device("neuron", self.device_index)

    def device_for(self, component: str) -> torch.device:
        """The Neuron device a given in-process component is placed on.

        Always this process's core: the runtime binds NEFF execution to it, so
        there is no per-component choice to make. T5 escapes that constraint by
        running in a child process instead (``text_encoder_worker``), and is not
        placed through this method.
        """
        if component not in FLUX_COMPONENTS:
            raise ValueError(f"unknown component {component!r}")
        return self.device

    def runs_in_worker(self, component: str) -> bool:
        """Whether ``component`` runs on Neuron via the child process."""
        return (
            component == "text_encoder_2"
            and self.text_encoder_worker
            and self.runs_on_device(component)
        )

    def neuronx_cc_args(self) -> list[str]:
        """Compiler flags for ``torch.compile(options={"compiler_args": ...})``.

        Mirrors the flags NeuronModelRunner passes for the LLM path, minus the
        FP8 and NKI-nested-loop options that only the quantized/MoE models need,
        plus ``--lnc`` to match the runtime (see :func:`lnc_compiler_arg`).
        """
        return [
            "--auto-cast=none",
            f"-O{self.optimization_level}",
            # FIXME: mirrors the LLM path — needed until NKI reports MAC counts
            # for kernels, so the compiler can pick modular flow on its own.
            "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
            *lnc_compiler_arg(),
            *self.compiler_args,
        ]
