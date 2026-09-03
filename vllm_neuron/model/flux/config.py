# SPDX-License-Identifier: Apache-2.0
"""Neuron configuration for the FLUX.1 diffusion transformer pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

# VAE spatial compression for the FLUX AutoencoderKL (8x), and the 2x2 patchify
# the pipeline applies on top of it before feeding the transformer.
VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2

# FLUX has 24 attention heads and the T5 encoder has 64, so a TP degree has to
# divide both -- and be a power of two, which is what Neuron's replica groups and
# every TP degree the LLM path supports are.
SUPPORTED_TP_DEGREES = (2, 4, 8)


def lnc_compiler_arg() -> list[str]:
    """``--lnc`` for ``neuronx-cc``, matching this process's runtime setting.

    A NEFF records the logical-core config it was built for, and the runtime
    refuses to load one that disagrees: "Runtime currently configured with
    `NEURON_LOGICAL_NC_CONFIG=1` but NEFF ... was compiled with `--lnc=2`".

    Passing the flag explicitly is what keeps that from happening silently.
    ``NEURON_LOGICAL_NC_CONFIG`` is not part of the compilation cache key, but the
    compiler arguments are -- so mirroring the runtime setting into a flag both
    compiles the right NEFF and gives each setting its own cache entry. Without
    it, switching the setting replays the other NEFFs from cache.

    Note that LNC=1 is not a useful setting for a model this size: each logical
    core is then one physical NeuronCore instead of a fused pair, so a rank gets
    half the compute. Measured at 1227 ms/step against 791 for the same
    unsharded transformer.

    Returns:
        ``["--lnc=<n>"]`` when the environment sets a logical-core config, else an
        empty list, which leaves the compiler on its default.
    """
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    return [f"--lnc={lnc}"] if lnc else []


@dataclass
class FluxNeuronConfig:
    """Static shape and parallelism config for a Neuron FLUX pipeline.

    Unlike the autoregressive models in this package, a diffusion transformer has
    no KV cache and no sequence-length bucketing: image resolution and prompt
    length are fixed for the lifetime of the pipeline, so every denoising step
    replays the exact same graph. That makes one NEFF per (resolution, prompt
    length, tp_degree) sufficient -- which is why those live here and are frozen
    at load time.

    Attributes:
        height: Output image height in pixels. Must be a multiple of 16.
        width: Output image width in pixels. Must be a multiple of 16.
        max_sequence_length: T5 prompt length, padded to this exact value so the
            joint attention sequence stays static. FLUX caps this at 512.
        dtype: Compute dtype for all components. Only bfloat16 is supported.
        tp_degree: How many logical NeuronCores the model is sharded over, one
            rank process each. 2, 4 or 8; it has to divide the transformer's 24
            attention heads and T5's 64, and be a power of two.

            There is no ``tp_degree=1``: the four components together are
            24.44 GiB of BF16 weights against a ~22 GiB HBM partition, so one core
            cannot hold them. 2 is the floor for this checkpoint, and on a
            trn2.3xlarge (four logical cores at the default
            ``logical-neuroncore-config: 2``) 4 is the ceiling.
        tp_core_ids: Physical logical-core ids for the ranks, one each. Defaults
            to ``0..tp_degree-1``. Each becomes that rank's
            ``NEURON_RT_VISIBLE_CORES``.
        lora_slots: How many LoRA adapters to keep resident on device, on top of
            the unmodified model. 0 turns LoRA off entirely and leaves the graph
            exactly as it was. Above 0, the transformer's adaptable layers get slot
            tensors and the step graph reads a device-side index saying which slot
            is live -- so adapters can be loaded and switched at runtime without
            recompiling. See ``lora.py``.

            Slots cost device memory whether or not they hold anything, and the
            cost scales with ``lora_max_rank``. Switching between resident adapters
            is a single 4-byte copy; loading one from disk is not.
        lora_max_rank: Width every slot is allocated at. Adapters may be narrower
            (they are zero-padded, which is exact) but not wider. Real FLUX
            adapters run 8-128.
        fuse_scheduler_step: Fold the FlowMatchEulerDiscreteScheduler update into
            the compiled step graph so latents never leave the device during
            denoising. See NeuronFluxTransformer for the exact update. Turning it
            off is not supported here: the latents stay in the ranks for the whole
            loop, so a host-side scheduler step cannot see them.
        use_nki_attention: Route the transformer's joint attention through this
            package's NKI flash-attention kernel instead of
            ``F.scaled_dot_product_attention``.
        optimization_level: ``neuronx-cc`` ``-O`` level (1-3). Lower compiles
            faster, higher may run faster.
        compiler_args: Extra ``neuronx-cc`` flags, appended last.
    """

    height: int = 1024
    width: int = 1024
    max_sequence_length: int = 512
    dtype: torch.dtype = torch.bfloat16
    tp_degree: int = 2
    tp_core_ids: tuple[int, ...] | None = None
    lora_slots: int = 0
    lora_max_rank: int = 64
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
        if self.tp_degree not in SUPPORTED_TP_DEGREES:
            raise ValueError(
                f"tp_degree={self.tp_degree} is not supported; use one of "
                f"{list(SUPPORTED_TP_DEGREES)}. There is no tp_degree=1: the four "
                "components are 24.44 GiB of BF16 weights and one core's HBM "
                "partition holds ~22 GiB."
            )
        if self.tp_core_ids is None:
            self.tp_core_ids = tuple(range(self.tp_degree))
        else:
            self.tp_core_ids = tuple(self.tp_core_ids)
            if len(self.tp_core_ids) != self.tp_degree:
                raise ValueError(
                    f"tp_core_ids={self.tp_core_ids} has {len(self.tp_core_ids)} "
                    f"entries but tp_degree is {self.tp_degree}; one core per rank."
                )
            if len(set(self.tp_core_ids)) != len(self.tp_core_ids):
                raise ValueError(f"tp_core_ids={self.tp_core_ids} repeats a core.")
        if not self.fuse_scheduler_step:
            raise ValueError(
                "fuse_scheduler_step=False is not supported: the latents stay in "
                "the rank processes for the whole denoising loop, so a host-side "
                "scheduler step cannot see them."
            )
        if self.lora_slots < 0:
            raise ValueError(f"lora_slots={self.lora_slots} cannot be negative.")
        if self.lora_slots and not 1 <= self.lora_max_rank <= 512:
            raise ValueError(
                f"lora_max_rank={self.lora_max_rank} must be in [1, 512]."
            )
        if not 1 <= self.optimization_level <= 3:
            raise ValueError(
                f"optimization_level={self.optimization_level} must be in [1, 3]."
            )

    @property
    def lora_enabled(self) -> bool:
        """Whether the step graph carries LoRA slots."""
        return self.lora_slots > 0

    @property
    def lora_total_slots(self) -> int:
        """Slots including slot 0, which stays zero and means "no adapter"."""
        return self.lora_slots + 1

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
