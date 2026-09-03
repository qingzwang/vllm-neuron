# SPDX-License-Identifier: Apache-2.0
"""The FLUX transformer sharded across NeuronCores, in worker processes.

One process drives one logical core -- the compile backend loads every NEFF onto
the process's own core -- so tensor parallelism means one process per rank. This
module holds both halves of that: what each worker runs, and the handle the
pipeline uses to drive them.

The workers come from ``vllm_neuron.utils.executor.MPExecutor``, the same pool the
LLM path uses, which spawns a process per rank, pins it, brings up the process
group and initializes the Neuron parallel state that ``cpl``/``rpl`` and this
module's collectives read.

## Keeping the loop off the host

MPExecutor's protocol is one broadcast per call, so a naive port would ship every
step's inputs to every rank: prompt embeddings alone are 4 MiB at a 512-token
budget, which is more traffic per step than the step's own latents. Instead each
worker keeps the request's invariants -- prompt embeddings, pooled projection,
guidance, RoPE tables -- and the latents themselves, so a step sends three scalars
and gets back a one-element fence. The latents come home once, after the last
step.

That is what the ``mode`` first argument is for:

| ``mode`` | Sends | Does |
|---|---|---|
| ``"context"`` | request invariants and initial latents | stores them on device |
| ``"step"`` | timestep, sigma, sigma_next | one denoising step, latents updated in place |
| ``"latents"`` | nothing | returns the final latents |

Only the step is compiled, and it is compiled once: the outer dispatch is plain
Python, so the graph sees fixed shapes and no branches.

## Host memory during load

Every rank loads the full checkpoint before keeping its shard of it, so four ranks
loading at once would need four times 15.2 GiB of host memory at the same moment.
They take a file lock instead, which makes the load sequential -- slower to start,
but it fits.
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
from .parallel import shard_flux_transformer
from .transformer import NeuronFluxTransformer, patch_untraceable_activations

logger = logging.getLogger(__name__)

# Modes the worker accepts. See the module docstring.
_CONTEXT, _STEP, _LATENTS = "context", "step", "latents"


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


class _RankStep(nn.Module):
    """One rank's shard of the transformer, plus the state of the request.

    Not itself compiled: it dispatches on ``mode`` and calls the compiled step,
    so the graph stays branch-free.

    Args:
        step: The rank's ``NeuronFluxTransformer``, already sharded and on device.
        config: Shape config, for the compiler arguments.
    """

    def __init__(self, step: NeuronFluxTransformer, config: FluxNeuronConfig) -> None:
        super().__init__()
        from vllm_neuron.envs import get_compile_backend_name

        torch._dynamo.config.cache_size_limit = 2**62
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 2**62

        self.compiled = torch.compile(
            step,
            backend=get_compile_backend_name(),
            fullgraph=True,
            options={
                "alias_meta_to_neuron": True,
                "compiler_args": config.neuronx_cc_args(),
            },
        )
        self.latents: torch.Tensor | None = None
        self._context: tuple[torch.Tensor, ...] = ()

    def forward(self, mode: str, *args: torch.Tensor) -> torch.Tensor:
        if mode == _CONTEXT:
            prompt_embeds, pooled, guidance, freqs_cos, freqs_sin, latents = args
            self._context = (prompt_embeds, pooled, guidance, freqs_cos, freqs_sin)
            self.latents = latents
            return latents.reshape(-1)[:1]

        if mode == _STEP:
            timestep, sigma, sigma_next = args
            prompt_embeds, pooled, guidance, freqs_cos, freqs_sin = self._context
            out, tag = self.compiled(
                self.latents,
                prompt_embeds,
                pooled,
                timestep,
                guidance,
                freqs_cos,
                freqs_sin,
                sigma,
                sigma_next,
            )
            self.latents = out
            # A one-element view: reading it on the host is the cheapest fence
            # against the runtime's asynchronous execution queue.
            return tag

        if mode == _LATENTS:
            return self.latents

        raise ValueError(f"unknown mode {mode!r}")


def load_rank_step(model_path: str, config: FluxNeuronConfig) -> _RankStep:
    """Build this rank's shard. Runs inside a worker; used as ``model_load``.

    Args:
        model_path: HF repo id or local path of the ``FluxPipeline`` folder.
        config: Shape and placement config, shared by every rank.

    Returns:
        The rank's step module, on device with its graph compiled.
    """
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    with _load_serialized():
        transformer = FluxTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=config.dtype
        )
        transformer.requires_grad_(False).eval()
        patch_untraceable_activations(transformer)
        shard_flux_transformer(transformer)

    step = NeuronFluxTransformer(transformer, config)
    step.to(torch.device("neuron", 0))
    return _RankStep(step, config)


class TPTransformer:
    """Parent-side handle on the sharded transformer.

    Spawns one worker per rank and compiles the step in each, so construction is
    where compilation cost and any failure surface -- as with the in-process path,
    which compiles from a warmup call at load time.

    Args:
        model_path: Checkpoint the workers load from.
        config: Shape and placement config. ``tp_degree`` and ``tp_core_ids``
            decide the world.
        warmup_args: Synthetic ``(prompt_embeds, pooled, guidance, freqs_cos,
            freqs_sin, latents)`` at the exact runtime shapes, used to compile.

    Raises:
        RuntimeError: If a worker fails to load, shard or compile.
    """

    def __init__(
        self,
        model_path: str,
        config: FluxNeuronConfig,
        warmup_args: tuple[torch.Tensor, ...],
    ) -> None:
        from functools import partial

        from vllm_neuron.utils.executor import MPExecutor

        self.config = config
        core_ids = ",".join(str(core) for core in config.tp_core_ids)
        logger.info(
            "Starting %d transformer workers on cores %s",
            config.tp_degree,
            core_ids,
        )

        # MPExecutor reads NEURON_RT_VISIBLE_CORES to decide which core each
        # worker gets. This process latched its own core when the runtime came up
        # at import, so setting it now only reaches the children -- but restore
        # it immediately, because anything else in this process that reads it
        # would otherwise be told it owns the workers' cores.
        previous = os.environ.get("NEURON_RT_VISIBLE_CORES")
        os.environ["NEURON_RT_VISIBLE_CORES"] = core_ids
        try:
            self.executor = MPExecutor(
                world_size=config.tp_degree,
                model_load=partial(load_rank_step, model_path, config),
            )
        finally:
            if previous is None:
                os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
            else:
                os.environ["NEURON_RT_VISIBLE_CORES"] = previous

        # Compile by running one context and one step, the same warmup-driven
        # compilation the in-process components use.
        self.set_context(*warmup_args)
        self.step(
            torch.ones(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        )

    def _run(self, mode: str, *args: torch.Tensor) -> list:
        self.executor.dispatch(mode, *args)
        return self.executor.collect()

    def set_context(
        self,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        guidance: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        latents: torch.Tensor,
    ) -> None:
        """Give every rank the invariants and initial latents for one request."""
        self._run(_CONTEXT, prompt_embeds, pooled, guidance, freqs_cos, freqs_sin, latents)

    def step(
        self, timestep: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor
    ) -> None:
        """Advance every rank by one denoising step.

        Returns once the workers have, since collecting their fence tensors is
        what waits for the device.
        """
        self._run(_STEP, timestep, sigma, sigma_next)

    def latents(self) -> torch.Tensor:
        """The latents after the last step, from rank 0.

        Every rank holds the same latents: the residual stream is replicated and
        the collectives make each step's output identical across ranks.
        """
        return self._run(_LATENTS)[0].to(self.config.dtype)

    def close(self) -> None:
        """Stop the workers and release their cores."""
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown()
            self.executor = None
