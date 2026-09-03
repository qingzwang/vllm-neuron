# SPDX-License-Identifier: Apache-2.0
"""The rank processes, from the pipeline's side.

Thin: ``MPExecutor`` already spawns one process per rank, pins it, brings up the
process group and initializes the Neuron parallel state that ``cpl``/``rpl`` and
``parallel.py``'s collectives read. What is left is choosing the cores, loading the
ranks, and turning a command into one broadcast plus one answer per rank.
"""

from __future__ import annotations

import logging
import os
from functools import partial

import torch

from .config import FluxNeuronConfig
from .worker import load_rank

logger = logging.getLogger(__name__)


class TensorParallelFlux:
    """Handle on ``tp_degree`` rank processes holding the sharded model.

    Construction is where weight loading, sharding and compilation happen, so a
    failure surfaces here rather than inside the first request.

    Args:
        model_path: Checkpoint the ranks load from.
        config: Shape and parallelism config. ``tp_degree`` and ``tp_core_ids``
            decide the world.

    Raises:
        RuntimeError: If a rank fails to load, shard or compile. A message about
            NRT and forking means this process touched the device before the ranks
            were started -- it must not.
    """

    def __init__(self, model_path: str, config: FluxNeuronConfig) -> None:
        from vllm_neuron.utils.executor import MPExecutor

        self.config = config
        core_ids = ",".join(str(core) for core in config.tp_core_ids)
        logger.info("Starting %d FLUX ranks on cores %s", config.tp_degree, core_ids)

        # MPExecutor reads NEURON_RT_VISIBLE_CORES to decide which core each rank
        # gets. Set it around the spawn only: it reaches the children, and nothing
        # in this process should conclude from it that it owns those cores.
        previous = os.environ.get("NEURON_RT_VISIBLE_CORES")
        os.environ["NEURON_RT_VISIBLE_CORES"] = core_ids
        try:
            self.executor = MPExecutor(
                world_size=config.tp_degree,
                model_load=partial(load_rank, model_path, config),
            )
        except Exception:
            # A rank that fails to load leaves the pool's processes alive and
            # holding their cores, which then makes the next attempt fail for a
            # different and more confusing reason ("runtime could not be
            # initialized"). Release them here.
            from vllm_neuron.utils.executor import _worker_pool

            _worker_pool.shutdown_all()
            raise
        finally:
            if previous is None:
                os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
            else:
                os.environ["NEURON_RT_VISIBLE_CORES"] = previous

    def run(self, mode: str, *args: torch.Tensor) -> list[torch.Tensor]:
        """Send one command to every rank and collect their answers.

        Args:
            mode: One of the modes in ``worker.py``.
            *args: Host tensors; the workers move them to their own core.

        Returns:
            One tensor per rank, in rank order. Sharded stages return the same
            value on every rank, so callers take rank 0's.
        """
        self.executor.dispatch(mode, *args)
        return self.executor.collect()

    def close(self) -> None:
        """Stop the ranks and release their cores."""
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown()
            self.executor = None
