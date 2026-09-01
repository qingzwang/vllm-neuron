# SPDX-License-Identifier: Apache-2.0
"""T5 prompt encoder on a second logical NeuronCore, in its own process.

Why a process and not just another device index
-----------------------------------------------
The two big components do not fit on one logical core. A core addresses ~22 GiB
of usable trn2 HBM (measured; 24 GB nominal at ``logical-neuroncore-config: 2``,
less the runtime's reserve), while BF16 FLUX.1-lite's transformer is 15.2 GiB and
T5-XXL is 8.9 GiB before any activation memory.

Splitting them across cores inside one process does not work either. The compile
backend loads every NEFF at ``start_nc = 0`` offset by the process's distributed
rank, so a process owns exactly one core; uploading weights to a second core and
then executing gets rejected by the runtime ("Tensor allocated on HBM N was
passed to a model loaded on lnc 0"). One core per process is the shape the stack
gives us, so a second core means a second process.

The child is launched with ``NEURON_RT_VISIBLE_CORES`` set, which remaps its
core 0 onto whichever physical core we picked. That is set through the
subprocess environment rather than by mutating this process's ``os.environ``:
importing ``vllm_neuron`` initializes the Neuron backend, so the variable has to
be in place before the child's interpreter starts, not after.

This mirrors, in miniature, what ``vllm_neuron/vllm/disaggregated_encoder`` does
for vision encoders in the LLM path -- an encoder pool separate from the pool
that runs the main model.

Protocol
--------
Length-prefixed pickles: requests on the child's stdin, replies on a private
duplicate of its stdout. The child points fd 1 at stderr before importing
anything, because importing ``vllm_neuron`` logs to stdout -- left alone, those
log lines land in the middle of the binary stream and the first length prefix
read is whatever ASCII happens to be there. Everything the child prints
thereafter, its own logs included, goes to the inherited stderr.

Config reaches the child through the environment rather than argv: the compiler
flags are themselves ``--``-prefixed, which argparse reads as options rather than
values.

Only token ids go out (4 KiB) and only embeddings come back (4 MiB at a 512-token
budget), so the transport is not on the critical path.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import struct
import subprocess
import sys
import traceback
from typing import Any, Self

import torch

logger = logging.getLogger(__name__)

_LENGTH_PREFIX = struct.Struct(">Q")

# Largest plausible message: embeddings are 4 MiB at a 512-token budget. Anything
# far above that means the stream is not carrying our frames.
_MAX_MESSAGE_BYTES = 256 * 1024 * 1024

_ENV_MODEL_PATH = "VLLM_NEURON_FLUX_T5_MODEL_PATH"
_ENV_SEQ_LEN = "VLLM_NEURON_FLUX_T5_SEQ_LEN"
_ENV_COMPILER_ARGS = "VLLM_NEURON_FLUX_T5_COMPILER_ARGS"

# Runs before vllm_neuron is imported, so the package's stdout logging cannot get
# into the reply stream. fd 1 is moved aside and handed to _serve as the private
# reply channel; fd 1 itself becomes a second handle on stderr.
_BOOTSTRAP = (
    "import os,sys;"
    "reply_fd=os.dup(1);os.dup2(2,1);"
    "from vllm_neuron.model.flux.text_encoder_worker import serve;"
    "sys.exit(serve(reply_fd))"
)

# Generous: on a cold compilation cache the child has to load 8.9 GiB of weights
# and run neuronx-cc before it can answer.
DEFAULT_STARTUP_TIMEOUT_S = 1800.0


def _send(stream, payload: dict[str, Any]) -> None:
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_LENGTH_PREFIX.pack(len(blob)))
    stream.write(blob)
    stream.flush()


def _recv(stream) -> dict[str, Any] | None:
    header = stream.read(_LENGTH_PREFIX.size)
    if not header:
        return None
    (length,) = _LENGTH_PREFIX.unpack(header)
    if length > _MAX_MESSAGE_BYTES:
        # Almost certainly not a length: something wrote plain text into the
        # stream. Say so, rather than trying to allocate it.
        raise RuntimeError(
            f"framed message claims {length} bytes, above the "
            f"{_MAX_MESSAGE_BYTES} limit; the protocol stream is corrupt "
            f"(header bytes: {header!r})"
        )
    blob = stream.read(length)
    if len(blob) != length:
        return None
    return pickle.loads(blob)


def parse_visible_cores(spec: str) -> set[int]:
    """Parse a ``NEURON_RT_VISIBLE_CORES`` spec into core indices.

    Accepts the runtime's forms: a single index, a comma-separated list, and
    inclusive ranges (``"0"``, ``"0,2"``, ``"0-3"``, ``"0,2-3"``).

    Args:
        spec: The environment variable's value.

    Returns:
        The set of logical core indices it selects. Unparseable fragments are
        skipped rather than raising: this feeds a diagnostic, and guessing wrong
        about a spec the runtime accepts should not break the pipeline.
    """
    cores: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                low, high = part.split("-", 1)
                cores.update(range(int(low), int(high) + 1))
            else:
                cores.add(int(part))
        except ValueError:
            continue
    return cores


def explain_core_conflict(worker_core: int) -> str | None:
    """Check whether a worker could claim ``worker_core``, from this process.

    The Neuron runtime hands a core to one process at a time. A process that
    does not restrict itself claims *every* logical core when it initializes,
    even if it only ever runs graphs on one -- so the default configuration
    leaves nothing for a worker, and the worker's failure surfaces as an opaque
    ``nrt_init`` error inside the child.

    This is checked up front so the reason is stated once, in the parent, before
    tens of seconds are spent on a launch that cannot succeed.

    Args:
        worker_core: The logical core the worker wants.

    Returns:
        A sentence explaining why it cannot be claimed, or ``None`` if it can.
    """
    visible = os.environ.get("NEURON_RT_VISIBLE_CORES")
    num_cores = os.environ.get("NEURON_RT_NUM_CORES")

    fix = (
        "Set NEURON_RT_VISIBLE_CORES to just the pipeline's own core (e.g. "
        '"0") before vllm_neuron is imported, which is when the runtime '
        "initializes."
    )
    if visible is None and num_cores is None:
        return (
            "this process claimed every logical NeuronCore when the Neuron "
            "runtime initialized, because neither NEURON_RT_VISIBLE_CORES nor "
            f"NEURON_RT_NUM_CORES is set, so core {worker_core} is not free. " + fix
        )
    if visible is not None and worker_core in parse_visible_cores(visible):
        return (
            f"NEURON_RT_VISIBLE_CORES={visible!r} includes core {worker_core}, "
            "so this process holds it; the worker needs a core this process "
            "does not. " + fix
        )
    if visible is None and num_cores is not None:
        try:
            claimed = int(num_cores)
        except ValueError:
            return None
        if worker_core < claimed:
            return (
                f"NEURON_RT_NUM_CORES={num_cores!r} means this process claimed "
                f"cores 0..{claimed - 1}, which includes core {worker_core}. " + fix
            )
    return None


class TextEncoderWorker:
    """Client for a T5 encoder running on another logical NeuronCore.

    Args:
        model_path: HF repo id or local path to the ``FluxPipeline`` folder; the
            child loads the ``text_encoder_2`` subfolder from it.
        device_index: Physical logical-core index for the child to run on. Must
            differ from the core the pipeline itself uses.
        max_sequence_length: Prompt budget, and the exact static shape the child
            compiles for.
        compiler_args: ``neuronx-cc`` flags, so the child compiles with the same
            settings as the rest of the pipeline.
        startup_timeout_s: How long to wait for the child to finish loading and
            compiling before giving up.

    Raises:
        RuntimeError: From :meth:`start` if the child fails to become ready, and
            from :meth:`encode` if it dies or reports an error. The pipeline
            treats either as a signal to fall back to its own CPU copy.
    """

    def __init__(
        self,
        model_path: str,
        device_index: int,
        max_sequence_length: int,
        compiler_args: list[str],
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    ) -> None:
        self.model_path = model_path
        self.device_index = device_index
        self.max_sequence_length = max_sequence_length
        self.compiler_args = compiler_args
        self.startup_timeout_s = startup_timeout_s
        self._process: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the child and block until it has compiled and warmed up."""
        env = dict(os.environ)
        env["NEURON_RT_VISIBLE_CORES"] = str(self.device_index)
        # The child must import the same vllm_neuron and diffusers this process
        # did, which are not necessarily on the default path.
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        env[_ENV_MODEL_PATH] = self.model_path
        env[_ENV_SEQ_LEN] = str(self.max_sequence_length)
        env[_ENV_COMPILER_ARGS] = json.dumps(self.compiler_args)

        logger.info(
            "Starting T5 encoder worker on logical NeuronCore %d", self.device_index
        )
        self._process = subprocess.Popen(
            [sys.executable, "-c", _BOOTSTRAP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )

        # No incremental progress to poll for, so this is a blocking read on the
        # child's first message; the timeout is enforced by the caller's own
        # patience plus the child exiting on failure.
        reply = self._exchange(None, expect_ready=True)
        if reply.get("status") != "ready":
            raise RuntimeError(
                f"T5 encoder worker failed to start: {reply.get('error')}\n"
                f"{reply.get('traceback', '')}"
            )
        logger.info("T5 encoder worker ready on core %d", self.device_index)

    def close(self) -> None:
        """Ask the child to exit, then make sure it did."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            _send(process.stdin, {"op": "shutdown"})
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("T5 encoder worker did not exit; terminating")
            process.kill()
            process.wait(timeout=30)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode one padded prompt.

        Args:
            input_ids: ``[1, max_sequence_length]`` int64 token ids, on the host.

        Returns:
            ``[1, max_sequence_length, joint_attention_dim]`` embeddings, on the
            host.
        """
        reply = self._exchange({"op": "encode", "input_ids": input_ids})
        return reply["embeds"]

    def _exchange(
        self, request: dict[str, Any] | None, expect_ready: bool = False
    ) -> dict[str, Any]:
        process = self._process
        if process is None:
            raise RuntimeError("T5 encoder worker is not running")
        try:
            if request is not None:
                _send(process.stdin, request)
            reply = _recv(process.stdout)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"T5 encoder worker connection lost: {exc}") from exc
        if reply is None:
            code = process.poll()
            raise RuntimeError(
                f"T5 encoder worker exited (returncode={code}) without replying"
            )
        if reply.get("status") == "error":
            raise RuntimeError(
                f"T5 encoder worker error: {reply.get('error')}\n"
                f"{reply.get('traceback', '')}"
            )
        if not expect_ready and reply.get("status") != "ok":
            raise RuntimeError(f"unexpected reply from T5 encoder worker: {reply}")
        return reply


# ----------------------------------------------------------------------
# Child process
# ----------------------------------------------------------------------


def serve(reply_fd: int) -> int:
    """Child entry point.

    Args:
        reply_fd: Duplicate of the original stdout, opened by the bootstrap
            before any import could write to it. Replies go here; fd 1 now
            points at stderr, so anything the child or its libraries print is
            kept out of the protocol stream.

    Returns:
        Process exit status.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s t5-worker: %(message)s",
        stream=sys.stderr,
    )

    requests = sys.stdin.buffer
    replies = os.fdopen(reply_fd, "wb")

    model_path = os.environ[_ENV_MODEL_PATH]
    max_sequence_length = int(os.environ[_ENV_SEQ_LEN])
    compiler_args = json.loads(os.environ[_ENV_COMPILER_ARGS])

    try:
        from transformers import T5EncoderModel

        from vllm_neuron.envs import get_compile_backend_name

        # Index 0 of what this process can see, which NEURON_RT_VISIBLE_CORES has
        # already remapped onto the core the parent picked.
        device = torch.device("neuron", 0)

        logger.info("Loading T5 encoder from %s", model_path)
        encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2", dtype=torch.bfloat16
        )
        encoder.requires_grad_(False).eval()
        encoder.to(device)

        compiled = torch.compile(
            _T5Encoder(encoder),
            backend=get_compile_backend_name(),
            fullgraph=True,
            options={
                "alias_meta_to_neuron": True,
                "compiler_args": compiler_args,
            },
        )

        logger.info("Compiling for a %d-token prompt", max_sequence_length)
        with torch.no_grad():
            warmup = torch.zeros(
                1, max_sequence_length, dtype=torch.long, device=device
            )
            _, tag = compiled(warmup)
            tag.cpu()
    except Exception as exc:  # surfaced to the parent, which falls back to CPU
        _send(replies, _error_payload(exc))
        return 1

    _send(replies, {"status": "ready"})
    logger.info("Ready")

    while True:
        try:
            request = _recv(requests)
        except (RuntimeError, EOFError, OSError):
            logger.exception("Malformed request stream; exiting")
            return 1
        if request is None or request.get("op") == "shutdown":
            return 0
        try:
            with torch.no_grad():
                embeds, tag = compiled(request["input_ids"].to(device))
                tag.cpu()
                _send(replies, {"status": "ok", "embeds": embeds.cpu()})
        except Exception as exc:
            # The compiled graph is fixed-shape and stateless, so a failure here
            # is not something the next request would recover from.
            _send(replies, _error_payload(exc))
            return 1


def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "error",
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }


class _T5Encoder(torch.nn.Module):
    """Static-shape wrapper so the T5 encoder compiles as one graph.

    FLUX pads the prompt to a fixed length and passes no attention mask, so the
    only input is a token-id tensor of constant shape.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.inner(input_ids, output_hidden_states=False)[0]
        # Second output is the host-side fence: NEFF execution is queued, so the
        # parent reads one element to wait for this graph.
        return out, out.reshape(-1)[:1]
