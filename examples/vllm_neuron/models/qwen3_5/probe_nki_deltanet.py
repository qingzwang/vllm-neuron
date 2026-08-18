#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compare the NKI DeltaNet kernel against the torch reference, on device.

The torch ``chunk_gated_delta_rule`` is the validated reference: it matches HF to
~1e-7 in float32 on CPU. This asks two questions about replacing it with the
vendored NKI kernel on device:

1. **Does it agree?** Same inputs through both paths on the device, so the only
   difference is the kernel. Tolerance is loose because the kernel accumulates in
   float32 with a 128-wide chunk where the torch path uses 64 — the same maths,
   different grouping.
2. **Is it faster?** Timed both ways. Profiling put the torch delta rule at 72-78%
   of prefill, so this is the number the whole port turns on.

``VLLM_NEURON_QWEN35_ENABLE_NKI=1`` selects the kernel; torch is the default. The
answer this script produced: correct (1.2e-05) but 0.10x the speed at seq 1024 and
0.08x at 4096, so the torch path ships.

Usage:

    NEURON_SKIP_EFA_AFFINITY=1 VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm \
    NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp" \
    PYTHONPATH=/mnt/nvme/vllm-neuron PATH=$V/bin:$PATH $V/bin/python \
      probe_nki_deltanet.py [--seq 1024] [--heads 4] [--time 10]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

DEVICE = "neuron:0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument(
        "--heads", type=int, default=4, help="value heads per rank (16 / TP)"
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--time", type=int, default=10)
    parser.add_argument("--tol", type=float, default=5e-2)
    return parser.parse_args()


def make_inputs(args):
    """Inputs in the ranges the real checkpoint produces (see probe_device_ops)."""
    from vllm_neuron.model.qwen3_5.deltanet import l2norm

    torch.manual_seed(0)
    h, t, d = args.heads, args.seq, args.dim
    return (
        l2norm(torch.randn(1, h, t, d)),
        l2norm(torch.randn(1, h, t, d)),
        torch.randn(1, h, t, d) * 0.1,
        -torch.rand(1, h, t) * 4.6,
        torch.rand(1, h, t),
        torch.zeros(1, h, d, d),
    )


def run(args, use_nki: bool):
    """Compile and run the delta rule one way; return (output, state, ms)."""
    import vllm_neuron  # noqa: F401
    from vllm_neuron.compile.backend import compile as neuron_compile
    from vllm_neuron.model.qwen3_5 import deltanet as dn

    os.environ["VLLM_NEURON_QWEN35_ENABLE_NKI"] = "1" if use_nki else "0"
    cpu_args = make_inputs(args)
    device_args = tuple(a.to(DEVICE) for a in cpu_args)

    compiled = torch.compile(
        dn.chunk_gated_delta_rule, backend=neuron_compile, dynamic=False
    )
    out, state = compiled(*device_args)
    out_cpu, state_cpu = out.cpu(), state.cpu()

    # Sync every iteration: the runtime queue is shallow and unsynchronised
    # submissions fail with "Execution Queue Full".
    started = time.perf_counter()
    for _ in range(args.time):
        compiled(*device_args)[0].cpu()
    ms = (time.perf_counter() - started) * 1000.0 / args.time
    return out_cpu.float(), state_cpu.float(), ms


def report(name, ours, theirs, tol) -> bool:
    scale = theirs.abs().max().clamp(min=1e-9)
    rel = ((ours - theirs).abs().max() / scale).item()
    ok = rel <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:16s} rel={rel:.3e}")
    return ok


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    print(
        f"heads={args.heads} seq={args.seq} dim={args.dim} on {DEVICE}, "
        f"{args.time} timed calls\n"
    )

    torch_out, torch_state, torch_ms = run(args, use_nki=False)
    print(f"  torch  {torch_ms:8.2f} ms/call")
    nki_out, nki_state, nki_ms = run(args, use_nki=True)
    print(f"  nki    {nki_ms:8.2f} ms/call   speedup {torch_ms / nki_ms:.2f}x\n")

    ok = report("output", nki_out, torch_out, args.tol)
    ok &= report("final state", nki_state, torch_state, args.tol)

    print("\n" + ("AGREES" if ok else "DISAGREES — do not ship"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
