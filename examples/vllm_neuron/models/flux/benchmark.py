# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for FLUX.1-lite-8B on Neuron.

Reports the per-stage breakdown of a request -- prompt encode, per denoising
step, VAE decode -- plus percentiles over repeated requests, and writes the raw
numbers to JSON.

The first request of a process is discarded by default. It pays for loading each
NEFF onto the device, which is a one-off tens of seconds and would otherwise
dominate the average; ``--include-warmup`` keeps it and reports it separately.

Usage:
    python examples/vllm_neuron/models/flux/benchmark.py --iterations 3

    # Sweep resolutions and step counts
    python examples/vllm_neuron/models/flux/benchmark.py \
        --sizes 512,1024 --steps 8,28 --iterations 3 --json results.json
"""

import argparse
import gc
import json
import logging
import os
import statistics
import time
from typing import Any

os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "3600")

from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline
from vllm_neuron.model.flux.config import DEFAULT_ON_DEVICE

PROMPT = (
    "A close-up photo of a red panda wearing tiny round glasses, reading a "
    "leather-bound book in a cozy library, warm afternoon light, shallow "
    "depth of field"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-checkpoint", default="Freepik/flux.1-lite-8B")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--sizes",
        default="1024",
        help="Comma-separated square resolutions to sweep.",
    )
    parser.add_argument(
        "--steps",
        default="28",
        help="Comma-separated denoising step counts to sweep.",
    )
    parser.add_argument(
        "--max-sequence-length", type=int, default=512, help="T5 prompt budget."
    )
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Measured requests per configuration, after warmup.",
    )
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Count the first (NEFF-loading) request in the statistics too.",
    )
    parser.add_argument("--on-device", default=",".join(DEFAULT_ON_DEVICE))
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument(
        "--save-images",
        default=None,
        help="Directory to write one image per configuration into.",
    )
    parser.add_argument("--json", default=None, help="Write raw results here.")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; iteration counts here are far too small for
    interpolation to mean anything."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def run_config(args: argparse.Namespace, size: int, steps: int) -> dict[str, Any]:
    """Compile one configuration and time ``iterations`` requests through it."""
    config = FluxNeuronConfig(
        height=size,
        width=size,
        max_sequence_length=args.max_sequence_length,
        on_device=tuple(c for c in args.on_device.split(",") if c),
        optimization_level=args.optimization_level,
    )
    pipeline = NeuronFluxPipeline.from_pretrained(args.model_checkpoint, config)

    load_start = time.perf_counter()
    pipeline.compile()
    compile_s = time.perf_counter() - load_start

    runs = []
    image = None
    total_iterations = args.iterations + (0 if args.include_warmup else 1)
    for i in range(total_iterations):
        image, timing = pipeline.generate(
            args.prompt,
            num_inference_steps=steps,
            guidance_scale=args.guidance,
            seed=42 + i,
        )
        runs.append(timing)
        print(
            f"  [{size}px/{steps} steps] iter {i}: "
            f"{timing.total_ms / 1e3:.2f} s total, "
            f"{timing.median_step_ms:.0f} ms/step",
            flush=True,
        )

    warmup = None if args.include_warmup else runs.pop(0)

    if args.save_images and image is not None:
        os.makedirs(args.save_images, exist_ok=True)
        path = os.path.join(args.save_images, f"flux_{size}px_{steps}steps.png")
        image.save(path)
        print(f"  wrote {path}")

    totals = [r.total_ms for r in runs]
    # Pooled across requests: within one request the steps are identical work,
    # so the spread that matters is over all of them, not over per-request means.
    all_steps = [ms for r in runs for ms in r.step_ms]
    result = {
        "size": size,
        "steps": steps,
        "max_sequence_length": args.max_sequence_length,
        "image_seq_len": config.image_seq_len,
        "joint_seq_len": config.joint_seq_len,
        "placement": dict(pipeline.placement),
        "compile_s": round(compile_s, 1),
        "compile_s_per_component": {
            k: round(v / 1e3, 1) for k, v in pipeline.compile_ms.items()
        },
        "warmup_request_s": round(warmup.total_ms / 1e3, 2) if warmup else None,
        "iterations": len(runs),
        "total_s": {
            "mean": round(statistics.fmean(totals) / 1e3, 3),
            "p50": round(percentile(totals, 0.50) / 1e3, 3),
            "p90": round(percentile(totals, 0.90) / 1e3, 3),
            "min": round(min(totals) / 1e3, 3),
            "max": round(max(totals) / 1e3, 3),
        },
        "step_ms": {
            "mean": round(statistics.fmean(all_steps), 2),
            "p50": round(percentile(all_steps, 0.50), 2),
            "p90": round(percentile(all_steps, 0.90), 2),
            "min": round(min(all_steps), 2),
            "max": round(max(all_steps), 2),
        },
        "stage_ms": {
            stage: round(statistics.fmean([getattr(r, f"{stage}_ms") for r in runs]), 2)
            for stage in ("encode", "latent_init", "denoise", "decode", "postprocess")
        },
        "runs": [r.as_dict() for r in runs],
    }

    # Sweeping builds one pipeline per configuration, so drop this one's weights
    # before the next allocates 16 GB of HBM on the same core.
    del pipeline, image
    gc.collect()
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    step_counts = [int(s) for s in args.steps.split(",")]

    results = []
    for size in sizes:
        for steps in step_counts:
            print(f"\n=== {size}x{size}, {steps} steps ===", flush=True)
            results.append(run_config(args, size, steps))

    print(f"\n{'resolution':>12} {'steps':>6} {'ms/step':>9} {'p90':>8} {'total s':>9}")
    for r in results:
        print(
            f"{r['size']:>9}px {r['steps']:>6} {r['step_ms']['p50']:>9.1f} "
            f"{r['step_ms']['p90']:>8.1f} {r['total_s']['p50']:>9.2f}"
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
