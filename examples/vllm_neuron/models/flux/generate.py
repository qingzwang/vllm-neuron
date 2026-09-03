# SPDX-License-Identifier: Apache-2.0
"""Text-to-image generation with FLUX.1-lite-8B on Neuron.

Unlike the other examples in this directory this does not go through the vLLM
offline API: vLLM has no text-to-image request path, so FLUX runs through
``vllm_neuron.model.flux.NeuronFluxPipeline`` directly. See
``docs/model-recipes/flux-1-lite-8b.md``.

The first run compiles (a few minutes); later runs hit the compilation cache.

Usage:
    python examples/vllm_neuron/models/flux/generate.py \
        --prompt "a red panda reading a book in a cozy library"

    # Faster: fewer steps, shorter prompt budget, lower resolution
    python examples/vllm_neuron/models/flux/generate.py \
        --prompt "..." --steps 8 --max-sequence-length 256 --size 512

    # Run a component on CPU instead of Neuron (A/B comparison)
    python examples/vllm_neuron/models/flux/generate.py \
        --prompt "..." --on-device transformer

    # Shard the transformer over two cores: ~2x faster per step
    python examples/vllm_neuron/models/flux/generate.py --prompt "..." --tp 2
"""

import argparse
import json
import logging
import os
import sys

# The T5 encoder runs on a second logical NeuronCore, in a child process. A
# process that does not restrict itself claims *every* core when the Neuron
# runtime initializes, leaving none for that child -- so pin this one to a single
# core here, before importing vllm_neuron triggers the init. Override
# NEURON_RT_VISIBLE_CORES in the environment to use a different core.
#
# With --tp above 1 the transformer moves into its own worker per rank: ranks take
# cores 0..tp-1, T5 takes core tp, and this process keeps core tp+1 for CLIP and
# the VAE. Cores are exclusive, so that layout has to be settled before the
# import, which is why --tp is read here rather than after parse_args().


def _tp_from_argv() -> int:
    for i, arg in enumerate(sys.argv):
        if arg == "--tp" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith("--tp="):
            return int(arg.split("=", 1)[1])
    return 1


_TP = _tp_from_argv()
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", str(_TP + 1) if _TP > 1 else "0")
os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "3600")

from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline
from vllm_neuron.model.flux.config import DEFAULT_ON_DEVICE

DEFAULT_PROMPT = (
    "A close-up photo of a red panda wearing tiny round glasses, reading a "
    "leather-bound book in a cozy library, warm afternoon light, shallow "
    "depth of field"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        default="Freepik/flux.1-lite-8B",
        help="HF repo id or local path to a diffusers FluxPipeline folder.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", default="flux_output.png")
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Square output resolution. Must be a multiple of 16.",
    )
    parser.add_argument("--steps", type=int, default=28, help="Denoising steps.")
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Distilled guidance embedding value (not classifier-free "
        "guidance: cost does not depend on it).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="T5 prompt budget. Shorter means a shorter attention sequence and "
        "a faster step.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--on-device",
        default=",".join(DEFAULT_ON_DEVICE),
        help="Comma-separated components to place on Neuron. Anything omitted "
        "runs on CPU in eager mode.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Shard the transformer across this many NeuronCores (power of two, "
        "and it must divide the 24 attention heads). tp=4 needs eight logical "
        "cores, i.e. NEURON_LOGICAL_NC_CONFIG=1.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=1,
        help="neuronx-cc -O level. Higher may run faster but compiles slower.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the discarded single-step request. That request exists only "
        "so the reported latency is warm: it loads every NEFF onto the device, "
        "which costs ~20 s once per process and would otherwise be charged to "
        "the image you asked for.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    config = FluxNeuronConfig(
        height=args.size,
        width=args.size,
        max_sequence_length=args.max_sequence_length,
        on_device=tuple(c for c in args.on_device.split(",") if c),
        optimization_level=args.optimization_level,
        tp_degree=args.tp,
        tp_core_ids=tuple(range(args.tp)) if args.tp > 1 else None,
        worker_device_index=args.tp if args.tp > 1 else 1,
    )
    # `with` releases the T5 worker's NeuronCore on the way out.
    with NeuronFluxPipeline.from_pretrained(args.model_checkpoint, config) as pipeline:
        pipeline.compile()

        if not args.no_warmup:
            print("Warming up (loading NEFFs onto the device)...")
            pipeline.generate(args.prompt, num_inference_steps=1, seed=args.seed)

        image, timing = pipeline.generate(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
    image.save(args.output)

    print(f"\nSaved {args.size}x{args.size} image to {args.output}")
    print(f"Placement:    {json.dumps(pipeline.placement)}")
    print(
        f"Compile time: "
        f"{json.dumps({k: round(v / 1e3, 1) for k, v in pipeline.compile_ms.items()})} s"
    )
    label = "Latency (cold)" if args.no_warmup else "Latency (warm)"
    print(
        f"{label}: {timing.total_ms / 1e3:.2f} s total "
        f"({timing.median_step_ms:.0f} ms/step x {args.steps} steps, "
        f"encode {timing.encode_ms:.0f} ms, decode {timing.decode_ms:.0f} ms)"
    )


if __name__ == "__main__":
    main()
