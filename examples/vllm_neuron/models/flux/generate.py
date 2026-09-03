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

    # Four cores instead of two: ~1.9x faster per step
    python examples/vllm_neuron/models/flux/generate.py --prompt "..." --tp 4
"""

import argparse
import json
import logging
import os

# Nothing here touches the device: the model is tensor-parallel across
# --tp NeuronCores, one rank process each, and those processes own it. This one
# tokenizes, drives the loop and saves the image. So no core pinning is needed
# for this process -- the ranks pin themselves.
os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "3600")

from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline

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
        "--tp",
        type=int,
        default=2,
        help="How many NeuronCores to shard the model over: 2, 4 or 8. There is "
        "no 1 -- the four components are 24.44 GiB of BF16 weights against a "
        "~22 GiB HBM partition. A trn2.3xlarge has four logical cores.",
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
        optimization_level=args.optimization_level,
        tp_degree=args.tp,
    )
    # `with` releases the ranks' NeuronCores on the way out.
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
    print(f"Parallelism:  tp_degree={config.tp_degree} on cores "
          f"{list(config.tp_core_ids)}")
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
