#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Measure text-only TTFT / TPOT / E2E for Qwen3.5-2B on Neuron.

Uses ``AsyncLLM`` and streams tokens so TTFT is the real time to the *first*
token rather than a whole-request latency divided by something. TPOT is measured
over the remaining tokens only, which is what makes it comparable to the NxDI
reference's published TP=4 figures (TTFT 42.2 ms, TPOT 4.75 ms, 210 tok/s at
seq_len 1024).

Input length is pinned by tokenising filler text and truncating, so a run at
``--input-tokens 1024`` really is a 1024-token prompt. Note the plugin compiles
one graph per batch bucket, so ``--max-num-seqs`` must match the concurrency you
want to measure — measure batch 1 and batch 4 in separate processes rather than
configuring several buckets (multiple ``num_seqs_buckets`` hangs on this plugin).

Usage:

    NEURON_SKIP_EFA_AFFINITY=1 VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm \
    NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp" \
    PYTHONPATH=/mnt/nvme/vllm-neuron PATH=$V/bin:$PATH $V/bin/python \
      benchmark_latency.py --max-num-seqs 1 --input-tokens 1024 --output-tokens 128
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1800")

FILLER = (
    "The history of computing is a long sequence of small steps and occasional "
    "leaps, in which each generation of engineers rediscovers that the hard part "
    "was never the arithmetic but the bookkeeping around it. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--vision-bucket",
        type=int,
        default=0,
        help="non-zero switches to a single-image prompt and builds the VL model, "
        "so TTFT then includes the vision tower",
    )
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def build_prompt(model_path: str, num_tokens: int, salt: int) -> str:
    """Filler text truncated to exactly ``num_tokens`` tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # A distinct prefix per request so nothing can be served from a cache.
    text = f"Note {salt}. " + FILLER * (num_tokens // 20 + 4)
    ids = tokenizer(text, add_special_tokens=False).input_ids[:num_tokens]
    return tokenizer.decode(ids)


def build_vl_prompt(model_path: str, image_size: int):
    """A single-image chat prompt. One item, so the vision bucket stays small."""
    from transformers import AutoProcessor
    from vllm.assets.image import ImageAsset

    processor = AutoProcessor.from_pretrained(model_path)
    image = ImageAsset("cherry_blossom").pil_image.resize((image_size, image_size))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "multi_modal_data": {"image": [image]}}


async def one_round(engine, prompts, sampling_params, request_offset: int):
    """Run ``prompts`` concurrently; return per-request (ttft, tpot, e2e, tokens)."""
    from vllm.inputs import TokensPrompt  # noqa: F401  (documents the API)

    async def stream(index: int, prompt: str):
        start = time.perf_counter()
        first_at = None
        count = 0
        async for out in engine.generate(
            prompt,
            sampling_params,
            request_id=f"req-{request_offset}-{index}",
        ):
            produced = len(out.outputs[0].token_ids)
            if first_at is None and produced >= 1:
                first_at = time.perf_counter()
            count = produced
        end = time.perf_counter()
        ttft = (first_at - start) * 1000.0
        # TPOT over the tokens after the first; undefined for a 1-token output.
        tpot = (
            (end - first_at) * 1000.0 / (count - 1) if count > 1 else float("nan")
        )
        return ttft, tpot, (end - start) * 1000.0, count

    return await asyncio.gather(
        *(stream(i, p) for i, p in enumerate(prompts))
    )


async def main_async(args) -> None:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1 if args.vision_bucket else 0, "video": 0},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": True},
                # See run.py: the runner's default breaks codegen for this model.
                "hlo2tensorizer_options": "",
            },
            # Supplying this at all is what selects the VL implementation, so a
            # zero bucket means "text-only" rather than "vision with no budget".
            **(
                {
                    "vision_neuron_config": {
                        "num_vision_tokens_buckets": [args.vision_bucket],
                        "vision_attention_block_size": args.vision_bucket,
                    }
                }
                if args.vision_bucket
                else {}
            ),
        },
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=args.output_tokens, min_tokens=args.output_tokens, temperature=0.0
    )

    mode = (
        f"vision, one {args.image_size}x{args.image_size} image"
        if args.vision_bucket
        else f"text-only, input_tokens={args.input_tokens}"
    )
    print(
        f"model={args.model}\n"
        f"tp={args.tensor_parallel_size} batch={args.max_num_seqs} {mode} "
        f"output_tokens={args.output_tokens} iterations={args.iterations}\n"
    )

    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []
    round_wall: list[float] = []
    total_out = 0

    for iteration in range(args.iterations + 1):
        if args.vision_bucket:
            # The same image every time: TTFT here is dominated by the tower, and
            # prefix caching is off so nothing is reused between rounds.
            prompts = [
                build_vl_prompt(args.model, args.image_size)
                for _ in range(args.max_num_seqs)
            ]
        else:
            prompts = [
                build_prompt(args.model, args.input_tokens, salt=iteration * 100 + i)
                for i in range(args.max_num_seqs)
            ]
        started = time.perf_counter()
        results = await one_round(engine, prompts, sampling_params, iteration)
        wall = time.perf_counter() - started
        if iteration == 0:
            print("(discarding the first round as warmup)\n")
            continue
        for ttft, tpot, e2e, count in results:
            ttfts.append(ttft)
            if count > 1:
                tpots.append(tpot)
            e2es.append(e2e)
            total_out += count
        round_wall.append(wall)

    def stat(name: str, values: list[float], unit: str = "ms") -> None:
        if not values:
            print(f"  {name:22s} n/a")
            return
        median = statistics.median(values)
        print(
            f"  {name:22s} mean {statistics.fmean(values):8.2f} {unit}   "
            f"median {median:8.2f} {unit}   "
            f"min {min(values):8.2f}   max {max(values):8.2f}"
        )

    print(f"results over {len(round_wall)} rounds x {args.max_num_seqs} requests")
    stat("TTFT", ttfts)
    stat("TPOT", tpots)
    stat("E2E", e2es)
    throughput = total_out / sum(round_wall)
    print(
        f"\n  output throughput     {throughput:8.2f} tok/s "
        f"(all {args.max_num_seqs} concurrent requests)"
    )
    if tpots:
        print(
            f"  per-stream decode     {1000.0 / statistics.fmean(tpots):8.2f} tok/s"
        )

    # AsyncLLM.shutdown() is synchronous; awaiting it raises.
    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        shutdown()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
