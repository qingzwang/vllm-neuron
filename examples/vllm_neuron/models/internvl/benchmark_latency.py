# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for InternVL3-8B on Neuron: TTFT, TPOT, end-to-end.

Runs ONE configuration per invocation (one batch size, one tile count) and appends
a JSON record to --output. Loop it from the shell to build a sweep:

    for BS in 1 2 4 8; do
      python examples/vllm_neuron/models/internvl/benchmark_latency.py \
        --batch-size $BS --tiles 7 --output /tmp/internvl_latency.json
    done

One engine per process on purpose. Passing several num_seqs_buckets to a single
engine hangs on the first request (recorded on the Qwen3-VL branch), and tearing
an engine down to build another in the same process is not something this plugin
is known to survive.

Why AsyncLLM and not LLM.generate: TTFT needs the timestamp of the *first* token,
which only the streaming API exposes. LLM.generate returns once everything is done.

Two measurement traps this avoids, both of which produced wrong numbers before:

- **Identical images read as cache hits.** EncoderCacheManager keys on mm_hash and
  keeps entries past request completion, so sending the same image twice skips the
  vision encoder entirely and understates TTFT (by 42% on Qwen3-VL). Every request
  here gets byte-unique pixels. --reuse-image measures the cached path deliberately.
- **Oversized max_model_len.** With decode_context_length_buckets unset, decode
  attends over the whole window per sequence; a 4096 window for a workload needing
  ~2100 cost 3.1x TPOT on Qwen3-VL. max_model_len is sized to the workload here.

Read the generated text before trusting any number: a wrong TP shard produces
plausible timings with garbage output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "3600")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
os.environ.setdefault("NEURON_CC_FLAGS", "--temp-dir=/tmp/neuroncc_tmp")
os.makedirs("/tmp/neuroncc_tmp", exist_ok=True)

DEFAULT_MODEL = "/mnt/nvme/models/InternVL3-8B-Instruct"
QUESTION = "Describe this image in detail."

# InternVL tiles are always image_size x image_size, and pixel shuffle collapses
# each 2x2 patch neighbourhood, so one tile is a fixed 256 embed tokens.
PATCHES_PER_TILE = 1024
EMBEDS_PER_TILE = 256


def build_image(tiles: int, image_size: int):
    """Return a PIL image that InternVL's own tiler splits into ``tiles`` tiles.

    Asks the processor's geometry rather than reimplementing it: pick the aspect
    ratio whose tile count matches, then confirm by running the real tiler. A
    hand-derived size is exactly the kind of thing that silently drifts from the
    processor and makes the reported tile count a lie.
    """
    from PIL import Image
    from vllm.transformers_utils.processors.internvl import (
        dynamic_preprocess_internvl,
        get_internvl_target_ratios,
    )

    # A thumbnail is appended whenever the image splits into more than one block,
    # so `tiles` blocks of image content means tiles-1 blocks plus the thumbnail.
    #
    # Several aspect ratios give the same block count (6 blocks is 3x2 or 6x1);
    # pick the squarest, so the image resembles a photo rather than a strip. Device
    # cost depends only on the tile count, not on the ratio, so this does not move
    # the numbers -- it only stops the reported image size from being misleading.
    blocks = 1 if tiles == 1 else tiles - 1
    ratios = get_internvl_target_ratios(1, 12)
    candidates = [r for r in sorted(ratios) if r[0] * r[1] == blocks]
    ratio = min(candidates, key=lambda r: abs(r[0] / r[1] - 1.0), default=None)
    if ratio is None:
        raise SystemExit(
            f"--tiles {tiles} is not reachable: no aspect ratio gives {blocks} "
            f"blocks. Reachable tile counts are 1 and blocks+1 for blocks in "
            f"{sorted({w * h for w, h in ratios})}."
        )

    wr, hr = ratio
    image = Image.new("RGB", (image_size * wr, image_size * hr), (90, 140, 200))
    actual = len(
        dynamic_preprocess_internvl(
            image, target_ratios=ratios, image_size=image_size, use_thumbnail=True
        )
    )
    if actual != tiles:
        raise SystemExit(
            f"asked for {tiles} tiles, the processor produced {actual} for "
            f"{image.size}; the aspect-ratio table must have changed."
        )
    return image


def tag_image(image, tag: int):
    """Make an image byte-unique so its mm_hash misses the encoder cache.

    Three pixels is enough: mm_hash covers the whole item, so touching tile 0
    forces every tile to be re-encoded.
    """
    px = image.copy()
    px.putpixel((0, 0), (tag & 0xFF, (tag >> 8) & 0xFF, (tag >> 16) & 0xFF))
    return px


async def run(args) -> dict:
    from transformers import AutoTokenizer

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    image = build_image(args.tiles, args.image_size)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": f"<image>\n{QUESTION}"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Size max_model_len to the workload; see the module docstring. The <image>
    # placeholder expands to EMBEDS_PER_TILE tokens per tile.
    text_tokens = len(tok(prompt).input_ids)
    est_prompt_tokens = text_tokens + args.tiles * EMBEDS_PER_TILE
    needed = est_prompt_tokens + args.max_tokens
    # Round up to a multiple of 256, not to a power of two: a 7-tile prompt needs
    # ~2100 tokens, and the next power of two (4096) is nearly double that. Decode
    # attends over the whole window, so that slack is paid on every token.
    max_model_len = args.max_model_len or max(256, -(-needed // 256) * 256)
    max_num_batched_tokens = min(max_model_len, args.max_num_batched_tokens)

    vision_bucket = args.tiles * PATCHES_PER_TILE
    # One block per item per tile, plus headroom: the scheduler admits items on a
    # token budget while the allocator works in blocks, so a tight number here
    # crashes mid-stream with "Encoder cache full" once several distinct images
    # are in flight.
    encoder_cache_num_blocks = args.encoder_cache_num_blocks or (
        args.tiles * args.batch_size * 2 + 8
    )

    print(
        f"config: tiles={args.tiles} image={image.size} batch={args.batch_size}\n"
        f"        est_prompt_tokens={est_prompt_tokens} (text {text_tokens})\n"
        f"        max_model_len={max_model_len} vision_bucket={vision_bucket}\n"
        f"        encoder_cache_num_blocks={encoder_cache_num_blocks}",
        flush=True,
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [max_num_batched_tokens],
                "num_seqs_buckets": [args.batch_size],
                "on_device_sampling_config": {"all_greedy": True},
            },
            "vision_neuron_config": {
                "num_vision_tokens_buckets": [vision_bucket],
                "vision_attention_block_size": PATCHES_PER_TILE,
                "encoder_cache_num_blocks": encoder_cache_num_blocks,
            },
        },
    )

    t_engine = time.perf_counter()
    llm = AsyncLLM.from_engine_args(engine_args)
    engine_init_s = time.perf_counter() - t_engine

    # ignore_eos so every request emits exactly max_tokens: TPOT over a variable
    # token count is not comparable across runs.
    sampling = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True
    )

    tag = 0
    sample_text = ""

    async def one(req_id: str, tag: int) -> dict:
        nonlocal sample_text
        payload = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image if args.reuse_image else tag_image(image, tag)
            },
        }
        start = time.perf_counter()
        ttft = None
        n_tokens = 0
        text = ""
        async for out in llm.generate(payload, sampling, req_id):
            comp = out.outputs[0]
            if ttft is None and len(comp.token_ids) > 0:
                ttft = time.perf_counter() - start
            n_tokens = len(comp.token_ids)
            text = comp.text
            prompt_tokens = len(out.prompt_token_ids or ())
        e2e = time.perf_counter() - start
        sample_text = text
        # TPOT excludes the first token, which is produced by prefill.
        tpot = (e2e - ttft) / (n_tokens - 1) if n_tokens > 1 else float("nan")
        return {
            "ttft_ms": ttft * 1000,
            "tpot_ms": tpot * 1000,
            "e2e_ms": e2e * 1000,
            "out_tokens": n_tokens,
            "prompt_tokens": prompt_tokens,
        }

    async def one_round(round_idx: int) -> list[dict]:
        nonlocal tag
        tasks = []
        for slot in range(args.batch_size):
            tag += 1
            tasks.append(one(f"r{round_idx}-{slot}", tag))
        return await asyncio.gather(*tasks)

    print(f"warmup ({args.warmup} round(s))...", flush=True)
    for w in range(args.warmup):
        await one_round(-1 - w)

    per_request: list[dict] = []
    round_wall_ms: list[float] = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        rows = await one_round(i)
        round_wall_ms.append((time.perf_counter() - t0) * 1000)
        per_request.extend(rows)
        print(
            f"  round {i + 1}/{args.iters}: "
            f"ttft {statistics.mean(r['ttft_ms'] for r in rows):.1f} ms  "
            f"tpot {statistics.mean(r['tpot_ms'] for r in rows):.2f} ms  "
            f"e2e {statistics.mean(r['e2e_ms'] for r in rows):.1f} ms",
            flush=True,
        )

    def agg(key: str) -> dict:
        vals = sorted(r[key] for r in per_request)
        return {
            "mean": statistics.mean(vals),
            "p50": statistics.median(vals),
            "p99": vals[min(len(vals) - 1, int(0.99 * len(vals)))],
            "min": vals[0],
            "max": vals[-1],
        }

    record = {
        "model": args.model,
        "tiles": args.tiles,
        "image_size": list(image.size),
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_tokens": args.max_tokens,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "vision_bucket": vision_bucket,
        "encoder_cache_num_blocks": encoder_cache_num_blocks,
        "reuse_image": args.reuse_image,
        "engine_init_s": engine_init_s,
        "iters": args.iters,
        "num_requests": len(per_request),
        "prompt_tokens": per_request[0]["prompt_tokens"],
        "ttft_ms": agg("ttft_ms"),
        "tpot_ms": agg("tpot_ms"),
        "e2e_ms": agg("e2e_ms"),
        # Wall time for a whole batch: what a "one inference per second" target is
        # actually measured against, and not the same as mean per-request e2e.
        "round_wall_ms": {
            "mean": statistics.mean(round_wall_ms),
            "min": min(round_wall_ms),
            "max": max(round_wall_ms),
        },
        "throughput_req_per_s": args.batch_size
        / (statistics.mean(round_wall_ms) / 1000),
        "sample_text": sample_text,
    }

    print("\n" + "=" * 70)
    print(f"tiles={args.tiles} batch={args.batch_size}")
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        a = record[key]
        print(
            f"  {key:<9} mean {a['mean']:8.2f}  p50 {a['p50']:8.2f}  "
            f"p99 {a['p99']:8.2f}"
        )
    print(f"  batch wall  {record['round_wall_ms']['mean']:.1f} ms")
    print(f"  throughput  {record['throughput_req_per_s']:.2f} req/s")
    print(f"  text        {sample_text[:120]!r}")
    print("=" * 70, flush=True)

    llm.shutdown()
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("INTERNVL_PATH", DEFAULT_MODEL))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--tiles",
        type=int,
        default=7,
        help="tiles per image including the thumbnail: 7 means a 3x2 split plus "
        "the thumbnail, which is what a typical landscape photo produces. 1 is "
        "the single-tile floor, 13 the maximum (max_dynamic_patch=12 + thumbnail). "
        "Not every count is reachable -- it must be 1 or blocks+1 for some "
        "achievable block count.",
    )
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="default: smallest power of two that fits prompt+output. Oversizing "
        "this is the single biggest TPOT regression on this plugin.",
    )
    p.add_argument("--encoder-cache-num-blocks", type=int, default=None)
    p.add_argument(
        "--reuse-image",
        action="store_true",
        help="send the identical image every time, so the encoder cache hits and "
        "TTFT excludes the vision encoder. Measures the cached path, not a "
        "cold one.",
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    record = asyncio.run(run(args))

    if args.output:
        existing = []
        if args.output.exists():
            existing = json.loads(args.output.read_text())
        existing.append(record)
        args.output.write_text(json.dumps(existing, indent=2))
        print(f"appended to {args.output} ({len(existing)} record(s))")


if __name__ == "__main__":
    main()
