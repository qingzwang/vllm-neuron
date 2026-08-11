# SPDX-License-Identifier: Apache-2.0
"""Video latency benchmark for Qwen3-VL on Neuron: TTFT / TPOT / E2E per batch size.

Feeds a pre-sampled video (default 16 frames of 448x448, i.e. one second at
16 fps) plus a text question, decodes up to ``--max-tokens`` tokens, and reports
time-to-first-token, time-per-output-token and end-to-end latency for each
requested batch size. Batch size here means concurrent in-flight requests, all
submitted at once.

Latency is measured through ``AsyncLLM`` (offline, streaming) rather than an
HTTP server so that the frame count is exact: the video is handed to the engine
already sampled, with ``do_sample_frames=False``, which keeps the HF processor
from re-sampling it to its default fps. Numbers therefore exclude HTTP and
API-server overhead.

Usage:
    # Print the resolved config and vision-bucket math without loading weights
    python examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py \
        --model-checkpoint /path/to/Qwen_Qwen3-VL-8B-Instruct --dry-run

    # Benchmark one second of video (16 frames @ 448x448) at batch size 1
    python examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py \
        --model-checkpoint /path/to/Qwen_Qwen3-VL-8B-Instruct \
        --batch-sizes 1 --num-iters 5 --output-json results.json

    # Sweep batch sizes in a single engine (all num_seqs buckets compiled once)
    python examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py \
        --model-checkpoint /path/to/Qwen_Qwen3-VL-8B-Instruct \
        --batch-sizes 1,2,4,8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# Must be set before vLLM/Neuron import.
_ENV_DEFAULTS = {
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1200",
    "VLLM_NEURON_COMPILATION_TIMEOUT": "3600",
    # trn2.3xlarge has no EFA device; without this every worker dies looking
    # for /sys/bus/pci/devices/*/infiniband. CPU-affinity optimization only.
    "NEURON_SKIP_EFA_AFFINITY": "1",
    "NEURON_CC_FLAGS": "--temp-dir=/tmp/neuroncc_tmp",
}
for _k, _v in _ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)
os.makedirs("/tmp/neuroncc_tmp", exist_ok=True)

import numpy as np  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from transformers.video_utils import VideoMetadata  # noqa: E402

QUESTION = "Describe what happens in this video in detail."


# --------------------------------------------------------------------------- #
# Input construction
# --------------------------------------------------------------------------- #
def build_frames(num_frames: int, resolution: int, source: str) -> np.ndarray:
    """Return a ``[num_frames, resolution, resolution, 3]`` uint8 frame stack.

    ``source="asset"`` samples the vLLM baby_reading demo video so the
    generated text is a usable sanity check; ``source="synthetic"`` builds
    deterministic noise, which costs the same to encode and needs no network.
    """
    if source == "asset":
        import cv2

        from vllm.assets.video import VideoAsset

        raw = VideoAsset("baby_reading", num_frames=num_frames).np_ndarrays
        return np.stack(
            [
                cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_AREA)
                for f in raw
            ]
        )

    rng = np.random.default_rng(0)
    return rng.integers(
        0, 256, (num_frames, resolution, resolution, 3), dtype=np.uint8
    )


def build_metadata(num_frames: int, fps: float, resolution: int) -> dict[str, Any]:
    """Video metadata for an already-sampled clip.

    ``do_sample_frames=False`` tells the HF processor to consume the frames
    as-is. vLLM's Qwen3-VL processor pops this key before building
    ``VideoMetadata`` (which rejects it as a constructor kwarg) and forwards it
    as a processor kwarg.
    """
    return {
        "fps": float(fps),
        "duration": num_frames / fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "width": resolution,
        "height": resolution,
        "do_sample_frames": False,
    }


def probe_shapes(
    model: str, frames: np.ndarray, metadata: dict[str, Any], question: str
) -> tuple[str, list[int], int]:
    """Run the HF processor once on CPU to get the prompt and real grid shape.

    Deriving the vision bucket from the processor's own ``video_grid_thw``
    instead of re-implementing the patch geometry keeps the bucket correct for
    any fps/resolution the caller passes.

    Returns:
        (prompt, grid_thw, num_prompt_tokens)
    """
    processor = AutoProcessor.from_pretrained(model)
    messages = [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": question}],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    meta = {k: v for k, v in metadata.items() if k != "do_sample_frames"}
    out = processor(
        text=[prompt],
        videos=[frames],
        video_metadata=[VideoMetadata(**meta)],
        do_sample_frames=False,
        return_tensors="pt",
    )
    grid = out["video_grid_thw"][0].tolist()
    return prompt, grid, int(out["input_ids"].shape[1])


class PromptFactory:
    """Builds the per-request prompt payload.

    By default every request gets a byte-unique copy of the frame stack. This
    matters for correctness of the measurement, not realism of the pixels: the
    scheduler's ``EncoderCacheManager`` keys vision embeddings on the
    multimodal hash and keeps entries alive after a request finishes, so
    resending an identical video makes later requests reuse the earlier
    embeddings and skip the vision encoder entirely — TTFT then excludes vision
    encode and video preprocessing. Production traffic never repeats a video, so
    unique frames are the honest default; ``reuse=True`` measures the
    cache-hit path for comparison.
    """

    def __init__(
        self,
        prompt_text: str,
        frames: np.ndarray,
        metadata: dict[str, Any],
        reuse: bool,
    ):
        self.prompt_text = prompt_text
        self.frames = frames
        self.metadata = metadata
        self.reuse = reuse
        self._counter = 0

    def __call__(self) -> dict[str, Any]:
        if self.reuse:
            frames = self.frames
        else:
            frames = self.frames.copy()
            # Three bytes of one corner pixel = 16.7M unique videos, enough to
            # keep every request a cache miss without touching the shapes.
            self._counter += 1
            tag = self._counter
            frames[0, 0, 0, 0] = tag & 0xFF
            frames[0, 0, 0, 1] = (tag >> 8) & 0xFF
            frames[0, 0, 0, 2] = (tag >> 16) & 0xFF
        return {
            "prompt": self.prompt_text,
            "multi_modal_data": {"video": (frames, self.metadata)},
        }


def vision_bucket_for(grid_thw: list[int], block_size: int) -> tuple[int, int]:
    """Smallest vision bucket that holds one video's block-padded slices.

    The encoder packs whole temporal slices into ``block_size`` blocks and never
    splits a slice across blocks, so the bucket must cover the *padded* block
    span, not the raw patch count. A [T,H,W] video contributes T slices of H*W
    patches each.

    Returns:
        (bucket, num_blocks)
    """
    t, h, w = grid_thw
    patches_per_slice = h * w
    if patches_per_slice > block_size:
        raise ValueError(
            f"One temporal slice is {patches_per_slice} patches, which exceeds "
            f"vision_attention_block_size={block_size}. A slice cannot be split "
            f"across blocks; raise --vision-block-size."
        )
    slices_per_block = block_size // patches_per_slice
    num_blocks = math.ceil(t / slices_per_block)
    return num_blocks * block_size, num_blocks


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
@dataclass
class RequestResult:
    """Timings for a single request, all seconds relative to submit time."""

    batch_size: int
    iteration: int
    ttft: float
    e2e: float
    output_tokens: int
    # (elapsed_at_chunk, tokens_in_chunk) for every non-empty streamed chunk.
    chunks: list[tuple[float, int]] = field(default_factory=list)
    text: str = ""

    @property
    def tpot(self) -> float | None:
        """Mean decode-step latency, excluding the prefill step."""
        if self.output_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)

    def itls(self) -> list[float]:
        """Per-token inter-token latencies after the first token.

        A chunk carrying k tokens is charged k evenly-split intervals, since
        the engine can hand back more than one token per streamed chunk.
        """
        out: list[float] = []
        prev = self.ttft
        for i, (at, n) in enumerate(self.chunks):
            if i == 0:
                prev = at
                if n > 1:
                    out.extend([0.0] * (n - 1))
                continue
            out.extend([(at - prev) / n] * n)
            prev = at
        return out


async def run_request(
    engine, prompt: dict[str, Any], sampling_params, request_id: str
) -> tuple[float, float, int, list[tuple[float, int]], str]:
    start = time.perf_counter()
    ttft: float | None = None
    chunks: list[tuple[float, int]] = []
    total = 0
    text: list[str] = []

    async for out in engine.generate(prompt, sampling_params, request_id):
        now = time.perf_counter() - start
        completion = out.outputs[0]
        n = len(completion.token_ids)
        if n:
            if ttft is None:
                ttft = now
            chunks.append((now, n))
            total += n
            text.append(completion.text)
        if out.finished:
            break

    e2e = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError(f"request {request_id} produced no tokens")
    return ttft, e2e, total, chunks, "".join(text)


async def run_batch(
    engine,
    make_prompt,
    sampling_params,
    batch_size: int,
    iteration: int,
) -> tuple[list[RequestResult], float]:
    """Submit ``batch_size`` requests at once and wait for all of them."""
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(
            run_request(
                engine,
                make_prompt(),
                sampling_params,
                f"bs{batch_size}-it{iteration}-r{i}",
            )
        )
        for i in range(batch_size)
    ]
    done = await asyncio.gather(*tasks)
    wall = time.perf_counter() - start

    return [
        RequestResult(
            batch_size=batch_size,
            iteration=iteration,
            ttft=ttft,
            e2e=e2e,
            output_tokens=ntok,
            chunks=chunks,
            text=text,
        )
        for ttft, e2e, ntok, chunks, text in done
    ], wall


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q / 100 * len(ordered))) - 1))
    return ordered[idx]


def summarize(
    batch_size: int, results: list[RequestResult], walls: list[float]
) -> dict[str, Any]:
    ttfts = [r.ttft * 1e3 for r in results]
    e2es = [r.e2e * 1e3 for r in results]
    tpots = [r.tpot * 1e3 for r in results if r.tpot is not None]
    itls = [v * 1e3 for r in results for v in r.itls()]
    tokens = [r.output_tokens for r in results]

    def stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {}
        return {
            "mean": statistics.fmean(vals),
            "p50": statistics.median(vals),
            "p90": _pct(vals, 90),
            "min": min(vals),
            "max": max(vals),
        }

    total_out = sum(tokens)
    return {
        "batch_size": batch_size,
        "num_requests": len(results),
        "output_tokens_per_request": {
            "mean": statistics.fmean(tokens),
            "min": min(tokens),
            "max": max(tokens),
        },
        "ttft_ms": stats(ttfts),
        "tpot_ms": stats(tpots),
        "itl_ms": stats(itls),
        "e2e_ms": stats(e2es),
        "batch_wall_s": stats(walls),
        # Aggregate over whole batches: what the box actually sustains.
        "output_throughput_tok_s": total_out / sum(walls) if sum(walls) else 0.0,
        "request_throughput_req_s": len(results) / sum(walls) if sum(walls) else 0.0,
    }


def print_table(summaries: list[dict[str, Any]], reuse_video: bool) -> None:
    header = (
        f"{'BS':>3} {'reqs':>5} {'out_tok':>8} "
        f"{'TTFT p50':>9} {'TTFT mean':>10} {'TTFT p90':>9} {'TTFT max':>9} "
        f"{'TPOT p50':>9} {'TPOT p90':>9} "
        f"{'E2E p50':>9} {'E2E p90':>9} "
        f"{'tok/s':>8} {'req/s':>7}"
    )
    print("\n" + "=" * len(header))
    print("Qwen3-VL video latency (ms unless noted)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['batch_size']:>3} {s['num_requests']:>5} "
            f"{s['output_tokens_per_request']['mean']:>8.1f} "
            f"{s['ttft_ms']['p50']:>9.1f} {s['ttft_ms']['mean']:>10.1f} "
            f"{s['ttft_ms']['p90']:>9.1f} {s['ttft_ms']['max']:>9.1f} "
            f"{s['tpot_ms'].get('p50', float('nan')):>9.2f} "
            f"{s['tpot_ms'].get('p90', float('nan')):>9.2f} "
            f"{s['e2e_ms']['p50']:>9.1f} {s['e2e_ms']['p90']:>9.1f} "
            f"{s['output_throughput_tok_s']:>8.1f} "
            f"{s['request_throughput_req_s']:>7.2f}"
        )
    print("=" * len(header))
    if reuse_video:
        print(
            "TTFT covers prefill only: --reuse-video let the encoder cache serve "
            "the vision embeddings, so vision encode and video preprocessing are "
            "NOT included."
        )
    else:
        print(
            "TTFT covers video preprocessing + vision encode + prefill "
            "(unique video per request, encoder cache misses every time)."
        )
    print(
        "TPOT = (E2E - TTFT) / (out_tok - 1); tok/s and req/s are aggregate over "
        "concurrent batches."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Qwen3-VL video TTFT/TPOT/E2E on Neuron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-checkpoint",
        default="/mnt/nvme/models/Qwen_Qwen3-VL-8B-Instruct",
        help="Path to (or HF id of) the Qwen3-VL checkpoint",
    )
    p.add_argument("--fps", type=float, default=16.0, help="Frames per second")
    p.add_argument(
        "--duration-sec", type=float, default=1.0, help="Video length in seconds"
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frame count override (default: round(fps * duration_sec))",
    )
    p.add_argument(
        "--resolution", type=int, default=448, help="Square frame edge in pixels"
    )
    p.add_argument(
        "--video-source",
        choices=("asset", "synthetic"),
        default="asset",
        help="Real demo video (readable output) or deterministic noise (no network)",
    )
    p.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    p.add_argument(
        "--no-ignore-eos",
        dest="ignore_eos",
        action="store_false",
        help="Stop at EOS. Off by default so every request decodes the same "
        "token count, which keeps TPOT comparable across batch sizes.",
    )
    p.add_argument(
        "--batch-sizes",
        default="1",
        help="Comma-separated concurrent-request counts to sweep",
    )
    p.add_argument(
        "--reuse-video",
        action="store_true",
        help="Send byte-identical frames every request. Lets the scheduler's "
        "encoder cache serve the vision embeddings from an earlier request, so "
        "TTFT drops the vision encode and preprocessing. Off by default.",
    )
    p.add_argument("--num-iters", type=int, default=5, help="Measured iterations")
    p.add_argument(
        "--warmup-iters", type=int, default=1, help="Unmeasured iterations per batch"
    )
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument(
        "--vision-block-size",
        type=int,
        default=2048,
        help="vision_attention_block_size; the bucket is derived from it",
    )
    p.add_argument(
        "--vision-bucket",
        type=int,
        default=None,
        help="Override the derived num_vision_tokens_buckets entry",
    )
    p.add_argument(
        "--encoder-cache-num-blocks",
        type=int,
        default=None,
        help="Override the on-device encoder cache block count. The plugin "
        "auto-derives this from the scheduler's token budget, which is blind to "
        "per-item block padding, so a stream of distinct videos can exhaust the "
        "block allocator and kill the engine with 'Encoder cache full'. Set it "
        "to ceil(encoder_cache_size / embed_tokens_per_video) * blocks_per_video "
        "+ 1 (see the printed cache-headroom line).",
    )
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2048,
        help="Must be >= prompt length for a single-pass prefill",
    )
    p.add_argument("--output-json", default=None, help="Write full results here")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved shapes and config, then exit",
    )
    return p.parse_args()


async def benchmark(args: argparse.Namespace) -> int:
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    if not batch_sizes:
        raise ValueError("--batch-sizes is empty")
    num_frames = args.num_frames or round(args.fps * args.duration_sec)
    if num_frames % 2:
        raise ValueError(
            f"num_frames must be even (temporal patch size 2), got {num_frames}"
        )

    frames = build_frames(num_frames, args.resolution, args.video_source)
    metadata = build_metadata(num_frames, args.fps, args.resolution)
    prompt_text, grid_thw, num_prompt_tokens = probe_shapes(
        args.model_checkpoint, frames, metadata, QUESTION
    )

    derived_bucket, num_blocks = vision_bucket_for(grid_thw, args.vision_block_size)
    vision_bucket = args.vision_bucket or derived_bucket
    raw_patches = grid_thw[0] * grid_thw[1] * grid_thw[2]

    print("--- input ---")
    print(f"frames            : {num_frames} @ {args.resolution}x{args.resolution}")
    print(f"fps / duration    : {args.fps} / {num_frames / args.fps:.2f}s")
    print(f"video_grid_thw    : {grid_thw}")
    print(f"vision patches    : {raw_patches} raw -> {raw_patches // 4} embed tokens")
    print(f"prompt tokens     : {num_prompt_tokens}")
    print(
        f"vision bucket     : {vision_bucket} "
        f"({num_blocks} x {args.vision_block_size})"
    )
    print(f"batch sizes       : {batch_sizes}")
    print(f"max output tokens : {args.max_tokens} (ignore_eos={args.ignore_eos})")
    video_mode = (
        "reused (encoder cache hits, vision encode excluded)"
        if args.reuse_video
        else "unique (encoder cache miss every request)"
    )
    print(f"per-request video : {video_mode}")

    if num_prompt_tokens > args.max_num_batched_tokens:
        print(
            f"WARNING: prompt ({num_prompt_tokens}) exceeds "
            f"--max-num-batched-tokens ({args.max_num_batched_tokens}); prefill "
            f"will be segmented across passes and TTFT will rise."
        )
    if num_prompt_tokens + args.max_tokens > args.max_model_len:
        raise ValueError(
            f"prompt ({num_prompt_tokens}) + max_tokens ({args.max_tokens}) "
            f"exceeds --max-model-len ({args.max_model_len})"
        )

    additional_config = {
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [args.max_num_batched_tokens],
            "num_seqs_buckets": sorted(set(batch_sizes)),
            "on_device_sampling_config": {"all_greedy": True},
        },
        "vision_neuron_config": {
            "num_vision_tokens_buckets": [vision_bucket],
            "vision_attention_block_size": args.vision_block_size,
        },
    }
    if args.encoder_cache_num_blocks is not None:
        additional_config["vision_neuron_config"]["encoder_cache_num_blocks"] = (
            args.encoder_cache_num_blocks
        )

    # The on-device encoder cache allocates whole blocks per item while the
    # scheduler admits items against a token budget, so the allocator needs
    # headroom proportional to the padding waste or it runs dry mid-stream.
    embed_per_video = raw_patches // 4
    cache_block_size = args.vision_block_size // 4
    blocks_per_video = math.ceil(embed_per_video / cache_block_size)
    print(
        f"cache headroom    : video = {embed_per_video} embeds -> "
        f"{blocks_per_video} x {cache_block_size} = "
        f"{blocks_per_video * cache_block_size} slots "
        f"({blocks_per_video * cache_block_size / embed_per_video:.2f}x padding). "
        f"Needs blocks_per_video * (scheduler encoder_cache_size / "
        f"{embed_per_video}) + 1; encoder_cache_num_blocks="
        f"{args.encoder_cache_num_blocks or 'auto'}"
    )
    print("--- config ---")
    print(json.dumps(additional_config, indent=2))

    if args.dry_run:
        print("\n--dry-run: not loading the model.")
        return 0

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(batch_sizes),
        tensor_parallel_size=args.tensor_parallel_size,
        # Prefix caching hard-requires segmented prefill on Neuron and would
        # also make repeated identical prompts hit cache, hiding real prefill cost.
        enable_prefix_caching=False,
        limit_mm_per_prompt={"video": 1},
        additional_config=additional_config,
        disable_log_stats=True,
    )

    load_start = time.perf_counter()
    engine = AsyncLLM.from_engine_args(engine_args)
    print(f"\nengine ready in {time.perf_counter() - load_start:.1f}s")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=args.ignore_eos,
        output_kind=RequestOutputKind.DELTA,
    )
    make_prompt = PromptFactory(prompt_text, frames, metadata, args.reuse_video)

    all_results: list[RequestResult] = []
    summaries: list[dict[str, Any]] = []
    sample_text = ""

    try:
        for batch_size in batch_sizes:
            for it in range(args.warmup_iters):
                warm, _ = await run_batch(
                    engine, make_prompt, sampling_params, batch_size, -1 - it
                )
                if not sample_text and warm:
                    sample_text = warm[0].text
                print(
                    f"[bs={batch_size}] warmup {it + 1}/{args.warmup_iters}: "
                    f"ttft={warm[0].ttft * 1e3:.0f}ms e2e={warm[0].e2e * 1e3:.0f}ms "
                    f"tok={warm[0].output_tokens}"
                )

            results: list[RequestResult] = []
            walls: list[float] = []
            for it in range(args.num_iters):
                batch, wall = await run_batch(
                    engine, make_prompt, sampling_params, batch_size, it
                )
                results.extend(batch)
                walls.append(wall)
                print(
                    f"[bs={batch_size}] iter {it + 1}/{args.num_iters}: "
                    f"wall={wall * 1e3:.0f}ms "
                    f"ttft={statistics.fmean(r.ttft for r in batch) * 1e3:.0f}ms "
                    f"e2e={statistics.fmean(r.e2e for r in batch) * 1e3:.0f}ms"
                )

            all_results.extend(results)
            summaries.append(summarize(batch_size, results, walls))
    finally:
        engine.shutdown()

    if sample_text:
        print(f"\n--- sample output (first 400 chars) ---\n{sample_text[:400]}")

    print_table(summaries, args.reuse_video)

    if args.output_json:
        payload = {
            "input": {
                "model": args.model_checkpoint,
                "num_frames": num_frames,
                "fps": args.fps,
                "resolution": args.resolution,
                "video_source": args.video_source,
                "video_grid_thw": grid_thw,
                "raw_vision_patches": raw_patches,
                "prompt_tokens": num_prompt_tokens,
                "max_tokens": args.max_tokens,
                "ignore_eos": args.ignore_eos,
                "reuse_video": args.reuse_video,
                "tensor_parallel_size": args.tensor_parallel_size,
                "vision_bucket": vision_bucket,
                "vision_block_size": args.vision_block_size,
                "num_vision_blocks": num_blocks,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "num_iters": args.num_iters,
                "warmup_iters": args.warmup_iters,
            },
            "summaries": summaries,
            "requests": [
                {k: v for k, v in asdict(r).items() if k != "text"}
                for r in all_results
            ],
        }
        parent = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(parent, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.output_json}")

    return 0


def main() -> int:
    return asyncio.run(benchmark(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
