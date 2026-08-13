# SPDX-License-Identifier: Apache-2.0
"""On-device video smoke test for InternVL3-8B at TP=4.

Video needs no separate code path in this implementation: vLLM's video processor
asserts exactly one tile per frame, so a video arrives as one tile per frame in the
same layout images use, each frame costing the same 256 embed tokens. There is no
temporal merging and no timestamp text (unlike Qwen3-VL), so a frame is just an
independent single-tile image.

What this checks that the image tests cannot:
  1. the video kwargs reach embed_multimodal at all
     (``pixel_values_flat_video`` / ``video_num_patches``, not the image names)
  2. per-frame prompt expansion (``Frame1: <img>...``) lines up with the embeddings
  3. FRAME ORDER, via a video whose content changes over time

Point 3 is the one worth caring about. A shuffled frame order still produces fluent
text, so the video is built as a colour that changes red -> green -> blue over the
clip and the model is asked what changes. Getting "red then green then blue" back is
evidence the temporal order survived; "blue then red" or "no change" is not.

Usage:
    PATH=$V/bin:$PATH PYTHONPATH=/mnt/nvme/vllm-neuron \\
      python examples/vllm_neuron/models/internvl/video_smoke_test.py [--frames 16]
"""

import argparse
import os

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "3600")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
os.environ.setdefault("NEURON_CC_FLAGS", "--temp-dir=/tmp/neuroncc_tmp")
os.makedirs("/tmp/neuroncc_tmp", exist_ok=True)

MODEL = os.environ.get("INTERNVL_PATH", "/mnt/nvme/models/InternVL3-8B-Instruct")

# One tile per frame, 1024 raw patches and 256 embed tokens each.
PATCHES_PER_FRAME = 1024
EMBEDS_PER_FRAME = 256


def build_video(num_frames: int, size: int = 448):
    """A clip that goes red -> green -> blue, so frame order is checkable.

    Returns ``[frames, size, size, 3]`` uint8, which is what vLLM's video input
    accepts. Solid colours per frame keep the question unambiguous: any answer that
    names the three colours in the wrong order is an ordering bug, not the model
    being vague.
    """
    import numpy as np

    stops = [(200, 40, 40), (40, 180, 60), (40, 60, 200)]  # red, green, blue
    frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
    for i in range(num_frames):
        # Which third of the clip this frame falls in.
        frames[i, :, :, :] = stops[min(i * len(stops) // num_frames, len(stops) - 1)]
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frames",
        type=int,
        default=16,
        help="frames in the clip; 16 matches a 1-second 16fps clip, the scenario "
        "benchmarked for Qwen3-VL",
    )
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()

    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    video = build_video(args.frames)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": "<video>\nWhat colour is shown, and does it change "
                "during the video? Answer briefly.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    # One tile per frame, so the vision bucket is frames * 1024 raw patches. Note
    # this can exceed the image worst case (13 tiles = 13312): a 16-frame clip is
    # 16384.
    vision_bucket = args.frames * PATCHES_PER_FRAME
    # Per-frame prompt expansion adds "FrameN: " text around each frame's tokens,
    # so leave generous slack over frames * 256 rather than guessing it exactly.
    needed = args.frames * EMBEDS_PER_FRAME + 256 + args.max_tokens
    max_model_len = max(256, -(-needed // 256) * 256)

    print(
        f"frames={args.frames} vision_bucket={vision_bucket} "
        f"max_model_len={max_model_len}",
        flush=True,
    )

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        tensor_parallel_size=4,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"video": 1},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [max_model_len],
                "num_seqs_buckets": [1],
                "on_device_sampling_config": {"all_greedy": True},
            },
            "vision_neuron_config": {
                "num_vision_tokens_buckets": [vision_bucket],
                "vision_attention_block_size": PATCHES_PER_FRAME,
                "encoder_cache_num_blocks": args.frames * 2 + 8,
            },
        },
        disable_log_stats=True,
    )

    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"video": video}}],
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )
    text = outputs[0].outputs[0].text
    print("\n" + "=" * 70)
    print("GENERATED:", repr(text))
    print("=" * 70)
    print(
        "\nSanity: the clip runs red -> green -> blue.\n"
        "  Naming the colours in that order  -> frame order is preserved.\n"
        "  Wrong order, or 'no change'       -> ordering/embedding bug.\n"
        "  Garbage or repeated punctuation   -> numerical bug."
    )


if __name__ == "__main__":
    main()
