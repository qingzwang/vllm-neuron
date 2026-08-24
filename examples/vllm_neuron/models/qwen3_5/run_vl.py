#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Vision-language inference for Qwen3.5-2B on Neuron.

The vision tower is the plugin's Qwen3-VL encoder reused unchanged — HF's
``Qwen3_5VisionModel`` is ``Qwen3VLVisionModel`` minus the deepstack mergers, and
the checkpoint's ``model.visual.*`` names are identical. See
``vllm_neuron/model/qwen3_5/vl.py``.

Passing a ``vision_neuron_config`` is what selects the VL implementation; without
it the factory builds the text-only model and never pays for the tower. So this
script and ``run.py`` exercise two different classes on purpose.

Sizing, which has two independent knobs:

* ``vision_attention_block_size`` is the **per-block** size, and one block holds
  exactly one item (the packer uses ``one_item_per_block=True``), so it must be at
  least a single image's raw (pre-merge) token count ``T*H*W``. Note the *processor*
  decides that count, not ``--image-size``: it re-resizes to satisfy its own pixel
  bounds and the merge multiple. Measured for ``cherry_blossom``:

      fed size            grid        raw    merged   needs block >=
      224x224             1,16,16     256    64       256
      336x336             1,20,20     400    100      512
      448x448             1,28,28     784    196      1024
      672x672             1,42,42     1764   441      2048
      native 1770x1180    1,74,110    8140   2035     8192

  Bigger blocks cost compile time and wasted compute, so size to the workload.
* ``num_vision_tokens_buckets`` is the **total** budget across items, and it caps
  blocks per request at ``bucket / merge_factor / cache_block_size``. Two images
  therefore need ``bucket >= 2 * block_size``.

Setting the two equal works for one image and fails for two with "Vision block
truncation: request has more cached blocks than max_vision_blocks_per_request=1".

A **video** is packed per temporal group (``temporal_patch_size`` frames each), not
per frame, and ``mm_processor_kwargs["max_pixels"]`` does not reach the video
processor — so the block has to be sized for the video's native grid. Measured:
``baby_reading`` is 640x360 per frame, so 4 frames pair into 2 temporal groups of
880 raw tokens each (a 40x22 grid) = 440 merged total, and it needs
``--vision-block-size 1024 --vision-bucket 2048``. 256/1024 fails with "produces
440 embedding tokens, which exceeds the maximum supported by the compiled vision
encoder (256)".

Usage:

    NEURON_SKIP_EFA_AFFINITY=1 VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm \
    NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp" \
    PYTHONPATH=/mnt/nvme/vllm-neuron PATH=$V/bin:$PATH $V/bin/python \
      run_vl.py [--image-size 224] [--vision-bucket 256]
"""

import argparse
import os

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1800")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "2400")

QUESTION = "Describe this image in detail."
MULTI_QUESTION = "Compare these two images. What is different about them?"
VIDEO_QUESTION = "Describe what happens in this video."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="square resize; must be a multiple of patch_size * spatial_merge_size",
    )
    parser.add_argument(
        "--vision-block-size",
        type=int,
        default=256,
        help="per-block size; must be >= one image's raw token count",
    )
    parser.add_argument(
        "--vision-bucket",
        type=int,
        default=0,
        help="total vision token budget; defaults to num_images * block size",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="one block is allocated per image, so the bucket scales with this",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=0,
        help="non-zero switches to a video prompt. A video is packed per FRAME, so "
        "the block size must hold one frame and the bucket must hold every frame.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="per-image pixel cap passed to the HF processor; raw tokens per item "
        "are max_pixels / patch_size^2. NOTE this does *not* reach the video "
        "processor — measured: baby_reading still produced 440 embedding tokens "
        "with max_pixels=65536 set. Size the block/bucket for the video's native "
        "grid instead.",
    )
    parser.add_argument(
        "--vision-tp",
        type=int,
        default=0,
        help="vision encoder TP degree; 0 keeps the plugin default (tp=1, i.e. an "
        "unsharded encoder). 4 shards it and roughly halves vision latency",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def build_video_input(model_path: str, num_frames: int):
    """A single-video chat prompt.

    vLLM's video parser wants ``(frames, metadata)`` together — the metadata drives
    the per-frame timestamp tokens, and the resulting placeholder span is
    non-contiguous, which exercises the is_embed-aware position mapping.
    """
    from transformers import AutoProcessor
    from vllm.assets.video import VideoAsset

    processor = AutoProcessor.from_pretrained(model_path)
    asset = VideoAsset("baby_reading", num_frames=num_frames)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": VIDEO_QUESTION},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return VIDEO_QUESTION, {
        "prompt": prompt,
        "multi_modal_data": {"video": (asset.np_ndarrays, asset.metadata)},
    }


def build_input(model_path: str, image_size: int, num_images: int):
    """A chat prompt with ``num_images`` placeholders, from the model's processor."""
    from transformers import AutoProcessor
    from vllm.assets.image import ImageAsset

    processor = AutoProcessor.from_pretrained(model_path)
    assets = ["cherry_blossom", "stop_sign"][:num_images]
    images = [
        ImageAsset(name).pil_image.resize((image_size, image_size)) for name in assets
    ]
    question = QUESTION if num_images == 1 else MULTI_QUESTION
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}] * num_images
            + [{"type": "text", "text": question}],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return question, {"prompt": prompt, "multi_modal_data": {"image": images}}


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    is_video = args.video_frames > 0
    # One block per item, and a video's items are its frames.
    items = args.video_frames if is_video else args.num_images
    bucket = args.vision_bucket or items * args.vision_block_size
    print(
        f"vision: {items} {'frame' if is_video else 'image'}(s), block_size="
        f"{args.vision_block_size}, bucket={bucket}"
        + (f", max_pixels={args.max_pixels}" if args.max_pixels else "")
    )

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        **(
            {"mm_processor_kwargs": {"max_pixels": args.max_pixels}}
            if args.max_pixels
            else {}
        ),
        limit_mm_per_prompt={
            "image": 0 if is_video else args.num_images,
            "video": 1 if is_video else 0,
        },
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": True},
                # See run.py: the runner's default extra option breaks codegen
                # for this model's decode graph.
                "hlo2tensorizer_options": "",
            },
            # Supplying this at all is what selects the VL implementation.
            "vision_neuron_config": {
                "num_vision_tokens_buckets": [bucket],
                "vision_attention_block_size": args.vision_block_size,
                # Unset, resolve_tp_dp() picks tp=1/dp=world_size and the encoder
                # runs replicated, so a single-image request uses one rank.
                **(
                    {"tp_size": args.vision_tp, "dp_size": 1}
                    if args.vision_tp
                    else {}
                ),
            },
        },
    )

    if is_video:
        question, inputs = build_video_input(args.model, args.video_frames)
    else:
        question, inputs = build_input(
            args.model, args.image_size, args.num_images
        )
    outputs = llm.generate(
        [inputs], SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    )
    print(f"\nQuestion:  {question!r}")
    print(f"Generated: {outputs[0].outputs[0].text!r}\n")


if __name__ == "__main__":
    main()
