# SPDX-License-Identifier: Apache-2.0
"""On-device smoke test for InternVL3-8B on Neuron.

Minimal single-image, single-request run at TP=4. A square 448x448 image yields
one tile, so the vision path stays at 256 embed tokens and the first compile is
as small as possible.

READ THE GENERATED TEXT, not just "it ran". A wrong TP shard, a missed LayerScale
or a mis-ordered pixel shuffle all complete successfully and emit garbage.

Usage:
    PATH=$V/bin:$PATH PYTHONPATH=/mnt/nvme/vllm-neuron \\
      python examples/vllm_neuron/models/internvl/smoke_test.py
"""

import os

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "3600")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
os.environ.setdefault("NEURON_CC_FLAGS", "--temp-dir=/tmp/neuroncc_tmp")
os.makedirs("/tmp/neuroncc_tmp", exist_ok=True)

MODEL = os.environ.get("INTERNVL_PATH", "/mnt/nvme/models/InternVL3-8B-Instruct")
QUESTION = "Describe this image in detail."

# Debug aid for stalls in this process. A timer armed here does NOT cover
# EngineCore (it runs in a forked child, and this process's stack only ever shows
# zmq.poll under wait_for_engine_startup) -- use
# VLLM_NEURON_DEBUG_STACK_SECONDS for that, which arms inside EngineCore. Setting
# VLLM_ENABLE_V1_MULTIPROCESSING=0 is the other option: it collapses EngineCore
# into this process, which also turns worker errors into a plain traceback here.
if os.environ.get("IVL_STACK_DUMP_SECONDS"):
    import faulthandler

    faulthandler.dump_traceback_later(
        float(os.environ["IVL_STACK_DUMP_SECONDS"]), repeat=True, exit=False
    )


def main():
    from PIL import Image
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    # Case 1: a square 448x448 image lands on aspect ratio (1,1) in dynamic tiling,
    # which is a single tile, and no thumbnail is added when blocks == 1. So the
    # vision side is exactly 1024 patches -> 256 tokens without needing processor
    # kwargs (mm_processor_kwargs are forwarded to the video processor too, which
    # rejects max_dynamic_patch).
    flat = Image.new("RGB", (448, 448), (90, 140, 200))

    # Case 2: exercise dynamic tiling, which case 1 cannot. A 2:1 image tiles into
    # 2 crops plus a thumbnail = 3 tiles, so this is the first thing to cover the
    # multi-tile path: pixel-shuffle output concatenated across tiles, several
    # cache blocks per item, and padding up to the compiled bucket.
    #
    # Left half red, right half blue makes tile ORDER checkable from the text. Tile
    # order is easy to get wrong (pixel shuffle's trailing permute, the flat->block
    # reshape) and a wrong order still produces fluent output, just with the colours
    # swapped -- so read which colour the model puts on the left.
    tiled = Image.new("RGB", (896, 448), (200, 40, 40))
    tiled.paste(Image.new("RGB", (448, 448), (40, 60, 200)), (448, 0))

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def build(question):
        # InternVL's chat template wraps the image span in <img>...</img> and vLLM's
        # processor expands <image> into IMG_CONTEXT tokens.
        return tok.apply_chat_template(
            [{"role": "user", "content": f"<image>\n{question}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    cases = [
        ("1 tile, flat blue", flat, QUESTION),
        (
            "3 tiles (2x1 + thumbnail), red left / blue right",
            tiled,
            "What colour is the left half of this image, and the right half?",
        ),
    ]

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        tensor_parallel_size=4,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [2048],
                "num_seqs_buckets": [1],
                "on_device_sampling_config": {"all_greedy": True},
            },
            "vision_neuron_config": {
                # Raw (pre-merge) patches. The runner derives
                #   max_vision_blocks_per_request
                #     = ceil(bucket / merge_factor / cache_block_size)
                # and a request needing more blocks than that can never be
                # scheduled: the scheduler then spins with no log output at all,
                # which reads as a hang. Dynamic tiling emits up to
                # max_dynamic_patch (12) tiles plus a thumbnail, so size the
                # bucket for 13 tiles rather than the tile counts this test sends.
                "num_vision_tokens_buckets": [13312],
                "vision_attention_block_size": 1024,
                "encoder_cache_num_blocks": 64,
            },
        },
        disable_log_stats=True,
    )

    # One generate() per case: each is a separate prefill, so a failure is
    # attributable to that tile count rather than to batching.
    results = []
    for label, image, question in cases:
        outputs = llm.generate(
            [{"prompt": build(question), "multi_modal_data": {"image": image}}],
            SamplingParams(max_tokens=64, temperature=0.0),
        )
        results.append((label, outputs[0].outputs[0].text))

    print("\n" + "=" * 70)
    for label, text in results:
        print(f"[{label}]\n  {text!r}\n")
    print("=" * 70)
    print(
        "\nSanity checks:\n"
        "  1 tile  -> words like blue / solid / plain.\n"
        "  3 tiles -> red on the LEFT, blue on the RIGHT. Swapped colours mean the\n"
        "             tile order is wrong, not that the model is confused.\n"
        "Garbage or repeated punctuation means a numerical bug, not a config issue."
    )


if __name__ == "__main__":
    main()
