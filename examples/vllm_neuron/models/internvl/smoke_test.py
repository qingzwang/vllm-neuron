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


def main():
    from PIL import Image
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    # A square 448x448 image lands on aspect ratio (1,1) in dynamic tiling, which
    # is a single tile, and no thumbnail is added when blocks == 1. So the vision
    # side is exactly 1024 patches -> 256 tokens without needing processor kwargs
    # (mm_processor_kwargs are forwarded to the video processor too, which rejects
    # max_dynamic_patch).
    image = Image.new("RGB", (448, 448), (90, 140, 200))

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # InternVL's chat template wraps the image span in <img>...</img> and vLLM's
    # processor expands <image> into IMG_CONTEXT tokens.
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": f"<image>\n{QUESTION}"}],
        tokenize=False,
        add_generation_prompt=True,
    )

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
                # bucket for 13 tiles rather than the single tile this test sends.
                "num_vision_tokens_buckets": [13312],
                "vision_attention_block_size": 1024,
                "encoder_cache_num_blocks": 64,
            },
        },
        disable_log_stats=True,
    )

    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        SamplingParams(max_tokens=48, temperature=0.0),
    )
    text = outputs[0].outputs[0].text
    print("\n" + "=" * 70)
    print("GENERATED:", repr(text))
    print("=" * 70)
    print(
        "\nSanity: a flat blue image should draw words like blue / solid / plain.\n"
        "Garbage or repeated punctuation means a numerical bug, not a config issue."
    )


if __name__ == "__main__":
    main()
