# SPDX-License-Identifier: Apache-2.0
"""Text-only offline inference for Qwen3.5-2B on Neuron.

Qwen3.5 is a hybrid stack: 18 of its 24 layers are gated DeltaNet (a linear
recurrence with fixed-size state) and 6 are full attention. That makes it the
first model in this plugin to need two KV cache groups, so this script exists as
much to shake out the cache plumbing as to generate text.

Usage (this box: trn2.3xlarge, 4 NeuronCores, so TP=4 is the ceiling):

    NEURON_SKIP_EFA_AFFINITY=1 \
    VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm \
    NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp" \
    PYTHONPATH=/mnt/nvme/vllm-neuron \
    PATH=$V/bin:$PATH $V/bin/python \
      examples/vllm_neuron/models/qwen3_5/run.py --model /mnt/nvme/models/Qwen3.5-2B

``PATH`` must include the venv's bin: the plugin locates the compiler with
``shutil.which("neuronx-cc")``. ``PYTHONPATH`` must include the repo, because
the venv's editable install resolves through a static module map that predates
this package.
"""

import argparse
import os

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1800")

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "def fibonacci(n):",
    "Once upon a time, there was a",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--prefill-bucket", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.prefill_bucket,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        # Prefix caching hard-requires segmented prefill on this plugin, and a
        # DeltaNet layer has no notion of a reusable prefix anyway: its prefill
        # starts from a zero state.
        enable_prefix_caching=False,
        # The checkpoint declares a vision tower this port does not implement
        # yet. Refusing image and video items at the frontend is the honest
        # guard: otherwise a multimodal request would be answered from its text
        # alone, which reads as a bad model rather than a missing feature.
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.prefill_bucket],
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": True},
                # No extra hlo2tensorizer options. The runner's default
                # --modular-flow-mac-threshold=10 exists only for NKI kernels,
                # which this model has none of, and it makes neuronx-cc fail
                # codegen on the decode graph (NCC_IBTN006: a pftranspose whose
                # copy fails backend verification). Verified by recompiling the
                # cached HLO by hand: fails with the flag, succeeds without it.
                "hlo2tensorizer_options": "",
            },
        },
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    outputs = llm.generate(PROMPTS, sampling_params)
    for prompt, output in zip(PROMPTS, outputs):
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}\n")


if __name__ == "__main__":
    main()
