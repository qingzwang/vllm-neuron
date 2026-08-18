#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compare greedy generation on the Neuron device against HuggingFace on CPU.

The other checks establish that the *implementation* is right: ``check_model_vs_hf``
matches HF layer by layer in float32 on CPU, down to the same greedy token. What
they cannot show is what bf16 on device plus many decode steps does to the output,
because they only run one prefill.

So this generates the same tokens both ways and reports where they diverge. Two
numbers matter:

* **prefix length** — how many leading tokens match exactly. Divergence in the
  first few tokens is a bug; divergence later is expected, because bf16 and an
  independent implementation will eventually disagree on a near-tie and greedy
  decoding then amplifies it.
* **token match rate** — the reference port's own published bar is 53/80 tokens
  (66%) with 3/5 prompts exact, so that is the level to compare against rather
  than 100%.

HF runs in float32 on CPU, which is slow: budget a couple of minutes per prompt
at 32 tokens. Run the device side first (it writes a JSON), then the HF side, so
the device is free while HF grinds.

Usage:

    # 1. on the device
    ... PATH=$V/bin:$PATH $V/bin/python check_generation_vs_hf.py --side neuron
    # 2. on CPU, then compare
    PYTHONPATH=/mnt/nvme/vllm-neuron python check_generation_vs_hf.py --side hf
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1800")

PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "def fibonacci(n):",
    "Once upon a time, there was a",
    "The three primary colours are",
]

DEFAULT_JSON = "/tmp/qwen35_generation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument("--side", choices=("neuron", "hf"), required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--json", default=DEFAULT_JSON)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    return parser.parse_args()


def run_neuron(args) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=len(PROMPTS),
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [len(PROMPTS)],
                "on_device_sampling_config": {"all_greedy": True},
                "hlo2tensorizer_options": "",
            },
        },
    )
    outputs = llm.generate(
        PROMPTS, SamplingParams(max_tokens=args.tokens, temperature=0.0)
    )
    payload = {
        "tokens": args.tokens,
        "results": [
            {
                "prompt": prompt,
                "token_ids": list(out.outputs[0].token_ids),
                "text": out.outputs[0].text,
            }
            for prompt, out in zip(PROMPTS, outputs)
        ],
    }
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.json}")
    for entry in payload["results"]:
        print(f"  {entry['prompt']!r} -> {entry['text']!r}")


def run_hf_and_compare(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.json) as handle:
        payload = json.load(handle)
    if payload["tokens"] < args.tokens:
        raise SystemExit(
            f"{args.json} only has {payload['tokens']} tokens per prompt"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32
    ).eval()

    total_tokens = 0
    total_matched = 0
    exact_prompts = 0
    bad_first: list[str] = []
    print(f"comparing {args.tokens} greedy tokens per prompt\n")

    for entry in payload["results"]:
        prompt = entry["prompt"]
        device_ids = entry["token_ids"][: args.tokens]
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                min_new_tokens=args.tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        hf_ids = generated[0, inputs.input_ids.shape[1] :].tolist()[: args.tokens]

        matched = sum(a == b for a, b in zip(device_ids, hf_ids))
        prefix = 0
        for a, b in zip(device_ids, hf_ids):
            if a != b:
                break
            prefix += 1
        total_tokens += len(hf_ids)
        total_matched += matched
        exact = prefix == len(hf_ids)
        exact_prompts += int(exact)
        if prefix == 0:
            bad_first.append(prompt)

        flag = "EXACT" if exact else f"prefix {prefix}"
        print(f"  {flag:11s} {matched}/{len(hf_ids)} tokens   {prompt!r}")
        if not exact:
            print(f"      neuron: {tokenizer.decode(device_ids)!r}")
            print(f"      hf    : {tokenizer.decode(hf_ids)!r}")

    rate = 100.0 * total_matched / max(total_tokens, 1)
    print(
        f"\ntotal {total_matched}/{total_tokens} tokens ({rate:.1f}%), "
        f"{exact_prompts}/{len(payload['results'])} prompts exact"
    )
    print("reference port's published bar: 53/80 tokens (66%), 3/5 prompts exact")

    # A first-token mismatch is the failure mode this check exists to catch: it
    # means the prefill is wrong, not that bf16 lost a near-tie deep into decode.
    if bad_first:
        print(f"\nFIRST TOKEN WRONG for {len(bad_first)} prompt(s): {bad_first}")
        print("that is a prefill bug, not rounding — investigate before trusting output")
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.side == "neuron":
        run_neuron(args)
        return 0
    return run_hf_and_compare(args)


if __name__ == "__main__":
    sys.exit(main())
