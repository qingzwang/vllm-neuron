# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness check: Neuron InternVL3-8B vs the HF reference.

``validate_vision_encoder.py`` compares the vision tower and projector in
isolation. This compares what actually ships: the whole model, on a real photo,
through greedy decoding — which is the only thing that exercises the TP=4 text
shards, the encoder-cache round trip and the vision merge together.

Two halves, run separately so they never contend for the box:

    # reference, CPU only, no Neuron device needed
    python compare_vs_hf.py --side hf --out /tmp/hf_ref.json

    # implementation under test, TP=4 on device
    python compare_vs_hf.py --side neuron --out /tmp/neuron_out.json

    # verdict
    python compare_vs_hf.py --side compare --hf /tmp/hf_ref.json \
        --neuron /tmp/neuron_out.json

Greedy on both sides, so agreement is a token-id comparison rather than a
judgement call. Some late divergence is expected and not a bug: the reference runs
float32 on CPU and the implementation runs bf16 with NKI kernels, so once two
candidate logits are within bf16 noise the argmax can legitimately differ, and one
different token changes every token after it. What would be a real failure is
divergence in the first few tokens, or text that stops describing the photo.

The images ship with the checkpoint (``examples/image1.jpg``), so this needs no
network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODEL = os.environ.get("INTERNVL_PATH", "/mnt/nvme/models/InternVL3-8B-Instruct")

# Questions with checkable answers, so a human can spot a wrong-but-fluent answer.
# Kept short: every extra prompt token is another CPU-side reference token.
PROMPTS = [
    ("image1.jpg", "Describe this image in detail."),
    ("image1.jpg", "What animal is in this image? Answer with one word."),
    ("image2.jpg", "Describe this image in detail."),
]
MAX_TOKENS = 48


def stub_timm() -> None:
    """Satisfy modeling_intern_vit's ``from timm.layers import DropPath``.

    timm is not in the DLAMI venv and installing into it is off-limits. This
    checkpoint has drop_path_rate=0.0, so DropPath is the identity and the stub is
    faithful rather than an approximation. transformers must be imported first:
    its lazy-module init probes for timm with find_spec, which chokes on a bare
    stub module.
    """
    import importlib.machinery
    import types

    import torch.nn as nn

    if "timm" in sys.modules:
        return

    import transformers  # noqa: F401  (see docstring)

    class DropPath(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, x):
            return x

    timm = types.ModuleType("timm")
    timm.__spec__ = importlib.machinery.ModuleSpec("timm", None, is_package=True)
    timm.__path__ = []
    layers = types.ModuleType("timm.layers")
    layers.__spec__ = importlib.machinery.ModuleSpec("timm.layers", None)
    layers.DropPath = DropPath
    timm.layers = layers
    sys.modules["timm"] = timm
    sys.modules["timm.layers"] = layers


def patch_transformers_for_old_remote_code() -> None:
    """Give ``PreTrainedModel`` a default ``all_tied_weights_keys``.

    The installed transformers sets that attribute in ``post_init()``, and this
    checkpoint's remote ``InternVLChatModel.__init__`` never calls ``post_init``,
    so loading dies in ``_move_missing_keys_from_meta_to_device`` with
    "object has no attribute 'all_tied_weights_keys'".

    A class-level empty dict is the least invasive fix: compliant models overwrite
    it with a per-instance dict in ``post_init``, and this model ties nothing
    (``tie_word_embeddings`` is False), so the shared dict is only ever read. It is
    shared mutable state in principle -- acceptable in a single-shot reference
    script, not something to copy into library code.
    """
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


def build_prompt(tok, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": f"<image>\n{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_hf(args) -> list[dict]:
    """Reference: the checkpoint's own HF implementation, float32 on CPU.

    float32 rather than bf16 on purpose. This side is the yardstick, so it should
    carry as little numerical noise of its own as possible, and CPU float32 matmul
    is also considerably faster than CPU bf16.
    """
    stub_timm()
    patch_transformers_for_old_remote_code()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    from vllm.transformers_utils.processors.internvl import (
        build_transform,
        dynamic_preprocess_internvl,
        get_internvl_target_ratios,
    )

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()

    # Tile with the same code vLLM's processor uses, so the two sides see
    # identical pixels and any difference is the model, not the preprocessing.
    ratios = get_internvl_target_ratios(1, 12)
    transform = build_transform(input_size=448)

    results = []
    for fname, question in PROMPTS:
        image = Image.open(f"{MODEL}/examples/{fname}").convert("RGB")
        tiles = dynamic_preprocess_internvl(
            image, target_ratios=ratios, image_size=448, use_thumbnail=True
        )
        pixel_values = torch.stack([transform(t) for t in tiles]).to(torch.float32)

        print(f"[hf] {fname} {image.size} -> {len(tiles)} tiles: {question}", flush=True)
        text = model.chat(
            tok,
            pixel_values,
            question,
            dict(max_new_tokens=MAX_TOKENS, do_sample=False),
        )
        ids = tok(text, add_special_tokens=False).input_ids
        print(f"[hf] {text!r}\n", flush=True)
        results.append(
            {
                "image": fname,
                "question": question,
                "tiles": len(tiles),
                "text": text,
                "token_ids": ids,
            }
        )
    return results


def run_neuron(args) -> list[dict]:
    """Implementation under test: TP=4 on device, bf16, greedy."""
    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
    os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "3600")
    os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

    from PIL import Image
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.processors.internvl import (
        dynamic_preprocess_internvl,
        get_internvl_target_ratios,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ratios = get_internvl_target_ratios(1, 12)

    images, tile_counts = [], []
    for fname, _ in PROMPTS:
        image = Image.open(f"{MODEL}/examples/{fname}").convert("RGB")
        images.append(image)
        tile_counts.append(
            len(
                dynamic_preprocess_internvl(
                    image, target_ratios=ratios, image_size=448, use_thumbnail=True
                )
            )
        )

    # Compile for the worst case across the prompt set; a single bucket keeps this
    # to one graph, and the padding cost does not matter for a correctness run.
    max_tiles = max(tile_counts)
    max_model_len = max(
        256, -(-(max_tiles * 256 + 64 + MAX_TOKENS) // 256) * 256
    )
    print(f"[neuron] tiles per prompt {tile_counts}, bucket for {max_tiles}", flush=True)

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        tensor_parallel_size=4,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [max_model_len],
                "num_seqs_buckets": [1],
                "on_device_sampling_config": {"all_greedy": True},
            },
            "vision_neuron_config": {
                "num_vision_tokens_buckets": [max_tiles * 1024],
                "vision_attention_block_size": 1024,
                "encoder_cache_num_blocks": max_tiles * 2 + 8,
            },
        },
        disable_log_stats=True,
    )

    results = []
    for (fname, question), image, tiles in zip(PROMPTS, images, tile_counts):
        out = llm.generate(
            [
                {
                    "prompt": build_prompt(tok, question),
                    "multi_modal_data": {"image": image},
                }
            ],
            SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
        )
        text = out[0].outputs[0].text
        print(f"[neuron] {fname} ({tiles} tiles) {question}\n[neuron] {text!r}\n", flush=True)
        results.append(
            {
                "image": fname,
                "question": question,
                "tiles": tiles,
                "text": text,
                "token_ids": list(out[0].outputs[0].token_ids),
            }
        )
    return results


def compare(args) -> int:
    hf = json.loads(Path(args.hf).read_text())
    nr = json.loads(Path(args.neuron).read_text())
    if len(hf) != len(nr):
        raise SystemExit(f"{len(hf)} reference vs {len(nr)} neuron results")

    worst = 0
    print("=" * 78)
    for h, n in zip(hf, nr):
        assert (h["image"], h["question"]) == (n["image"], n["question"])
        # Tokenising HF's text is not the same as HF's own sampled ids, so compare
        # on text prefix as well as ids and report the softer of the two.
        a, b = h["token_ids"], n["token_ids"]
        first_div = next(
            (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
        )
        ha, na = h["text"].strip(), n["text"].strip()
        common = 0
        for x, y in zip(ha, na):
            if x != y:
                break
            common += 1
        print(f"{h['image']} ({h['tiles']} tiles) | {h['question']}")
        print(f"  hf     : {ha!r}")
        print(f"  neuron : {na!r}")
        print(
            f"  identical text prefix: {common}/{min(len(ha), len(na))} chars; "
            f"token ids agree for {first_div}"
        )
        print(f"  exact text match: {ha == na}")
        print("-" * 78)
        worst = max(worst, 0 if ha == na else 1)

    print(
        "Verdict guide: exact match on every prompt is a pass. Divergence late in a\n"
        "long description is expected (float32 CPU reference vs bf16 device), and\n"
        "the check is that the answer still describes the same photo. Divergence in\n"
        "the first few tokens, or an answer about a different subject, is a bug."
    )
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=("hf", "neuron", "compare"), required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--hf", type=Path, default=Path("/tmp/hf_ref.json"))
    p.add_argument("--neuron", type=Path, default=Path("/tmp/neuron_out.json"))
    args = p.parse_args()

    if args.side == "compare":
        raise SystemExit(compare(args))

    results = run_hf(args) if args.side == "hf" else run_neuron(args)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
