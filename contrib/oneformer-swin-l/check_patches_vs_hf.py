#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Do the Neuron patches change the model? Answered on CPU, before any compilation.

``src/patches.py`` replaces two pieces of HuggingFace's OneFormer: the deformable
attention (because ``grid_sample`` aborts the runtime) and the fully-masked-row guard
(because a data-dependent index cannot be traced). Both are meant to be exact
substitutions, so the whole model must produce the same class and mask logits either
way — on CPU, where nothing about Neuron is involved and a difference can only be the
patch's fault.

This is the cheap check that has to pass before the expensive one is worth running:
it takes seconds, needs no device, and it separates "the patch is wrong" from "the
compiler is wrong", which are otherwise indistinguishable at the end.

Usage:
    python contrib/oneformer-swin-l/check_patches_vs_hf.py \\
        --model /mnt/nvme/models/oneformer_coco_swin_large --size 384
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme/models/oneformer_coco_swin_large")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--task", default="panoptic")
    ap.add_argument("--tol", type=float, default=1e-5, help="relative tolerance")
    return ap.parse_args()


def build_inputs(size, task, seed=0):
    """Synthetic but shaped exactly like the processor's output.

    Synthetic on purpose: this check is about two implementations of the same maths
    agreeing, and random input exercises the deformable sampler over the whole
    coordinate range including out-of-range offsets, which a photo may not.
    """
    generator = torch.Generator().manual_seed(seed)
    pixel_values = torch.randn(1, 3, size, size, generator=generator)
    # The task token: whatever the tokenizer produces for this task, but its exact ids
    # do not matter here as long as both runs see the same ones.
    task_inputs = torch.randint(0, 49407, (1, 77), generator=generator)
    return {"pixel_values": pixel_values, "task_inputs": task_inputs}


def run(model_path, inputs, replace_gelu=None):
    from transformers import OneFormerForUniversalSegmentation

    model = OneFormerForUniversalSegmentation.from_pretrained(model_path)
    model.eval()
    if replace_gelu is not None:
        from src import patches as _p

        print(f"      replaced {replace_gelu(model)} GELU activation(s), "
              f"cached {_p.cache_position_embeddings(model)} position table(s)")
        _p.cache_reference_points(model, _p._installed_level_shapes)
        _p.relax_shape_assert()
        _p.constant_fold_pixel_decoder_split(_p._installed_level_shapes)
        _p.detensorize_mask_threshold()
    with torch.no_grad():
        out = model(**inputs)
    return out.class_queries_logits, out.masks_queries_logits


def main():
    args = parse_args()
    inputs = build_inputs(args.size, args.task)

    # Level shapes the pixel decoder sees, in the model's own order: smallest map
    # first, i.e. stride 32, 16, 8. Reversing this is silent — same total positions.
    level_shapes = [(args.size // s, args.size // s) for s in (32, 16, 8)]
    print(f"size {args.size}, deformable level shapes {level_shapes}\n")

    print("[1/2] unpatched HuggingFace")
    import transformers.models.oneformer.modeling_oneformer as m

    importlib.reload(m)  # make sure no earlier patch is still installed
    ref_cls, ref_mask = run(args.model, inputs)
    print(f"      class {tuple(ref_cls.shape)}  mask {tuple(ref_mask.shape)}")

    print("[2/2] with the Neuron patches")
    from src import patches

    importlib.reload(patches)
    patches.install(level_shapes)
    assert patches.is_installed()
    got_cls, got_mask = run(args.model, inputs, replace_gelu=patches.replace_gelu)

    failures = []
    for name, got, ref in (("class_queries_logits", got_cls, ref_cls),
                           ("masks_queries_logits", got_mask, ref_mask)):
        scale = max(ref.abs().max().item(), 1e-6)
        rel = (got - ref).abs().max().item() / scale
        exact = torch.equal(got, ref)
        verdict = "PASS" if rel <= args.tol else "FAIL"
        print(f"\n  {verdict}  {name}: rel={rel:.3e}  bitwise={exact}  "
              f"(ref max |x| = {scale:.3f})")
        if verdict == "FAIL":
            failures.append(name)

    # The argmax over classes is what post-processing acts on, so check it separately:
    # a tiny logit difference that flips a label matters more than its magnitude.
    same_labels = torch.equal(got_cls.argmax(-1), ref_cls.argmax(-1))
    print(f"  {'PASS' if same_labels else 'FAIL'}  per-query argmax label unchanged")
    if not same_labels:
        failures.append("argmax")

    print()
    if failures:
        raise SystemExit(f"patches changed the model: {', '.join(failures)}")
    print("the patched model is the same model")


if __name__ == "__main__":
    main()
