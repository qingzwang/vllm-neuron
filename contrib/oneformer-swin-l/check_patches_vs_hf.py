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

Give it more than one ``--size`` and it becomes a second check as well, on the patches
themselves rather than on the model: every input size the patches need is pinned to the
*instance*, so one process can hold several, and the way to find out whether that is
actually true is to run several through one process and require every one of them to match.
A size-dependent constant left in a module global passes at the first size and fails at the
second.

Usage:
    python contrib/oneformer-swin-l/check_patches_vs_hf.py \\
        --model /mnt/nvme/models/oneformer_coco_swin_large --size 640
    ... check_patches_vs_hf.py --size 640 480x864          # two sizes, one process
    ... check_patches_vs_hf.py --all-buckets               # every entry in src/buckets.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src import buckets  # noqa: E402 — after sys.path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme/models/oneformer_coco_swin_large")
    ap.add_argument("--size", nargs="+", default=["640"], metavar="H[xW]",
                    help="input sizes, each side a multiple of 32; '640' means 640x640. "
                         "This check is the cheap way to find out whether a new size works "
                         "at all: Swin pads every stage up to the window size, and a "
                         "padding bug would show up here before any compile time is spent. "
                         "More than one size also checks that the patches keep them apart")
    ap.add_argument("--all-buckets", action="store_true",
                    help="use every size in src/buckets.py, plus their transposes, which "
                         "is the case a length-keyed cache would get wrong")
    ap.add_argument("--task", default="panoptic")
    ap.add_argument("--tol", type=float, default=1e-5, help="relative tolerance")
    return ap.parse_args()


def build_inputs(height, width, seed=0):
    """Synthetic but shaped exactly like the processor's output.

    Synthetic on purpose: this check is about two implementations of the same maths
    agreeing, and random input exercises the deformable sampler over the whole
    coordinate range including out-of-range offsets, which a photo may not.
    """
    generator = torch.Generator().manual_seed(seed)
    pixel_values = torch.randn(1, 3, height, width, generator=generator)
    # The task token: whatever the tokenizer produces for this task, but its exact ids
    # do not matter here as long as both runs see the same ones.
    task_inputs = torch.randint(0, 49407, (1, 77), generator=generator)
    return {"pixel_values": pixel_values, "task_inputs": task_inputs}


def load(model_path):
    from transformers import OneFormerForUniversalSegmentation

    model = OneFormerForUniversalSegmentation.from_pretrained(model_path)
    return model.eval()


def forward(model, inputs):
    with torch.no_grad():
        out = model(**inputs)
    return out.class_queries_logits, out.masks_queries_logits


def main():
    args = parse_args()
    if args.all_buckets:
        sizes = [hw for h, w in buckets.BUCKETS for hw in ((h, w), (w, h))]
    else:
        sizes = [buckets.parse_size(text) for text in args.size]
    sizes = list(dict.fromkeys(sizes))
    inputs = {hw: build_inputs(*hw) for hw in sizes}
    for hw in sizes:
        print(f"  {buckets.describe(*hw)}")
    print()

    # Every unpatched forward happens before install(), because install() patches the
    # classes -- both models would share them, so an "unpatched" run afterwards is not one.
    print(f"[1/2] unpatched HuggingFace, {len(sizes)} size(s)")
    import transformers.models.oneformer.modeling_oneformer as m

    importlib.reload(m)  # make sure no earlier patch is still installed
    reference_model = load(args.model)
    reference = {}
    for hw in sizes:
        reference[hw] = forward(reference_model, inputs[hw])
        print(f"      {hw[0]}x{hw[1]}: class {tuple(reference[hw][0].shape)}  "
              f"mask {tuple(reference[hw][1].shape)}")
    del reference_model

    print(f"[2/2] with the Neuron patches, the same {len(sizes)} size(s) in this process")
    from src import patches

    importlib.reload(patches)
    patches.install()
    assert patches.is_installed()
    model = load(args.model)
    print(f"      replaced {patches.replace_gelu(model)} GELU activation(s), "
          f"cached {patches.cache_position_embeddings(model)} position table(s)")
    patches.relax_shape_assert()
    patches.patch_pixel_decoder_forward()
    patches.patch_prediction_heads()

    got = {}
    for hw in sizes:
        # The one thing that changes per size, and the point of the multi-size run: the
        # same instance is re-pinned and has to come out right at every size, including
        # after coming back to one it has already served.
        patches.pin_input_size(model, buckets.level_shapes(*hw))
        got[hw] = forward(model, inputs[hw])
    if len(sizes) > 1:
        # Back to the first size last, on an instance that has since served every other
        # one. A cache keyed by anything ambiguous -- the flattened length, say, which
        # equal-area sizes share -- passes the loop above and fails here.
        patches.pin_input_size(model, buckets.level_shapes(*sizes[0]))
        got[sizes[0]] = forward(model, inputs[sizes[0]])
        print(f"      re-ran {sizes[0][0]}x{sizes[0][1]} after all the others")

    failures = []
    for hw in sizes:
        print(f"\n  {hw[0]}x{hw[1]}")
        for name, a, b in (("class_queries_logits", got[hw][0], reference[hw][0]),
                           ("masks_queries_logits", got[hw][1], reference[hw][1])):
            scale = max(b.abs().max().item(), 1e-6)
            rel = (a - b).abs().max().item() / scale
            verdict = "PASS" if rel <= args.tol else "FAIL"
            print(f"    {verdict}  {name}: rel={rel:.3e}  bitwise={torch.equal(a, b)}  "
                  f"(ref max |x| = {scale:.3f})")
            if verdict == "FAIL":
                failures.append(f"{hw[0]}x{hw[1]} {name}")

        # The argmax over classes is what post-processing acts on, so check it separately:
        # a tiny logit difference that flips a label matters more than its magnitude.
        same = torch.equal(got[hw][0].argmax(-1), reference[hw][0].argmax(-1))
        print(f"    {'PASS' if same else 'FAIL'}  per-query argmax label unchanged")
        if not same:
            failures.append(f"{hw[0]}x{hw[1]} argmax")

    print()
    if failures:
        raise SystemExit(f"patches changed the model: {', '.join(failures)}")
    print(f"the patched model is the same model, at {len(sizes)} size(s) in one process")


if __name__ == "__main__":
    main()
