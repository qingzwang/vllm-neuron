#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Segment real images on a NeuronCore, next to CPU, with per-stage latency.

Three stages, and only one of them is on the device:

    preprocess   resize + normalize + the task token, on the host (the processor)
    forward      the compiled graph: backbone, pixel decoder, transformer decoder
    postprocess  logits -> segments: sigmoid, threshold, argmax, upsample, on the host

Steady state only: a couple of calls are discarded first, because the first forward in a
process also loads the NEFF onto the device, and the first postprocess pays for
importing scipy and friends. Medians over ``--iterations``.

Usage:
    python contrib/oneformer-swin-l/segment.py --image a.jpg b.jpg --compare-cpu \\
        --out /tmp/of_seg --iterations 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import vllm_neuron  # noqa: F401,E402 — registers the "neuron_libtorch" Dynamo backend

DEVICE = torch.device("neuron", 0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme/models/oneformer_coco_swin_large")
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--task", default="panoptic",
                    choices=("panoptic", "semantic", "instance"))
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--compare-cpu", action="store_true",
                    help="also run the same model on CPU and diff the segmentation")
    ap.add_argument("--cpu-iterations", type=int, default=3)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--compiler-args", default=None,
                    help="passed verbatim to neuronx-cc, e.g. "
                         "'--auto-cast=matmult --auto-cast-type=bf16'")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


class Heads(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, task_inputs, pixel_mask):
        out = self.model(
            pixel_values=pixel_values, task_inputs=task_inputs, pixel_mask=pixel_mask
        )
        return out.class_queries_logits, out.masks_queries_logits


def build(model_path, size):
    """Load, patch, and return (model, wrapper). See src/patches.py for the eight."""
    from src import patches

    level_shapes = [(size // s, size // s) for s in (32, 16, 8)]
    patches.install(level_shapes)

    from transformers import OneFormerForUniversalSegmentation

    model = OneFormerForUniversalSegmentation.from_pretrained(model_path).eval()
    patches.replace_gelu(model)
    patches.cache_position_embeddings(model)
    patches.cache_reference_points(model, level_shapes)
    patches.relax_shape_assert()
    patches.constant_fold_pixel_decoder_split(level_shapes)
    patches.detensorize_mask_threshold()
    return model, Heads(model), patches


def timed(fn, iterations, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples), max(samples)


def summarize(processor, config, cls_logits, mask_logits, task, size):
    from types import SimpleNamespace

    outputs = SimpleNamespace(
        class_queries_logits=cls_logits, masks_queries_logits=mask_logits
    )
    post = getattr(processor, f"post_process_{task}_segmentation")
    result = post(outputs, target_sizes=[(size, size)])[0]
    if isinstance(result, dict):
        seg = result["segmentation"].cpu().numpy()
        rows = []
        for info in result.get("segments_info", []):
            label = config.id2label.get(
                info["label_id"], config.id2label.get(str(info["label_id"]), "?")
            )
            rows.append((label, float(info.get("score", 1.0)),
                         int((seg == info["id"]).sum())))
    else:
        seg = result.cpu().numpy()
        rows = [
            (config.id2label.get(int(v), config.id2label.get(str(int(v)), "?")),
             1.0, int(c))
            for v, c in zip(*np.unique(seg, return_counts=True))
        ]
    rows.sort(key=lambda r: -r[2])
    return seg, rows


def save_overlay(image, seg, path, size):
    rng = np.random.default_rng(0)
    top = int(seg.max()) + 2
    palette = rng.integers(40, 235, size=(top, 3), dtype=np.uint8)
    rgb = palette[np.clip(seg + 1, 0, top - 1)]
    base = np.asarray(image.resize((size, size), Image.LANCZOS), dtype=np.float32)
    Image.fromarray((0.45 * base + 0.55 * rgb).astype(np.uint8)).save(path)


def main():
    args = parse_args()
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, OneFormerProcessor

    processor = OneFormerProcessor.from_pretrained(args.model)
    processor.image_processor.size = {"height": args.size, "width": args.size}
    config = AutoConfig.from_pretrained(args.model)

    # Two independent instances when comparing: one stays on the host, one moves to the
    # device. They cannot be the same object -- moving it would take the CPU side with it.
    dev_model, device_wrapper, patches = build(args.model, args.size)
    cpu_wrapper = build(args.model, args.size)[1] if args.compare_cpu else None

    pixel_mask = torch.ones(1, args.size, args.size)

    # One host forward first: it is what fills the position and reference caches, which
    # then get copied to the device. Building them on the device is what patches 4 and 5
    # exist to avoid.
    warm = processor(images=Image.open(args.image[0]).convert("RGB"),
                     task_inputs=[args.task], return_tensors="pt")
    with torch.no_grad():
        device_wrapper(warm["pixel_values"], warm["task_inputs"], pixel_mask)

    dtype = getattr(torch, args.dtype)
    device_wrapper = device_wrapper.to(device=DEVICE, dtype=dtype)
    print(f"[setup] copied {patches.move_position_cache(dev_model, DEVICE, dtype)} "
          f"position table(s) to the device, reference grid: "
          f"{patches.move_reference_cache(DEVICE, dtype)}")
    compile_options = {"compiler_args": args.compiler_args} if args.compiler_args else {}
    if compile_options:
        print(f"[setup] neuronx-cc args: {args.compiler_args}")
    compiled = torch.compile(
        device_wrapper, backend="neuron_libtorch", fullgraph=True,
        options=compile_options,
    )

    for path in args.image:
        image = Image.open(path).convert("RGB")
        name = Path(path).stem
        print(f"\n=== {path}  ({image.width}x{image.height} -> {args.size}x{args.size})")

        def preprocess():
            return processor(images=image, task_inputs=[args.task], return_tensors="pt")

        inputs = preprocess()
        pv = inputs["pixel_values"]
        ti = inputs["task_inputs"]
        # task_inputs stays integral: they are token ids, not activations.
        dev_inputs = [pv.to(device=DEVICE, dtype=dtype), ti.to(DEVICE),
                      pixel_mask.to(device=DEVICE, dtype=dtype)]

        def forward_device():
            with torch.no_grad():
                cls, msk = compiled(*dev_inputs)
            # .cpu() first: casting bf16 on the device fails with a dtype mismatch.
            return cls.cpu().float(), msk.cpu().float()

        dev_cls, dev_msk = forward_device()

        def postprocess():
            return summarize(processor, config, dev_cls, dev_msk, args.task, args.size)

        pre_ms = timed(preprocess, args.iterations, args.warmup)
        fwd_ms = timed(forward_device, args.iterations, args.warmup)
        post_ms = timed(postprocess, args.iterations, args.warmup)

        dev_seg, dev_rows = postprocess()

        print(f"  {'stage':<14}{'median':>10}{'min':>10}{'max':>10}")
        for label, (median, low, high) in (
            ("preprocess", pre_ms), ("forward (device)", fwd_ms), ("postprocess", post_ms)
        ):
            print(f"  {label:<14}{median:>9.2f}ms{low:>9.2f}ms{high:>9.2f}ms")
        total = pre_ms[0] + fwd_ms[0] + post_ms[0]
        print(f"  {'end to end':<14}{total:>9.2f}ms")

        if args.compare_cpu:
            def forward_cpu():
                with torch.no_grad():
                    return cpu_wrapper(pv, ti, pixel_mask)

            cpu_cls, cpu_msk = forward_cpu()
            cpu_fwd = timed(forward_cpu, args.cpu_iterations, 1)
            cpu_seg, cpu_rows = summarize(
                processor, config, cpu_cls, cpu_msk, args.task, args.size
            )
            print(f"  {'forward (CPU)':<14}{cpu_fwd[0]:>9.2f}ms"
                  f"{cpu_fwd[1]:>9.2f}ms{cpu_fwd[2]:>9.2f}ms"
                  f"   ({cpu_fwd[0] / fwd_ms[0]:.0f}x the device)")

            agree = float((cpu_seg == dev_seg).mean())
            same = [r[0] for r in cpu_rows] == [r[0] for r in dev_rows]
            print(f"\n  {'segment':<22}{'CPU':>22}{'Neuron':>22}")
            for i in range(max(len(cpu_rows), len(dev_rows))):
                left = cpu_rows[i] if i < len(cpu_rows) else None
                right = dev_rows[i] if i < len(dev_rows) else None
                fmt = lambda r: "-" if r is None else f"{r[0]} {r[1]:.3f} {r[2]}px"
                print(f"  {i:<22}{fmt(left):>22}{fmt(right):>22}")
            print(f"\n  same segments: {same} | pixel agreement: {agree * 100:.3f}%")
            if out:
                save_overlay(image, cpu_seg, out / f"{name}_{args.task}_cpu.png", args.size)
        if out:
            save_overlay(image, dev_seg, out / f"{name}_{args.task}_neuron.png", args.size)
            print(f"  wrote {out}/{name}_{args.task}_*.png")


if __name__ == "__main__":
    main()
