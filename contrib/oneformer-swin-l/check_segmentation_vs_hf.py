#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does the device output *segment* the image the same way? Logits are not the answer.

`run_device.py` compares logits, which is the right thing while bringing a model up but
the wrong thing for deciding whether a port is usable. Mask logits go through a sigmoid
and a 0.5 threshold, and class logits through an argmax, so what matters is whether the
segments come out the same -- their labels, their scores, and which pixels they cover.

Reads the two `.pt` files the other scripts write (HuggingFace on CPU, and the device
run) and post-processes both through the *same* processor.

Usage:
    python contrib/oneformer-swin-l/check_segmentation_vs_hf.py \\
        --ref /tmp/of_ref/hf_panoptic_384.pt --device /tmp/of_dev_logits.pt \\
        --out /tmp/of_seg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme/models/oneformer_coco_swin_large")
    ap.add_argument("--ref", required=True, help=".pt from check_hf_reference.py")
    ap.add_argument("--device", required=True, help=".pt from run_device.py --save")
    ap.add_argument("--task", default="panoptic",
                    choices=("panoptic", "semantic", "instance"))
    ap.add_argument("--out", default=None, help="write side-by-side masks here")
    return ap.parse_args()


def segment(processor, model_config, cls_logits, mask_logits, task, size):
    outputs = SimpleNamespace(
        class_queries_logits=cls_logits, masks_queries_logits=mask_logits
    )
    post = getattr(processor, f"post_process_{task}_segmentation")
    result = post(outputs, target_sizes=[(size, size)])[0]
    if isinstance(result, dict):
        seg = result["segmentation"].cpu().numpy()
        segments = []
        for info in result.get("segments_info", []):
            label = model_config.id2label.get(
                info["label_id"], model_config.id2label.get(str(info["label_id"]), "?")
            )
            segments.append(
                {
                    "id": info["id"],
                    "label": label,
                    "score": round(float(info.get("score", 1.0)), 4),
                    "pixels": int((seg == info["id"]).sum()),
                }
            )
    else:
        seg = result.cpu().numpy()
        segments = []
        for value, count in zip(*np.unique(seg, return_counts=True)):
            label = model_config.id2label.get(
                int(value), model_config.id2label.get(str(int(value)), "?")
            )
            segments.append({"id": int(value), "label": label, "pixels": int(count)})
    segments.sort(key=lambda s: -s["pixels"])
    return seg, segments


def main():
    args = parse_args()
    ref = torch.load(args.ref, weights_only=False)
    dev = torch.load(args.device, weights_only=False)
    size = ref["size"]

    from transformers import AutoConfig, OneFormerProcessor

    processor = OneFormerProcessor.from_pretrained(args.model)
    processor.image_processor.size = {"height": size, "width": size}
    config = AutoConfig.from_pretrained(args.model)

    cpu_seg, cpu_segments = segment(
        processor, config, ref["class_queries_logits"], ref["masks_queries_logits"],
        args.task, size,
    )
    dev_seg, dev_segments = segment(
        processor, config, dev["class_queries_logits"], dev["masks_queries_logits"],
        args.task, size,
    )

    print(f"{'':<4}{'CPU (HuggingFace)':<34}{'Neuron':<34}")
    for i in range(max(len(cpu_segments), len(dev_segments))):
        left = cpu_segments[i] if i < len(cpu_segments) else None
        right = dev_segments[i] if i < len(dev_segments) else None

        def fmt(entry):
            if entry is None:
                return "-"
            score = f" {entry['score']:.3f}" if "score" in entry else ""
            return f"{entry['label']}{score}  {entry['pixels']} px"

        print(f"{i:<4}{fmt(left):<34}{fmt(right):<34}")

    cpu_labels = [s["label"] for s in cpu_segments]
    dev_labels = [s["label"] for s in dev_segments]
    same_labels = cpu_labels == dev_labels
    agreement = float((cpu_seg == dev_seg).mean())

    print(f"\n  same segments, same order : {same_labels}")
    print(f"  pixel-for-pixel agreement : {agreement * 100:.3f}%")
    if "score" in (cpu_segments[0] if cpu_segments else {}):
        worst = max(
            abs(a["score"] - b["score"]) for a, b in zip(cpu_segments, dev_segments)
        ) if same_labels else float("nan")
        print(f"  largest score difference  : {worst:.4f}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        top = int(max(cpu_seg.max(), dev_seg.max())) + 2
        palette = rng.integers(40, 235, size=(top, 3), dtype=np.uint8)
        for name, seg in (("cpu", cpu_seg), ("neuron", dev_seg)):
            Image.fromarray(palette[np.clip(seg + 1, 0, top - 1)]).save(
                out / f"{args.task}_{name}.png"
            )
        # Where they disagree, in white.
        diff = (cpu_seg != dev_seg).astype(np.uint8) * 255
        Image.fromarray(diff).save(out / f"{args.task}_disagreement.png")
        print(f"\nwrote {out}/{args.task}_{{cpu,neuron,disagreement}}.png")

    # The port is usable if the segmentation is the same, not if the logits are close.
    raise SystemExit(0 if same_labels and agreement > 0.99 else 1)


if __name__ == "__main__":
    main()
