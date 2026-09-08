#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace OneFormer on CPU: the reference every later check is compared against.

Two jobs. First, prove the checkpoint works and produce something a human can look at
-- a panoptic map with named segments. Second, dump the raw head outputs
(`class_queries_logits`, `masks_queries_logits`) so the device port can be diffed
against them rather than against a picture.

The input size is **pinned**, which is the one thing this script does differently from
the model card's usage. OneFormer's processor defaults to shortest_edge 800 /
longest_edge 1333, i.e. a different shape per image, and an ahead-of-time compiler
needs one shape. 384x384 is the natural choice for this checkpoint: it is what Swin-L
was trained at, and it makes every stage's feature map (96, 48, 24, 12) divisible by
the window size 12, so no window padding is needed anywhere.

Usage:
    python contrib/oneformer-swin-l/check_hf_reference.py \\
        --model /mnt/nvme/models/oneformer_coco_swin_large \\
        --image /path/to/image.jpg --task panoptic --size 384 --out /tmp/of_ref
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme/models/oneformer_coco_swin_large")
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", default="panoptic",
                    choices=("panoptic", "semantic", "instance"))
    ap.add_argument("--size", type=int, default=384,
                    help="square input side; must be a multiple of 384 so every Swin "
                         "stage stays divisible by the window size")
    ap.add_argument("--out", default="/tmp/of_ref")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

    processor = OneFormerProcessor.from_pretrained(args.model)
    # Pin the shape. do_resize with a square size plus no size_divisor padding gives
    # exactly args.size x args.size, independent of the source aspect ratio.
    processor.image_processor.size = {"height": args.size, "width": args.size}
    processor.image_processor.do_resize = True

    start = time.perf_counter()
    model = OneFormerForUniversalSegmentation.from_pretrained(args.model)
    model.eval()
    print(f"[load] {time.perf_counter() - start:.1f} s", flush=True)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, task_inputs=[args.task], return_tensors="pt")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"[input] {k:16} {tuple(v.shape)} {v.dtype}")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    forward_s = time.perf_counter() - start
    print(f"[forward] {forward_s:.1f} s", flush=True)

    cls_logits = outputs.class_queries_logits  # (B, num_queries, num_classes + 1)
    mask_logits = outputs.masks_queries_logits  # (B, num_queries, H/4, W/4)
    print(f"[head] class_queries_logits {tuple(cls_logits.shape)}  "
          f"masks_queries_logits {tuple(mask_logits.shape)}")

    torch.save(
        {
            "task": args.task,
            "size": args.size,
            "pixel_values": inputs["pixel_values"],
            "task_inputs": inputs["task_inputs"],
            "class_queries_logits": cls_logits,
            "masks_queries_logits": mask_logits,
        },
        out / f"hf_{args.task}_{args.size}.pt",
    )

    # Something to look at, and something to compare semantically rather than numerically.
    post = getattr(processor, f"post_process_{args.task}_segmentation")
    result = post(outputs, target_sizes=[(image.height, image.width)])[0]
    seg = result["segmentation"] if isinstance(result, dict) else result
    seg = seg.cpu().numpy()

    id2label = model.config.id2label
    summary = []
    if isinstance(result, dict) and "segments_info" in result:
        for s in result["segments_info"]:
            label = id2label.get(s["label_id"], id2label.get(str(s["label_id"]), "?"))
            area = int((seg == s["id"]).sum())
            summary.append({"label": label, "score": round(float(s.get("score", 1.0)), 3),
                            "pixels": area})
        summary.sort(key=lambda s: -s["pixels"])
    else:
        ids, counts = np.unique(seg, return_counts=True)
        for i, count in sorted(zip(ids.tolist(), counts.tolist()), key=lambda p: -p[1]):
            label = id2label.get(i, id2label.get(str(i), "?"))
            summary.append({"label": label, "pixels": int(count)})

    print(f"\n[{args.task}] {len(summary)} segment(s), largest first:")
    for s in summary[:12]:
        score = f" score {s['score']}" if "score" in s else ""
        print(f"    {s['label']:24} {s['pixels']:>9} px{score}")

    (out / f"hf_{args.task}_{args.size}_summary.json").write_text(
        json.dumps(summary, indent=1)
    )

    # Colourize by segment id, deterministically, so two runs are comparable by eye.
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 235, size=(int(seg.max()) + 2, 3), dtype=np.uint8)
    rgb = palette[np.clip(seg + 1, 0, len(palette) - 1)]
    blended = (0.45 * np.asarray(image, dtype=np.float32)
               + 0.55 * rgb.astype(np.float32)).astype(np.uint8)
    Image.fromarray(rgb).save(out / f"hf_{args.task}_{args.size}_mask.png")
    Image.fromarray(blended).save(out / f"hf_{args.task}_{args.size}_overlay.png")
    print(f"\nwrote {out}/hf_{args.task}_{args.size}_{{mask,overlay}}.png and .pt")


if __name__ == "__main__":
    main()
