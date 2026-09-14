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
    ... segment.py --image a.jpg --size 480x864      # one rectangular bucket
    ... segment.py --image *.jpg --size auto         # a bucket per aspect ratio

``--size auto`` compiles one graph per bucket the run touches. That is one NEFF each
(50-59 MB, weights not included, so they share the one device copy) and a few minutes of
compile each; the point of :mod:`src.buckets` is that they should all take the same time to
*run*, because they all have the same area.
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
    ap.add_argument("--size", default="640", metavar="H[xW]",
                    help="input size, pinned for the compiler; each side a multiple of 32. "
                         "'640' means 640x640, the default test size; 'HxW' gives a "
                         "rectangle, e.g. 480x864 for 16:9 at the same area and so at "
                         "about the same latency. 'auto' picks the equal-area bucket "
                         "nearest each image's own aspect ratio (src/buckets.py), which "
                         "compiles one graph per bucket the run touches")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--compare-cpu", action="store_true",
                    help="also run the same model on CPU and diff the segmentation")
    ap.add_argument("--cpu-iterations", type=int, default=3)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--msda", default="torch", choices=("torch", "nki"),
                    help="which multi-scale deformable attention runs on device: our "
                         "PyTorch one (src/bilinear.py) or the NKI library kernel "
                         "(src/nki_msda.py). --compare-cpu always runs the PyTorch one "
                         "on the host")
    ap.add_argument("--gather", default="packed", choices=("corners", "packed"),
                    help="how src/bilinear.py fetches the 2x2 bilinear neighbourhood on "
                         "device: 'packed' is one gather into a 4x-wide table, 'corners' "
                         "four gathers at four indices. Bit-for-bit identical outputs; "
                         "packed is 466.7 ms against 614.8 at 640x640, and 143 against "
                         "230 at 384, because it issues 60%% fewer DMA packets for the "
                         "same bytes. Defaults to packed")
    ap.add_argument("--gather-split", type=int, default=1, metavar="N",
                    help="sample the deformable attention N groups of heads at a time "
                         "instead of all eight at once. Bit-for-bit identical at every N "
                         "-- nothing reduces across heads -- and it changes no packet's "
                         "size or count, only how much is live: at 640 the packed table "
                         "is 26.3 MiB and its gathered result 32.8 MiB against a 24 MiB "
                         "SBUF, and N=8 makes them 3.3 and 4.1. Measured and rejected: "
                         "451.77 ms at N=2 against 427.72 at N=1, and N=4 and N=8 run "
                         "neuronx-cc out of memory. Defaults to 1")
    ap.add_argument("--cross-attn", default="per_head", choices=("batched", "per_head"),
                    help="how the decoder's masked cross attention runs on device: "
                         "'per_head' loops over the eight heads, 'batched' is upstream's "
                         "nn.MultiheadAttention. Same arithmetic, not bit-for-bit (the "
                         "reduction over keys reassociates, 3.4e-08 on CPU); per_head "
                         "keeps one 3.66 MiB slice live where batched keeps two 29.3 MiB "
                         "tensors, against 24 MiB of SBUF. Defaults to per_head")
    ap.add_argument("--compiler-args", default="--optlevel=1",
                    help="passed verbatim to neuronx-cc. Defaults to --optlevel=1, the "
                         "fastest setting measured; pass '' for the compiler's own "
                         "default, or add '--auto-cast=all --auto-cast-type=bf16'")
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


def build(model_path, msda="torch", gather="corners", gather_split=1,
          cross_attn="per_head"):
    """Load, patch, and return (model, wrapper). See src/patches.py for the ten.

    No input size here: the patches are choices of implementation and are process-wide,
    while the size is a property of the *instance* and is set by ``select`` below. That
    split is what lets one model serve several buckets.
    """
    from src import patches

    patches.install(msda=msda, gather=gather, gather_split=gather_split,
                    cross_attn=cross_attn)

    from transformers import OneFormerForUniversalSegmentation

    model = OneFormerForUniversalSegmentation.from_pretrained(model_path).eval()
    patches.replace_gelu(model)
    patches.cache_position_embeddings(model)
    patches.relax_shape_assert()
    patches.patch_pixel_decoder_forward()
    patches.patch_prediction_heads()
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


def summarize(processor, config, cls_logits, mask_logits, task, target_size):
    from types import SimpleNamespace

    outputs = SimpleNamespace(
        class_queries_logits=cls_logits, masks_queries_logits=mask_logits
    )
    post = getattr(processor, f"post_process_{task}_segmentation")
    # The *image's* size, not the network's: the masks come out at a quarter of the input
    # and are upsampled here anyway, so upsampling them straight to the original costs
    # nothing extra and means the result can be used without a second resize.
    result = post(outputs, target_sizes=[target_size])[0]
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


def save_overlay(image, seg, path):
    rng = np.random.default_rng(0)
    top = int(seg.max()) + 2
    palette = rng.integers(40, 235, size=(top, 3), dtype=np.uint8)
    rgb = palette[np.clip(seg + 1, 0, top - 1)]
    base = np.asarray(image, dtype=np.float32)
    Image.fromarray((0.45 * base + 0.55 * rgb).astype(np.uint8)).save(path)


def main():
    args = parse_args()
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    from src import buckets

    from transformers import AutoConfig, OneFormerProcessor

    processor = OneFormerProcessor.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    auto = str(args.size).lower() == "auto"
    fixed = None if auto else buckets.parse_size(args.size)

    dtype = getattr(torch, args.dtype)
    # Two independent instances when comparing: one stays on the host, one moves to the
    # device. They cannot be the same object -- moving it would take the CPU side with it.
    dev_model, device_wrapper, patches = build(
        args.model, msda=args.msda, gather=args.gather,
        gather_split=args.gather_split, cross_attn=args.cross_attn,
    )
    cpu_model, cpu_wrapper = build(args.model)[:2] if args.compare_cpu else (None, None)

    # Cast *before* the host warmup pass, not after. The caches this fills are keyed by
    # the dtype they were asked for, and the device then looks them up under the dtype it
    # runs in -- so a float32 warmup followed by a bfloat16 device run misses the cache
    # and falls back to a host tensor inside the graph.
    #
    # Casting before the move is also required for a different reason: in one call it
    # fails, because OneFormer carries a float64 parameter it never uses in inference
    # (`criterion.logit_scale`, a scalar in the training loss) and casting f64 -> f32 as
    # part of the device transfer trips "Expected self.dtype() == dst.dtype()".
    device_wrapper = device_wrapper.to(dtype=dtype)

    # Which bucket each image lands in, decided up front: every size needs one *host*
    # forward to fill the position tables and the reference grid (building those on device
    # is what patches 4 and 5 exist to avoid), and doing them all before the model moves
    # means the 839 MB of weights cross to the device once instead of once per bucket.
    def size_for(path):
        with Image.open(path) as image:
            return buckets.pick(image.width, image.height) if auto else fixed

    sizes = {path: size_for(path) for path in args.image}

    def select(height, width):
        """Point both models at one bucket. Called outside any compiled region.

        The level shapes and the reference grid live on the instance, so switching size is
        just a re-pin -- nothing is rebuilt and nothing is recopied.
        """
        for model in (dev_model, cpu_model):
            if model is not None:
                patches.pin_input_size(model, buckets.level_shapes(height, width))

    for height, width in dict.fromkeys(sizes.values()):
        print(f"[setup] {buckets.describe(height, width)}")
        select(height, width)
        processor.image_processor.size = {"height": height, "width": width}
        warm = processor(images=Image.open(args.image[0]).convert("RGB"),
                         task_inputs=[args.task], return_tensors="pt")
        with torch.no_grad():
            device_wrapper(warm["pixel_values"].to(dtype), warm["task_inputs"],
                           torch.ones(1, height, width, dtype=dtype))

    device_wrapper = device_wrapper.to(device=DEVICE)
    print(f"[setup] copied {patches.move_position_cache(dev_model, DEVICE, dtype)} "
          f"position table(s) to the device, reference grid(s): "
          f"{patches.move_reference_cache(dev_model, DEVICE, dtype)}")

    compile_options = {"compiler_args": args.compiler_args} if args.compiler_args else {}
    if compile_options:
        print(f"[setup] neuronx-cc args: {args.compiler_args}")
    # One compiled callable serves every bucket: Dynamo keys its cache on the input
    # shapes, so a new size recompiles and an old one is still there. That guard is the
    # input tensor's own shape, which is the one thing that cannot disagree with the pin.
    compiled = torch.compile(
        device_wrapper, backend="neuron_libtorch", fullgraph=True,
        options=compile_options,
    )

    for path in args.image:
        image = Image.open(path).convert("RGB")
        name = Path(path).stem
        height, width = sizes[path]
        processor.image_processor.size = {"height": height, "width": width}
        print(f"\n=== {path}  ({image.width}x{image.height} -> {height}x{width}"
              f"{', auto' if auto else ''})")
        select(height, width)

        def preprocess():
            return processor(images=image, task_inputs=[args.task], return_tensors="pt")

        inputs = preprocess()
        pv = inputs["pixel_values"]
        ti = inputs["task_inputs"]
        pixel_mask = torch.ones(1, height, width)
        # task_inputs stays integral: they are token ids, not activations.
        dev_inputs = [pv.to(device=DEVICE, dtype=dtype), ti.to(DEVICE),
                      pixel_mask.to(device=DEVICE, dtype=dtype)]
        target_size = (image.height, image.width)

        def forward_device():
            with torch.no_grad():
                cls, msk = compiled(*dev_inputs)
            # .cpu() first: casting bf16 on the device fails with a dtype mismatch.
            return cls.cpu().float(), msk.cpu().float()

        dev_cls, dev_msk = forward_device()

        def postprocess():
            return summarize(processor, config, dev_cls, dev_msk, args.task, target_size)

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
                processor, config, cpu_cls, cpu_msk, args.task, target_size
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
                save_overlay(image, cpu_seg, out / f"{name}_{args.task}_cpu.png")
        if out:
            save_overlay(image, dev_seg, out / f"{name}_{args.task}_neuron.png")
            print(f"  wrote {out}/{name}_{args.task}_*.png")


if __name__ == "__main__":
    main()
