#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run the patched OneFormer on a NeuronCore and diff it against the CPU reference.

Everything cheap has already been done: ``probe_device_ops.py`` says which ops survive
the compiler, and ``check_patches_vs_hf.py`` says the two patches do not change the
model. So a difference here is the compiler's or the runtime's, which is the only
reason this script is worth its compile time.

It reads the ``.pt`` that ``check_hf_reference.py`` wrote, so the inputs and the
reference logits are exactly the ones a real image produced — no re-running HF, and no
chance of comparing against a differently preprocessed image.

Usage:
    python contrib/oneformer-swin-l/check_hf_reference.py --image ... --out /tmp/of_ref
    python contrib/oneformer-swin-l/run_device.py --ref /tmp/of_ref/hf_panoptic_384.pt
    ... run_device.py --ref ... --dtype bfloat16      # after fp32 agrees
    ... run_device.py --ref ... --module backbone     # bisect if the whole thing fails
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

import vllm_neuron  # noqa: F401,E402 — registers the "neuron_libtorch" Dynamo backend

DEVICE = torch.device("neuron", 0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme/models/oneformer_coco_swin_large")
    ap.add_argument("--ref", required=True, help=".pt written by check_hf_reference.py")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--module", default="full", choices=("full", "backbone", "pixel"),
                    help="what to compile: the whole model, Swin-L alone, or Swin-L "
                         "plus the pixel decoder. Bisecting in that order is how a "
                         "whole-model compiler failure gets localized cheaply")
    ap.add_argument("--fullgraph", action="store_true",
                    help="fail instead of breaking the graph; off by default so a "
                         "first run reports how many graphs it took")
    ap.add_argument("--save", default=None, help="write device logits here")
    ap.add_argument("--dump-dtypes", action="store_true",
                    help="trace with a no-op backend and report any float64 nodes, "
                         "which the compiler rejects outright ([NCC_ESPP004])")
    return ap.parse_args()


class Heads(nn.Module):
    """The two tensors that matter, as a plain tuple.

    Dynamo can trace through the dataclass HuggingFace returns, but a module boundary
    that returns tensors keeps the compiled region obvious and the comparison honest.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, task_inputs, pixel_mask):
        # pixel_mask is passed rather than left to default. Upstream would create it
        # with torch.ones(...) inside the forward, and Dynamo isolates that line into a
        # subgraph of its own whose only value is unused there — which the backend
        # rejects outright ("Cannot compile a module that has no output"). At a pinned
        # square input the mask is all ones either way, so this changes nothing but the
        # graph partitioning.
        out = self.model(
            pixel_values=pixel_values,
            task_inputs=task_inputs,
            pixel_mask=pixel_mask,
        )
        return out.class_queries_logits, out.masks_queries_logits


class PixelLevel(nn.Module):
    """Swin-L plus the pixel decoder: mask features and the three multi-scale maps."""

    def __init__(self, model):
        super().__init__()
        self.pixel_level_module = model.model.pixel_level_module

    def forward(self, pixel_values):
        out = self.pixel_level_module(pixel_values)
        # decoder_last_feature is the mask features; decoder_features are the three
        # multi-scale maps the transformer decoder attends over.
        return (out.decoder_last_feature, *out.decoder_features)


class Backbone(nn.Module):
    """Swin-L only: the four feature maps it hands to the pixel decoder."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.pixel_level_module.encoder

    def forward(self, pixel_values):
        return tuple(self.encoder(pixel_values).feature_maps)


def compare(name, got, ref, dtype):
    got = got.float().cpu()
    ref = ref.float().cpu()
    scale = max(ref.abs().max().item(), 1e-6)
    rel = (got - ref).abs().max().item() / scale
    mean = (got - ref).abs().mean().item() / scale
    # bf16 has ~3 decimal digits, so hold it to a different bar than fp32.
    tol = 2e-3 if dtype == torch.float32 else 5e-2
    verdict = "PASS" if rel <= tol else "FAIL"
    print(f"  {verdict}  {name:24} rel={rel:.3e}  mean={mean:.3e}  "
          f"(tolerance {tol:g})")
    return verdict == "PASS"


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    ref = torch.load(args.ref, weights_only=False)
    size = ref["size"]
    level_shapes = [(size // s, size // s) for s in (32, 16, 8)]

    from src import patches

    patches.install(level_shapes)
    print(f"patched for {size}x{size}, deformable levels {level_shapes}, {args.dtype}\n")

    from transformers import OneFormerForUniversalSegmentation

    model = OneFormerForUniversalSegmentation.from_pretrained(args.model).eval()
    # F.gelu / nn.GELU do not compile here (apply() takes no keyword arguments), so the
    # exact erf form goes in instead. See src/patches.py.
    print(f"replaced {patches.replace_gelu(model)} GELU activation(s) with the erf form")
    # The sine position tables are constants at a pinned input size, and building them
    # is what the compiler rejects ([NCC_IBIR243]). Cache them on the host instead.
    print(f"cached {patches.cache_position_embeddings(model)} sine position embedding(s)")
    # Same story for the pixel decoder's reference-point grid: constant at a pinned
    # size, and its meshgrid+reshape is rejected on device (not contiguous).
    patches.cache_reference_points(model, level_shapes)
    # Upstream's tensor-valued shape assert becomes an unbacked symbol under Dynamo;
    # keep it on the host, where it still catches level-shape mistakes.
    patches.relax_shape_assert()
    patches.constant_fold_pixel_decoder_split(level_shapes)

    builders = {"full": Heads, "backbone": Backbone, "pixel": PixelLevel}
    wrapper = builders[args.module](model).to(dtype)
    inputs = [ref["pixel_values"].to(dtype)]
    if args.module == "full":
        inputs.append(ref["task_inputs"])
        # float32 ones, which is exactly what upstream's default would build.
        inputs.append(torch.ones(1, size, size))

    print("[cpu] reference forward in this dtype (so the diff is compiler-only)")
    with torch.no_grad():
        cpu_out = wrapper(*inputs)
    if args.module == "full":
        # Sanity: this dtype on CPU must still match the fp32 reference the .pt holds.
        compare("cpu class vs ref", cpu_out[0], ref["class_queries_logits"], dtype)
        compare("cpu mask vs ref", cpu_out[1], ref["masks_queries_logits"], dtype)

    print(f"\n[device] compiling {args.module} "
          f"(fullgraph={args.fullgraph}) — this is the expensive part")
    wrapper = wrapper.to(DEVICE)
    # The CPU pass above filled the position caches with host tensors; they have to
    # follow the model, or the backend refuses the graph for mixing devices.
    print(f"[device] moved {patches.move_position_cache(model, DEVICE, dtype)} "
          f"cached position table(s), reference grid moved: "
          f"{patches.move_reference_cache(DEVICE, dtype)}")
    compiled = torch.compile(wrapper, backend="neuron_libtorch", fullgraph=args.fullgraph)

    if args.dump_dtypes:
        # Trace only: capture the graph Dynamo would hand the backend, then look at
        # what dtypes it contains. f64 anywhere is fatal for neuronx-cc, and the node's
        # stack trace says which line produced it.
        captured = []

        class _Captured(Exception):
            """Stop after tracing: the graph cannot be *run* eagerly on device."""

        def spy(gm, example_inputs):
            captured.append(gm)
            raise _Captured

        traced = torch.compile(wrapper, backend=spy, fullgraph=args.fullgraph)
        try:
            with torch.no_grad():
                traced(*[x.to(DEVICE) for x in inputs])
        except Exception:  # noqa: BLE001 — Dynamo wraps the sentinel
            if not captured:
                raise
        f64 = []
        for gm in captured:
            for node in gm.graph.nodes:
                value = node.meta.get("val")
                values = ([value] if isinstance(value, torch.Tensor)
                          else list(value) if isinstance(value, (list, tuple)) else [])
                for tensor in values:
                    if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float64:
                        f64.append((node, tuple(tensor.shape)))
        print(f"[dtypes] {len(captured)} graph(s), {len(f64)} float64 node(s)")
        for node, shape in f64[:6]:
            print(f"          {node.op} {str(node.target)[:60]} {shape}")
            trace = node.meta.get("stack_trace") or ""
            for line in [ln.strip() for ln in trace.splitlines() if ", line " in ln][-2:]:
                print(f"            {line[:140]}")
        raise SystemExit(0 if not f64 else 1)

    start = time.perf_counter()
    with torch.no_grad():
        device_out = compiled(*[x.to(DEVICE) for x in inputs])
    device_out = tuple(t.float().cpu() for t in device_out)
    print(f"[device] first call (compile included): {time.perf_counter() - start:.0f} s")

    from torch._dynamo.utils import counters

    graphs = counters["stats"].get("unique_graphs", 0)
    breaks = sum(counters["graph_break"].values())
    print(f"[device] {graphs} unique graph(s), {breaks} graph break(s)")
    if breaks:
        for reason, count in sorted(counters["graph_break"].items(), key=lambda p: -p[1])[:5]:
            print(f"          {count:4}x {reason[:110]}")

    start = time.perf_counter()
    with torch.no_grad():
        compiled(*[x.to(DEVICE) for x in inputs])
    print(f"[device] warm call: {(time.perf_counter() - start) * 1e3:.0f} ms")

    print("\n[compare] device vs CPU, same dtype, same input")
    ok = True
    if args.module == "full":
        ok &= compare("class_queries_logits", device_out[0], cpu_out[0], dtype)
        ok &= compare("masks_queries_logits", device_out[1], cpu_out[1], dtype)
        same = torch.equal(device_out[0].argmax(-1), cpu_out[0].float().argmax(-1))
        print(f"  {'PASS' if same else 'FAIL'}  per-query argmax label unchanged")
        ok &= same
    else:
        for i, (got, exp) in enumerate(zip(device_out, cpu_out)):
            ok &= compare(f"feature_map[{i}] {tuple(exp.shape)}", got, exp, dtype)

    if args.save:
        torch.save({"class_queries_logits": device_out[0],
                    "masks_queries_logits": device_out[1] if len(device_out) > 1 else None},
                   args.save)
        print(f"\nwrote {args.save}")

    print()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
