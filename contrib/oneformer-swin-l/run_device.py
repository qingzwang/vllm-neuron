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
    ap.add_argument("--module", default="full",
                    choices=("full", "backbone", "pixel", "decoder", "layers",
                             "query"),
                    help="what to compile: the whole model, Swin-L alone, Swin-L plus "
                         "the pixel decoder, or the transformer half on its own. "
                         "Bisecting in that order is how a whole-model compiler "
                         "failure gets localized cheaply")
    ap.add_argument("--fullgraph", action="store_true",
                    help="fail instead of breaking the graph; off by default so a "
                         "first run reports how many graphs it took")
    ap.add_argument("--save", default=None, help="write device logits here")
    ap.add_argument("--decoder-layers", type=int, default=0,
                    help="with --module layers: how many masked-attention layers to "
                         "keep. 0 leaves the query transformer and the prediction "
                         "heads, which is the cut that says whether the layers matter")
    ap.add_argument("--msda", default="torch", choices=("torch", "nki"),
                    help="which multi-scale deformable attention runs on device: our "
                         "PyTorch one (src/bilinear.py) or the NKI library kernel "
                         "(src/nki_msda.py). The CPU side of the comparison is always "
                         "the PyTorch one, so 'nki' makes this a kernel-vs-PyTorch diff")
    ap.add_argument("--gather", default="packed", choices=("corners", "packed"),
                    help="how src/bilinear.py fetches the 2x2 bilinear neighbourhood on "
                         "device: 'packed' is one gather into a 4x-wide table, 'corners' "
                         "four gathers at four indices. Bit-for-bit identical outputs; "
                         "packed is 143 ms against 230 because it issues 64%% fewer DMA "
                         "packets. Defaults to packed")
    ap.add_argument("--compiler-args", default="--optlevel=1",
                    help="passed verbatim to neuronx-cc. Defaults to --optlevel=1, which "
                         "measured fastest (237.9 ms against 241.0 at the compiler's "
                         "default), compiles 3x quicker and is slightly more accurate; "
                         "pass '' for the compiler's own default. Add "
                         "'--auto-cast=all --auto-cast-type=bf16' for bf16 arithmetic. "
                         "Part of the compile-cache key, so a setting cannot collide "
                         "with a previous build")
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


class QueryTransformer(nn.Module):
    """Just ``transformer_module.decoder.query_transformer``.

    OneFormer builds its object queries with a second, DETR-style decoder conditioned on
    the task token. It is separate code from the masked-attention layers, so it gets its
    own bisect step. Its inputs are *captured* from a real CPU forward rather than
    reconstructed, so this harness cannot drift from how the model actually calls it.
    """

    def __init__(self, model):
        super().__init__()
        self.query_transformer = model.model.transformer_module.decoder.query_transformer

    def forward(self, src, query_embed, pos_embed, task_token):
        out = self.query_transformer(src, None, query_embed, pos_embed, task_token)
        return (out[0],)


def capture_query_transformer_inputs(model, pixel_out, task_token):
    """Run the transformer module on CPU and record what query_transformer receives."""
    captured = {}
    target = model.model.transformer_module.decoder.query_transformer

    def hook(_module, args):
        # (src, mask, query_embed, pos_embed, task_token)
        captured["args"] = args
        return None

    handle = target.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model.model.transformer_module(
                multi_scale_features=list(pixel_out.decoder_features),
                mask_features=pixel_out.decoder_last_feature,
                task_token=task_token,
            )
    finally:
        handle.remove()
    src, _mask, query_embed, pos_embed, captured_task = captured["args"]
    return [src, query_embed, pos_embed,
            captured_task if captured_task is not None else task_token]


class DecoderLayers(nn.Module):
    """The masked-attention layers only, truncated to ``layers`` of them.

    Truncating is the cheap cut between "one of the nine layers does it" and "the
    prediction heads or the query transformer does it": each extra layer is the same
    code again, so if zero layers already fails the layers are innocent.
    """

    def __init__(self, model, layers):
        super().__init__()
        self.model = model.model
        kept = self.model.transformer_module.decoder.layers[:layers]
        self.model.transformer_module.decoder.layers = nn.ModuleList(kept)

    def forward(self, mask_features, level0, level1, level2, task_token):
        out = self.model.transformer_module(
            multi_scale_features=[level0, level1, level2],
            mask_features=mask_features,
            task_token=task_token,
        )
        return out.prediction_class[-1], out.prediction_masks[-1]


class Decoder(nn.Module):
    """The other half: the transformer module, fed the pixel decoder's outputs.

    Splitting the model here is what localizes a whole-model compiler failure: the
    pixel half compiles, so anything left is in these ten layers, the task MLP and the
    prediction heads. The inputs come from a CPU forward of the pixel half, so this
    harness needs no device work of its own to set up.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def forward(self, mask_features, level0, level1, level2, task_token):
        out = self.model.transformer_module(
            multi_scale_features=[level0, level1, level2],
            mask_features=mask_features,
            task_token=task_token,
        )
        # The transformer module names these prediction_class / prediction_masks; the
        # top-level model renames the last entry of each to the *_queries_logits it
        # returns, which is what the reference .pt holds.
        return out.prediction_class[-1], out.prediction_masks[-1]


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
    got = got.cpu().float()
    ref = ref.cpu().float()
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

    patches.install(level_shapes, msda=args.msda, gather=args.gather)
    print(f"patched for {size}x{size}, deformable levels {level_shapes}, {args.dtype}, "
          f"device deformable attention: {args.msda}/{args.gather}\n")

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
    # Tensor-valued split/view sizes, plus the FPN's 2x resize as shifts and adds.
    patches.patch_pixel_decoder_forward(level_shapes)
    # `mask_logits < 0.5` promotes 0.5 to f64 in the lowering ([NCC_ESPP004]), and the
    # mask downsample is a 9216x144 matmul unless it is written out.
    patches.patch_prediction_heads()

    builders = {"full": Heads, "backbone": Backbone, "pixel": PixelLevel,
                "decoder": Decoder}
    if args.module == "query":
        with torch.no_grad():
            pixel_out = model.model.pixel_level_module(ref["pixel_values"].to(dtype))
            task_token = model.model.task_encoder(ref["task_inputs"].to(dtype))
        wrapper = QueryTransformer(model).to(dtype)
        inputs = capture_query_transformer_inputs(model, pixel_out, task_token)
        print("[query] captured inputs: "
              + ", ".join(str(tuple(x.shape)) for x in inputs))
    elif args.module == "layers":
        wrapper = DecoderLayers(model, args.decoder_layers).to(dtype)
        print(f"[layers] kept {args.decoder_layers} masked-attention layer(s)")
    else:
        wrapper = builders[args.module](model).to(dtype)
    if args.module != "query":
        inputs = [ref["pixel_values"].to(dtype)]
    if args.module == "full":
        inputs.append(ref["task_inputs"])
        # float32 ones, which is exactly what upstream's default would build.
        inputs.append(torch.ones(1, size, size))
    elif args.module in ("decoder", "layers"):
        # Run the pixel half on CPU once to get this half's real inputs.
        with torch.no_grad():
            pixel_out = model.model.pixel_level_module(ref["pixel_values"].to(dtype))
            task_token = model.model.task_encoder(ref["task_inputs"].to(dtype))
        inputs = [pixel_out.decoder_last_feature, *pixel_out.decoder_features, task_token]
        print(f"[decoder] inputs: mask_features {tuple(inputs[0].shape)}, "
              f"levels {[tuple(x.shape) for x in inputs[1:4]]}, "
              f"task_token {tuple(inputs[4].shape)}")

    print("[cpu] reference forward in this dtype (so the diff is compiler-only)")
    with torch.no_grad():
        cpu_out = wrapper(*inputs)
    if args.module in ("full", "decoder"):
        # Sanity: this dtype on CPU must still match the fp32 reference the .pt holds.
        # (Not for --module layers: a truncated decoder is a different model.)
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
    compile_options = {"compiler_args": args.compiler_args} if args.compiler_args else {}
    if compile_options:
        print(f"[device] neuronx-cc args: {args.compiler_args}")
    compiled = torch.compile(
        wrapper, backend="neuron_libtorch", fullgraph=args.fullgraph,
        options=compile_options,
    )

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
    # .cpu() before .float(): casting a bf16 tensor while it is still on the device
    # fails with "Expected self.dtype() == dst.dtype()".
    device_out = tuple(t.cpu().float() for t in device_out)
    print(f"[device] first call (compile included): {time.perf_counter() - start:.0f} s")

    from torch._dynamo.utils import counters

    graphs = counters["stats"].get("unique_graphs", 0)
    breaks = sum(counters["graph_break"].values())
    print(f"[device] {graphs} unique graph(s), {breaks} graph break(s)")
    if breaks:
        for reason, count in sorted(counters["graph_break"].items(), key=lambda p: -p[1])[:5]:
            print(f"          {count:4}x {reason[:110]}")

    # Read one element back, or this measures *dispatch* and nothing else: Neuron
    # execution is asynchronous, so timing a call whose outputs are never touched
    # reports single-digit milliseconds for any graph, however large. That mistake is
    # what made an early version of this script claim a 3 ms forward.
    start = time.perf_counter()
    with torch.no_grad():
        warm = compiled(*[x.to(DEVICE) for x in inputs])
        float(warm[0].reshape(-1)[0].cpu())
    print(f"[device] warm call (synced): {(time.perf_counter() - start) * 1e3:.0f} ms")

    print("\n[compare] device vs CPU, same dtype, same input")
    ok = True
    if args.module in ("full", "decoder", "layers"):
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
