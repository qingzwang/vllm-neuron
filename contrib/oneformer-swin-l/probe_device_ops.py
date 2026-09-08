#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Op-level device probes for OneFormer Swin-L on Trainium.

Compiling the whole model to find out what the compiler dislikes costs many minutes
per attempt and only ever answers "something is wrong". These probes compile one
function at a time at the shapes OneFormer actually uses, and diff the result against
CPU:

    OK     compiles and matches CPU
    WRONG  compiles but disagrees  (the dangerous one -- silent)
    FAIL   does not compile

The list is not arbitrary. Each entry is something in OneFormer that a
static-shape ahead-of-time compiler has a reason to struggle with:

  grid_sample          the core of multi-scale deformable attention: a gather at
                       *computed float* coordinates with bilinear weights
  ms_deform_attn       HF's own pure-PyTorch deformable attention, end to end
  window_partition     Swin's rank-6 view+permute, and its inverse
  roll                 Swin's shifted windows
  interpolate          mask logits upsampled to image resolution
  masked_fill_neg_inf  the transformer decoder's masked cross-attention
  softmax_neg_inf      a fully-masked row, which OneFormer can produce

Compilation goes through ``torch.compile(backend="neuron_libtorch")``, the backend the
vllm-neuron plugin registers with Dynamo. Importing ``vllm_neuron`` is the *only* thing
this model takes from the plugin -- registering that backend. Nothing here touches its
model registry, runner or executor, because OneFormer is not an LLM: it has no KV
cache, no token loop, and one fixed-shape forward per image.

Usage (in the plugin's venv, with its bin on PATH so neuronx-cc is found):

    V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
    PATH=$V/bin:$PATH $V/bin/python contrib/oneformer-swin-l/probe_device_ops.py
    ... probe_device_ops.py grid_sample interpolate      # or a subset
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

import vllm_neuron  # noqa: F401 — registers the "neuron_libtorch" Dynamo backend

DEVICE = torch.device("neuron", 0)
DEVICE_DTYPE = torch.float32  # probe in fp32 so a mismatch means logic, not rounding
TOL = 2e-3


# --------------------------------------------------------------------------- probes
def probe_grid_sample():
    """Bilinear sampling at computed coordinates — MSDeformAttn's inner loop."""
    value = torch.rand(8, 32, 96, 96, dtype=DEVICE_DTYPE)          # (B*heads, C, H, W)
    grid = torch.rand(8, 150, 4, 2, dtype=DEVICE_DTYPE) * 2 - 1     # (B*heads, Q, P, 2)

    def fn(value, grid):
        return F.grid_sample(
            value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    return fn, (value, grid)


def probe_ms_deform_attn():
    """HF's multi_scale_deformable_attention, at OneFormer's pixel-decoder shapes."""
    from transformers.models.oneformer import modeling_oneformer as m

    fn_impl = getattr(m, "multi_scale_deformable_attention", None)
    if fn_impl is None:
        return None, None  # older/newer transformers moved it; skip rather than lie

    batch, heads, dim, queries, levels, points = 1, 8, 32, 150, 3, 4
    shapes = torch.tensor([[96, 96], [48, 48], [24, 24]])
    total = int((shapes[:, 0] * shapes[:, 1]).sum())
    value = torch.rand(batch, total, heads, dim, dtype=DEVICE_DTYPE)
    locations = torch.rand(batch, queries, heads, levels, points, 2, dtype=DEVICE_DTYPE)
    weights = torch.rand(batch, queries, heads, levels, points, dtype=DEVICE_DTYPE)
    weights = weights / weights.sum(-1, keepdim=True)

    # spatial_shapes goes in as a *Python list*, not a captured CPU tensor: a captured
    # CPU tensor makes the backend refuse the graph ("tensors not on neuron"), and the
    # level sizes have to be compile-time constants anyway.
    shapes_list = [(int(h), int(w)) for h, w in shapes.tolist()]

    def fn(value, locations, weights):
        try:
            return fn_impl(value, shapes_list, locations, weights)
        except TypeError:
            return fn_impl(value, shapes, shapes_list, locations, weights)

    return fn, (value, locations, weights)


def probe_window_partition():
    """Swin's window partition and its inverse: rank-6 view + permute."""
    window = 12
    x = torch.rand(1, 96, 96, 192, dtype=DEVICE_DTYPE)  # (B, H, W, C) after patch embed

    def fn(x):
        b, h, w, c = x.shape
        windows = x.view(b, h // window, window, w // window, window, c)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window, window, c)
        # ... and back, which is what window_reverse does
        back = windows.view(b, h // window, w // window, window, window, c)
        back = back.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, c)
        return windows.sum(), back

    return fn, (x,)


def probe_roll():
    """Shifted windows."""
    x = torch.rand(1, 96, 96, 192, dtype=DEVICE_DTYPE)
    shift = -6

    def fn(x):
        return torch.roll(x, shifts=(shift, shift), dims=(1, 2))

    return fn, (x,)


def probe_interpolate():
    """Mask logits upsampled from 1/4 resolution to the image."""
    x = torch.rand(1, 150, 96, 96, dtype=DEVICE_DTYPE)

    def fn(x):
        return F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)

    return fn, (x,)


def probe_masked_fill_neg_inf():
    """The decoder's masked cross-attention bias."""
    scores = torch.rand(8, 150, 9216, dtype=DEVICE_DTYPE)
    mask = torch.rand(8, 150, 9216) > 0.5

    def fn(scores, mask):
        filled = scores.masked_fill(mask, float("-inf"))
        return torch.softmax(filled, dim=-1)

    return fn, (scores, mask)


def probe_softmax_all_masked():
    """A query whose mask excludes everything — OneFormer guards against this.

    Upstream replaces a fully-masked row with "attend to everything"; the point of
    probing it is to see whether the device path produces NaN where CPU does too, so
    the guard can be written once and trusted.
    """
    scores = torch.rand(2, 4, 16, dtype=DEVICE_DTYPE)
    mask = torch.ones(2, 4, 16, dtype=torch.bool)  # everything masked

    def fn(scores, mask):
        filled = scores.masked_fill(mask, float("-inf"))
        out = torch.softmax(filled, dim=-1)
        return torch.nan_to_num(out, nan=0.0), out.isnan().any().to(scores.dtype)

    return fn, (scores, mask)


def probe_bilinear_sample():
    """Our grid_sample replacement, against CPU F.grid_sample as the reference."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
    from bilinear import bilinear_sample

    value = torch.rand(8, 32, 96, 96, dtype=DEVICE_DTYPE)
    # Deliberately includes out-of-range coordinates, which is where zeros padding and
    # index clamping have to agree with upstream.
    grid = torch.rand(8, 150, 4, 2, dtype=DEVICE_DTYPE) * 2.4 - 1.2

    reference = F.grid_sample(
        value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    def fn(value, grid):
        return bilinear_sample(value, grid)

    # Check the replacement against grid_sample on CPU first: if that disagrees the
    # device number is meaningless.
    cpu_rel = ((fn(value, grid) - reference).abs().max() / reference.abs().max()).item()
    print(f"       (vs CPU F.grid_sample, on CPU: rel={cpu_rel:.2e})")
    return fn, (value, grid)


def probe_ms_deform_attn_ours():
    """Our whole multi-scale deformable attention, at the pixel decoder's shapes."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
    from bilinear import multi_scale_deformable_attention

    batch, heads, dim, queries, points = 1, 8, 32, 150, 4
    shapes = [(96, 96), (48, 48), (24, 24)]
    total = sum(h * w for h, w in shapes)
    value = torch.rand(batch, total, heads, dim, dtype=DEVICE_DTYPE)
    locations = torch.rand(batch, queries, heads, len(shapes), points, 2, dtype=DEVICE_DTYPE)
    weights = torch.rand(batch, queries, heads, len(shapes), points, dtype=DEVICE_DTYPE)
    weights = weights / weights.sum(-1, keepdim=True)

    def fn(value, locations, weights):
        return multi_scale_deformable_attention(value, shapes, locations, weights)

    return fn, (value, locations, weights)


PROBES = {
    "grid_sample": probe_grid_sample,
    "bilinear_sample": probe_bilinear_sample,
    "ms_deform_attn_ours": probe_ms_deform_attn_ours,
    "ms_deform_attn": probe_ms_deform_attn,
    "window_partition": probe_window_partition,
    "roll": probe_roll,
    "interpolate": probe_interpolate,
    "masked_fill_neg_inf": probe_masked_fill_neg_inf,
    "softmax_all_masked": probe_softmax_all_masked,
}


# --------------------------------------------------------------------------- driver
def flat(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for o in out for t in flat(o)]
    return []


def run_in_process(name):
    """Compile and compare one probe. Returns a verdict string; may abort the process.

    An unsupported op does not always raise: the runtime can abort with
    ``Check failed: pjrt_data->buffer != nullptr PjRt buffer is null in
    TransferFromDevice`` when a graph produced no real output. That is a SIGABRT, not
    an exception, which is why :func:`main` runs every probe in a subprocess.
    """
    build = PROBES[name]
    fn, inputs = build()
    if fn is None:
        print("SKIP   not available in this transformers version")
        return "SKIP"

    with torch.no_grad():
        expected = fn(*inputs)

    # fullgraph=True on purpose: a graph break here means part of the op would run on
    # the host in the real model, which is worth knowing now rather than later.
    compiled = torch.compile(fn, backend="neuron_libtorch", fullgraph=True)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            got = compiled(*[x.to(DEVICE) for x in inputs])
        got = [t.float().cpu() for t in flat(got)]
    except Exception as exc:  # noqa: BLE001 — the point is to report, not to handle
        msg = str(exc).replace("\n", " ")[:200]
        print(f"FAIL   {type(exc).__name__}: {msg}")
        return "FAIL"
    compile_s = time.perf_counter() - start

    worst = 0.0
    for a, b in zip(got, flat(expected)):
        b32 = b.float().cpu()
        scale = max(b32.abs().max().item(), 1e-6)
        worst = max(worst, (a - b32).abs().max().item() / scale)

    verdict = "OK" if worst <= TOL else "WRONG"
    print(f"{verdict:6} rel={worst:.2e}  (compiled and ran in {compile_s:.0f} s)")
    return verdict


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--one":
        verdict = run_in_process(sys.argv[2])
        raise SystemExit(0 if verdict in ("OK", "SKIP") else 3)

    wanted = sys.argv[1:] or list(PROBES)
    unknown = [w for w in wanted if w not in PROBES]
    if unknown:
        raise SystemExit(f"unknown probe(s): {unknown}. Known: {list(PROBES)}")

    print(f"probing {len(wanted)} op(s) in {DEVICE_DTYPE}, tolerance {TOL:g} relative")
    print("(each probe runs in its own process: an unsupported op can abort the "
          "runtime rather than raise)\n")

    results = {}
    for name in wanted:
        proc = subprocess.run(
            [sys.executable, __file__, "--one", name],
            capture_output=True, text=True,
        )
        tail = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        verdict_line = tail[-1] if tail else ""
        verdict = verdict_line.split()[0] if verdict_line[:1].isalpha() else ""
        if verdict not in ("OK", "WRONG", "FAIL", "SKIP"):
            # aborted: no verdict printed. Pull the runtime's complaint out of stderr.
            reason = ""
            for line in proc.stderr.splitlines():
                if "Check failed" in line or "what():" in line:
                    reason = line.split("]")[-1].strip()[:120]
                    break
            verdict = "ABORT"
            verdict_line = f"ABORT  exit {proc.returncode}: {reason or 'no output'}"
        print(f"  {name:22} {verdict_line}")
        results[name] = verdict

    print()
    for bad in ("ABORT", "FAIL", "WRONG"):
        names = [n for n, v in results.items() if v == bad]
        if names:
            print(f"{bad}: {', '.join(names)}")
    if all(v in ("OK", "SKIP") for v in results.values()):
        print("every probe compiled and matched CPU")


if __name__ == "__main__":
    main()
