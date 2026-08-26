#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Time the *reference* (NxDI) vision path, to separate its encoder from its glue.

This probes the other implementation, not ours. It exists because the end-to-end
comparison in HANDOFF showed this port 2.5x faster at 1024x1024 and that begged the
question of where the reference actually spends the time — encoder, or the Python
that surrounds it.

Two nested levels, median of 5 after a warmup:

  L1  the raw traced graph, ``_TPVisionAdapter.__call__`` on pre-built inputs
  L2  ``wrapper.forward``, i.e. L1 plus CPU patch-embed / RoPE / mask / pad / merge

L3 is the full VL TTFT, which ``run_vl_benchmark.py`` already reports. Measured at
TP=4 on trn2.3xlarge, L1 was 101.1 ms at 4096 patches — matching the reference
README's "101 ms standalone" — against an L3 of 575.1 ms, so two thirds of its TTFT
is glue. See HANDOFF, "One graph or two".

Prerequisites: the reference cloned at /mnt/nvme/nxdi_ref, its TP=4 vision encoder
compiled to /tmp/nxdi_vl/vision_tp4 by ``compile_vision_encoder_tp.py``, and images
staged at /tmp/test_image_<size>.jpg.

Run it with the *NxDI* venv, not the vLLM one, and put that venv's bin on PATH
(``libneuronpjrt-path`` is an executable there):

    N=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
    PATH=$N/bin:/opt/aws/neuron/bin:$PATH VIRTUAL_ENV=$N \
    NEURON_SKIP_EFA_AFFINITY=1 $N/bin/python probe_nxdi_vision.py
"""
import os, sys, json, time, statistics
os.environ.setdefault("QWEN36_DELTANET_CTE_IMPL", "legacy_direct")
os.environ.setdefault("QWEN36_DELTANET_MULTIHEAD_CTE", "0")
sys.path.insert(0, os.environ.get(
    "NXDI_QWEN35_SRC", "/mnt/nvme/nxdi_ref/contrib/models/Qwen3.5-2B"))

import torch
from PIL import Image
from transformers import AutoProcessor
from src.modeling_qwen35_vision import NeuronQwen35VisionModelWrapper
from types import SimpleNamespace

MODEL = "/mnt/nvme/models/Qwen3.5-2B"
VE_DIR = "/tmp/nxdi_vl/vision_tp4"
REPEATS = 5

# transformers 4.57.6 does not register `qwen3_5`; read the json as they do.
full = json.load(open(os.path.join(MODEL, "config.json")))
vconf = full["vision_config"]
vconf.setdefault("spatial_merge_size", 2)
vconf.setdefault("temporal_patch_size", 2)
cfg = SimpleNamespace(**vconf)

wrapper = NeuronQwen35VisionModelWrapper(config=cfg, model_cls=None)
wrapper.load_compiled(VE_DIR)
wrapper.load_vision_weights_from_hf(MODEL)
print("loaded buckets:", sorted(wrapper._compiled_buckets.keys()), flush=True)

proc = AutoProcessor.from_pretrained(MODEL)

def med(fn, n=REPEATS):
    fn()                                   # warm
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)

out = {}
for size in (512, 1024):
    img = Image.open(f"/tmp/test_image_{size}.jpg").convert("RGB")
    enc = proc.image_processor(images=[img], return_tensors="pt")
    pv, gthw = enc["pixel_values"], enc["image_grid_thw"]
    raw = int(gthw[0].prod())
    l2 = med(lambda: wrapper.forward(pv, gthw))

    # L1: rebuild exactly what forward() hands the graph, then time only the call.
    bucket = min(b for b in sorted(wrapper._compiled_buckets) if b >= raw)
    model = wrapper._compiled_buckets[bucket]
    H = cfg.hidden_size
    hd = H // cfg.num_heads
    hs = torch.zeros((bucket, H), dtype=torch.bfloat16)
    mask = torch.zeros((1, 1, bucket, bucket), dtype=torch.bfloat16)
    cos = torch.zeros((bucket, hd), dtype=torch.bfloat16)
    sin = torch.zeros((bucket, hd), dtype=torch.bfloat16)
    l1 = med(lambda: model(hs, mask, cos, sin))

    out[size] = dict(raw=raw, bucket=bucket, l1_graph=l1, l2_wrapper=l2)
    print(f"{size}x{size}: raw={raw} bucket={bucket}  "
          f"L1 graph {l1:.1f} ms   L2 wrapper {l2:.1f} ms   "
          f"prep {l2-l1:.1f} ms", flush=True)

json.dump(out, open("/tmp/nxdi_ve_probe.json", "w"), indent=1)
