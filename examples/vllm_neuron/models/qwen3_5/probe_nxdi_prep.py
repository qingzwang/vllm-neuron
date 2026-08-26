#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Break the reference's per-call CPU vision prep into components.

Companion to ``probe_nxdi_vision.py``, which showed the reference spends 119 ms of
CPU per request between the HF processor and its traced vision graph. This says
where: at 1024x1024 it is ``patch_embed`` at 86 ms — a 4096x1536 -> 1024 matmul run
on the host — plus ~25 ms of grid-only work that could simply be cached. The first
guess (the 4096x4096 attention mask) turned out to cost 9.8 ms, not the bulk.

Matters because it decides whether the reference's slower end-to-end TTFT is a
model property or an integration one: it is the latter. See HANDOFF, "Could the
reference remove its glue?".

Same prerequisites and venv as ``probe_nxdi_vision.py``.
"""
import os, sys, json, time, statistics
os.environ.setdefault("QWEN36_DELTANET_CTE_IMPL","legacy_direct")
sys.path.insert(0,"/mnt/nvme/nxdi_ref/contrib/models/Qwen3.5-2B")
import torch
from PIL import Image
from transformers import AutoProcessor
from src.modeling_qwen35_vision import NeuronQwen35VisionModelWrapper
from types import SimpleNamespace
MODEL="/mnt/nvme/models/Qwen3.5-2B"
full=json.load(open(os.path.join(MODEL,"config.json"))); vc=full["vision_config"]
vc.setdefault("spatial_merge_size",2); vc.setdefault("temporal_patch_size",2)
w=NeuronQwen35VisionModelWrapper(config=SimpleNamespace(**vc), model_cls=None)
w.load_compiled("/tmp/nxdi_vl/vision_tp4"); w.load_vision_weights_from_hf(MODEL)
proc=AutoProcessor.from_pretrained(MODEL)
def med(fn,n=5):
    fn(); ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)
print(f"torch threads: {torch.get_num_threads()}")
for size in (512,1024):
    e=proc.image_processor(images=[Image.open(f'/tmp/test_image_{size}.jpg').convert('RGB')],return_tensors="pt")
    pv,g=e["pixel_values"],e["image_grid_thw"]
    t_patch = med(lambda: w.patch_embed(pv))
    t_pos   = med(lambda: w.fast_pos_embed_interpolate(g))
    t_rope  = med(lambda: w.rot_pos_emb(g))
    hs = w.patch_embed(pv); sl = hs.shape[0]
    t_mask  = med(lambda: w._build_vision_attention_mask(g, sl, hs.dtype))
    print(f"{size}x{size} raw={int(g[0].prod())}: patch_embed {t_patch:7.1f} | pos {t_pos:5.1f} "
          f"| rope {t_rope:5.1f} | mask {t_mask:5.1f}  (ms)  pixel_values {tuple(pv.shape)}")
