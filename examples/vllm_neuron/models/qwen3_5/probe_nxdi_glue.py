#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""cProfile one *reference* (NxDI) VL prefill, to see where its TTFT goes.

Third of three probes aimed at the other implementation, after
``probe_nxdi_vision.py`` (encoder vs glue) and ``probe_nxdi_prep.py`` (the CPU prep
by component). This one exists because subtracting measured parts from measured
totals produced a 264 ms residual that was assumed to be Python orchestration --
and profiling showed no such residual: the text CTE call is simply 368 ms, not the
90.6 ms its standalone smoke test suggested.

Findings at 1024x1024, TP=4, on trn2.3xlarge: text forward 368 ms, vision graph
101 ms, CPU conv3d patch-embed 85 ms, grid-only prep 14 ms, mRoPE ~1 ms. See
HANDOFF, "What the reference's glue actually does".

Note it reuses the reference's own ``build_vl_config``/``compile_and_load``, so the
model config matches its benchmark exactly rather than being re-guessed here. Needs
the text model already compiled at /tmp/nxdi_vl/text_tp4 (``--skip-compile`` is
passed), the TP=4 encoder at /tmp/nxdi_vl/vision_tp4, and the image staged.

Same venv and PATH requirements as ``probe_nxdi_vision.py``.
"""
import os, sys, json, cProfile, pstats, io, time
os.environ.setdefault("QWEN36_DELTANET_CTE_IMPL", "legacy_direct")
os.environ.setdefault("QWEN36_DELTANET_MULTIHEAD_CTE", "0")
REF = "/mnt/nvme/nxdi_ref"
sys.path.insert(0, f"{REF}/contrib/models/Qwen3.5-2B")
sys.path.insert(0, f"{REF}/contrib/models/Qwen3.5-2B/test/integration")
import torch
from PIL import Image
from transformers import AutoProcessor
from run_vl_benchmark import build_vl_config, compile_and_load

MODEL = "/mnt/nvme/models/Qwen3.5-2B"
text_config, vl_config = build_vl_config(MODEL, 4, 4096, [512, 1024, 2048, 4096])
proc = AutoProcessor.from_pretrained(MODEL)
vl = compile_and_load(MODEL, "/tmp/nxdi_vl/text_tp4", text_config, vl_config,
                      skip_compile=True, vision_compiled_dir="/tmp/nxdi_vl/vision_tp4")

img = Image.open("/tmp/test_image_1024.jpg").convert("RGB")
msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                     {"type": "text", "text": "What is in this image? Describe it briefly."}]}]
inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True)
kw = dict(input_ids=inp["input_ids"],
          attention_mask=inp.get("attention_mask", torch.ones_like(inp["input_ids"])),
          pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"],
          max_new_tokens=1, temperature=0.0)
print("input_ids:", tuple(inp["input_ids"].shape), flush=True)

vl.generate(**kw)                     # warm
t0 = time.perf_counter(); vl.generate(**kw); wall = (time.perf_counter()-t0)*1000
print(f"un-profiled prefill wall: {wall:.1f} ms", flush=True)

pr = cProfile.Profile(); pr.enable(); vl.generate(**kw); pr.disable()
st = pstats.Stats(pr); st.sort_stats("cumulative")
buf = io.StringIO(); st.stream = buf; st.print_stats(45)
print(buf.getvalue()[:7000])
print("=== by total (self) time ===")
buf2 = io.StringIO(); st.stream = buf2; st.sort_stats("tottime"); st.print_stats(20)
print(buf2.getvalue()[:3500])
