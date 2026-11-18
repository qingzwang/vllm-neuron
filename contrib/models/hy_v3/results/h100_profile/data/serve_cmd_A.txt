#!/bin/bash
# Config A: gpu_serve.sh flags (TP=8, BF16 KV, no MTP), single node (no PD: --kv-transfer-config / Mooncake / EFA env removed),
# port 8000, image vllm/vllm-openai:v0.29.0, + torch profiler (sec 6.3). /root/.cache mounted only to persist compile caches.
docker rm -f hy3-single >/dev/null 2>&1 || true
docker run -d --name hy3-single --gpus all --privileged --ipc=host --network host \
  -v /opt/dlami/nvme/models:/models \
  -v /opt/dlami/nvme/vllm_cache:/root/.cache \
  -v $HOME/h100_bench/prof_decode:/workspace/prof_decode \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  vllm/vllm-openai:v0.29.0 /models/Hy3-FP8 --served-model-name hy3 \
  --tensor-parallel-size 8 \
  --tool-call-parser hy_v3 --enable-auto-tool-choice --reasoning-parser hy_v3 \
  --max-model-len 87040 \
  --port 8000 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/prof_decode"}'
