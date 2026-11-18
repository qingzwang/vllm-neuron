#!/bin/bash
# Config B: scripts/pd/gpu/gpu_single_serve.sh with IMAGE=vllm/vllm-openai:v0.29.0, MAX_LEN=87040, no OFFLOAD_BYTES
# (TP=8, FP8 KV, MTP k=3), + torch profiler (sec 6.3). Extra mounts: profiler dir, /root/.cache (compile caches only).
docker rm -f hy3-single >/dev/null 2>&1 || true
docker run -d --name hy3-single --gpus all --ipc=host --network host \
  -v /opt/dlami/nvme/models:/models \
  -v /opt/dlami/nvme/vllm_cache:/root/.cache \
  -v $HOME/h100_bench/prof_decode:/workspace/prof_decode \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  vllm/vllm-openai:v0.29.0 /models/Hy3-FP8 --served-model-name hy3 \
  --tensor-parallel-size 8 \
  --tool-call-parser hy_v3 --enable-auto-tool-choice --reasoning-parser hy_v3 \
  --max-model-len 87040 --port 8000 \
  --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/prof_decode"}'
