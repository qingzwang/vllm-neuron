#!/bin/bash
# 85k-context config without DCP, for MTP on the bench_data/prefix prompts
# (MTP does not run with DCP, so scripts/serve_256k.sh cannot take MTP_K).
# Same as serve_256k.sh except: no DCP, max_model_len 86016 (85000-token
# prompts + 256 output tokens + chat template), decode context buckets
# 8192 / 32768 / 65536 (+ the implicit 86016) and max_num_seqs 4.
# The scheduler admits requests by worst case (max_model_len each), so 4
# requests run at once and the rest wait: KV_CAP 0.65 holds 376k tokens
# (4.37x). With MTP the draft layer's KV and buffers need more HBM: 0.65 runs
# out at 4 concurrent requests (nrt_tensor_allocate), 0.60 holds only 3.99
# requests, so the MTP default is 0.61 (349k tokens, 4.05x). KV_CAP 0.85
# leaves too little HBM for the decode NEFFs to load.
# Env: MODEL, PORT, MTP_K, KV_CAP (default 0.65, with MTP_K 0.61) as for serve.sh; with MTP_K,
#      HY3_DECODE_HITS_MAX_T defaults to 4 x (MTP_K + 1) (4-request verify steps).
# Benchmark: bench_pd_split.sh "1 4 8".
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEFAULT_KV_CAP=0.65
if [ -n "${MTP_K:-}" ]; then
    export HY3_DECODE_HITS_MAX_T=${HY3_DECODE_HITS_MAX_T:-$((4 * (MTP_K + 1)))}
    DEFAULT_KV_CAP=0.61
fi
KV_CAP=${KV_CAP:-$DEFAULT_KV_CAP} PREFILL_BUCKETS='[512, 1024, 2048, 4096]' \
    exec "$HERE/serve.sh" 86016 4 '[1, 4]' '[8192, 32768, 65536]'
