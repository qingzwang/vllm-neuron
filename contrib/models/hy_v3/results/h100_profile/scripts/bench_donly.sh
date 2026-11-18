#!/bin/bash
# bench_donly.sh <cfg A|B> <concurrencies, e.g. "1 4 8"> [output_len=1200] [tag]
# Single-server adaptation of contrib/models/hy_v3/scripts/pd/pd_bench_dcust.sh (D-only replay):
#   P_URL = D_URL = PROXY = http://127.0.0.1:8000 (one server, no PD).
#   1. warm pass: all 80 measured requests of in85000_p78336_c8 (prefix + suffix), output 1, C=8, same cache_salt.
#      Skipped if SKIP_WARM=1 (when the salt was already warmed on this server process).
#   2. per C: first 10*C requests (capped at 80), same salt, output 1200, --ignore-eos, 2 warm-ups,
#      temperature 0.9, seed 0, --request-rate inf. Same vllm bench serve flags as pd_bench_dcust.sh.
# Full bench stdout -> decode_sweep/<cfg>[_tag]/c<C>.txt ; JSON (+ hit rate / MTP from /metrics) -> c<C>.json
set -uo pipefail
CFG=${1:?cfg}; CONCS=${2:-1 4 8}; OUTPUT_LEN=${3:-1200}; TAG=${4:-}
URL=http://127.0.0.1:8000
IMAGE=hy3-bench-client  # vllm/vllm-openai:v0.29.0 + pandas/datasets (vllm[bench] deps)
CODE=$HOME/vllm-neuron/contrib/models/hy_v3
SALT=${SALT:-donly-$CFG}
OUT=$HOME/h100_bench/decode_sweep/${CFG}${TAG:+_$TAG}
mkdir -p "$OUT"
DATA=$HOME/h100_bench/decode_sweep/all.jsonl
if [ ! -s "$DATA" ]; then
python3 - "$CODE/bench_data/prefix/in85000_p78336_c8.jsonl.gz" "$DATA" <<'EOF'
import gzip, json, sys
with gzip.open(sys.argv[1], "rt") as f:
    rows = [json.loads(line) for line in f]
with open(sys.argv[2], "w") as f:
    for r in rows[2:]:  # the 80 measured requests of the c8 file (same as pd_bench_dcust.sh)
        f.write(json.dumps({"prompt": rows[0]["prefix"] + r["suffix"]}, ensure_ascii=False) + "\n")
EOF
fi
N_ALL=$(wc -l < "$DATA")

counter() { curl -s "$URL/metrics" | awk -v n="$1" '$1 ~ "^"n"(_total)?([{]|$)" {s += $NF} END {printf "%d", s}'; }
snap() {
  eval "$1_q=$(counter vllm:prefix_cache_queries) $1_h=$(counter vllm:prefix_cache_hits)"
  eval "$1_sd=$(counter vllm:spec_decode_num_drafts) $1_sa=$(counter vllm:spec_decode_num_accepted_tokens) $1_st=$(counter vllm:spec_decode_num_draft_tokens)"
}
bench() {  # bench <name> <vllm bench serve args...>
  local name=$1; shift
  snap a
  curl -s "$URL/metrics" > "$OUT/$name.metrics_before.txt"
  docker run --rm --network host --entrypoint vllm \
    -v /opt/dlami/nvme/models:/models -v $HOME/h100_bench/decode_sweep:/data -v "$OUT":/out \
    $IMAGE bench serve --backend openai-chat --base-url "$URL" --endpoint /v1/chat/completions \
    --model /models/Hy3-FP8 --served-model-name hy3 --tokenizer /models/Hy3-FP8 --trust-remote-code \
    --dataset-name custom --dataset-path /data/all.jsonl --skip-chat-template --disable-shuffle --request-rate inf --ignore-eos \
    --ready-check-timeout-sec 0 --extra-body "{\"cache_salt\": \"$SALT\"}" --metric-percentiles 50,99 \
    --temperature 0.9 --seed 0 --save-result --result-dir /out --result-filename "$name.json" \
    "$@" 2>&1 | tee "$OUT/$name.txt" | grep -E "Successful|Failed|duration|(Mean|Median|P99) (TTFT|TPOT|ITL)|throughput"
  snap b
  curl -s "$URL/metrics" > "$OUT/$name.metrics_after.txt"
  sudo chown -R $(id -u):$(id -g) "$OUT" 2>/dev/null
  python3 - "$OUT/$name.json" $((b_q - a_q)) $((b_h - a_h)) $((b_sd - a_sd)) $((b_sa - a_sa)) $((b_st - a_st)) <<'EOF' | tee -a "$OUT/$name.txt"
import json, sys
path = sys.argv[1]
q, h, sd, sa, st = map(int, sys.argv[2:])
d = json.load(open(path))
d.update(prefix_hit_rate=h / q if q else 0.0, spec_num_drafts=sd, spec_accepted=sa, spec_draft_tokens=st)
d["spec_tokens_per_step"] = 1 + sa / sd if sd else 1.0
if st: d["spec_accept_rate"] = sa / st
json.dump(d, open(path, "w"))
print(f"[metrics] prefix hit rate {d['prefix_hit_rate']:.1%} (queries {q}, hits {h}); "
      f"MTP tokens/step {d['spec_tokens_per_step']:.3f} (drafts {sd}, accepted {sa}, draft tokens {st})")
EOF
}

curl -sf "$URL/health" >/dev/null || { echo "server not up"; exit 1; }
echo "$(date -u +%T) cfg $CFG salt $SALT concs [$CONCS] out $OUT"
if [ "${SKIP_WARM:-0}" != 1 ]; then
  echo "=== warm pass: $N_ALL requests, output 1"
  bench warm --custom-output-len 1 --num-prompts "$N_ALL" --max-concurrency 8 --percentile-metrics ttft
fi
for C in $CONCS; do
  N=$((10 * C)); [ $N -gt "$N_ALL" ] && N=$N_ALL
  echo "=== $(date -u +%T) D c$C ($N requests, output $OUTPUT_LEN)"
  bench c$C --custom-output-len "$OUTPUT_LEN" --num-prompts $N --max-concurrency $C --num-warmups 2 \
    --percentile-metrics ttft,tpot,itl,e2el
done
