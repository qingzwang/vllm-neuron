#!/bin/bash
# prof_decode_run.sh <cfg>: sec 6.3 method A. Runs the C=8 D-only bench (same server, same salt, already warmed)
# and captures a 3 s torch-profiler window ~30 s into steady decode (8 running requests).
set -uo pipefail
CFG=${1:?cfg}
URL=http://127.0.0.1:8000
LOG=$HOME/h100_bench/prof_decode/prof_${CFG}_run.log
mkdir -p $HOME/h100_bench/prof_decode
SKIP_WARM=1 bash $HOME/h100_bench/bench_donly.sh $CFG "8" 1200 prof > $HOME/h100_bench/decode_sweep/sweep_${CFG}_prof.log 2>&1 &
BP=$!
running() { curl -s $URL/metrics | awk '$1 ~ /^vllm:num_requests_running/ {s += $NF} END {printf "%d", s}'; }
until [ "$(running)" -ge 8 ]; do sleep 1; done
echo "$(date -u +%T.%N) 8 requests running; waiting 30 s" | tee $LOG
sleep 30
echo "$(date -u +%T.%N) start_profile" | tee -a $LOG
curl -s -X POST $URL/start_profile -w " HTTP %{http_code}\n" | tee -a $LOG
sleep 3
echo "$(date -u +%T.%N) stop_profile (running=$(running))" | tee -a $LOG
curl -s -m 1800 -X POST $URL/stop_profile -w " HTTP %{http_code}\n" | tee -a $LOG
echo "$(date -u +%T.%N) stop_profile returned" | tee -a $LOG
wait $BP
echo "$(date -u +%T) bench done" | tee -a $LOG
ls -lhR $HOME/h100_bench/prof_decode | tee -a $LOG
