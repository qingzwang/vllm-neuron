#!/bin/bash
# dma_bench_export.sh <profile dir>: export the nc_0 session of a dma_bench.py PROF run (all NEFFs) to <dir>/pq, then analyze
D=$1; rm -rf $D/sess $D/pq; mkdir -p $D/sess
ln -sf $(find $D -name "profile_nc_0_session_*.ntff" | head -1) $D/sess/
find $D -name "*.neff" -not -path "*/sess/*" -exec ln -sf {} $D/sess/ \;
(cd $D && timeout 1800 /opt/aws/neuron/bin/neuron-explorer view -d sess --output-format parquet --output-file pq --ignore-nc-buf-usage --disable-ui > export.log 2>&1) &
for i in $(seq 1 180); do grep -q "Schema info is available" $D/export.log 2>/dev/null && break; sleep 10; done
sleep 3; ps -eo pid,args | awk -v d="sess" '$2 ~ /neuron-explorer$/ && $4 == d {print $1}' | xargs -r kill
${PY:-python} -I $(dirname "$0")/dma_bench_analyze.py $(ls -d $D/pq/*/) $D/order.json 2>&1 | grep -v Warn
