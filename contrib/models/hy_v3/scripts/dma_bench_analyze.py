"""Bandwidths from a dma_bench.py device profile.

python dma_bench_analyze.py <parquet dir> <order.json>
Per execution: device time and the DMA bytes the profile saw; per (mode, row size): bandwidth as the
slope of device time between the two pass counts, and the bytes seen / expected (1.0 = no DMA dropped).
"""
import json, sys
import numpy as np, pandas as pd

d, order = sys.argv[1], json.load(open(sys.argv[2]))
ex = pd.read_parquet(f"{d}/ExecutionInfo.parquet").sort_values("execution_start_ts").reset_index(drop=True)
print(f"{len(ex)} executions in the profile, {len(order)} expected")
ex = ex.tail(len(order)).reset_index(drop=True)
dm = pd.read_parquet(f"{d}/DmaPacket.parquet", columns=["start_ts", "transfer_bytes", "queue_type"])
ts = np.sort(dm.start_ts.to_numpy()); cb = np.concatenate([[0], np.cumsum(dm.sort_values("start_ts").transfer_bytes.to_numpy())])
rows = []
for (mode, F, r, nbytes), e in zip(order, ex.itertuples()):
    i0, i1 = np.searchsorted(ts, e.execution_start_ts), np.searchsorted(ts, e.execution_end_ts)
    rows.append((mode, F, r, nbytes, (e.execution_end_ts - e.execution_start_ts) / 1e6, cb[i1] - cb[i0]))
df = pd.DataFrame(rows, columns=["mode", "F", "reps", "bytes", "ms", "dma_bytes"])
g = df.groupby(["mode", "F", "reps"], sort=False).agg(ms=("ms", "median"), bytes=("bytes", "first"), seen=("dma_bytes", "median")).reset_index()
out = []
for (mode, F), h in g.groupby(["mode", "F"], sort=False):
    h = h.sort_values("reps")
    a, b = h.iloc[0], h.iloc[-1]
    gbps = (b.bytes - a.bytes) / ((b.ms - a.ms) * 1e-3) / 1e9
    out.append((mode, F, round(a.ms, 3), round(b.ms, 3), round(gbps, 1), round(b.seen / b.bytes, 2)))
print(pd.DataFrame(out, columns=["mode", "row bytes", f"ms ({int(a.reps)} passes)", f"ms ({int(b.reps)} passes)", "GB/s", "bytes seen/expected"]).to_string(index=False))
