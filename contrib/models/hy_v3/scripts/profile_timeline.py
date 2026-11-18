"""One decode layer's engine timeline (merged busy intervals per engine, DMA active, collectives)
for plotting. python profile_timeline.py <parquet dir> <exec index> <layer> <out.json>"""
import json, sys, numpy as np, pandas as pd
d, xi, li, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ex = pd.read_parquet(f"{d}/ExecutionInfo.parquet"); e = ex[ex.execution_index == xi].iloc[0]
F = [("start_ts", ">=", int(e.execution_start_ts)), ("start_ts", "<", int(e.execution_end_ts))]
cc = pd.read_parquet(f"{d}/CcOp.parquet", filters=F).sort_values("start_ts")
ar = cc[cc.operation == "ALLREDUCE"].reset_index(drop=True)
w0, w1 = int(ar.end_ts[2 * li]), int(ar.end_ts[2 * li + 2])
G = [("start_ts", ">=", w0 - 200_000), ("start_ts", "<", w1)]
ins = pd.read_parquet(f"{d}/Instruction.parquet", filters=G, columns=["start_ts", "end_ts", "engine", "opcode", "nki_source_location"])
ins = ins[~ins.opcode.isin(["EVENT_SEMAPHORE", "DRAIN", "NOP"]) & (ins.end_ts > w0)]
dm = pd.read_parquet(f"{d}/DmaPacket.parquet", filters=G, columns=["start_ts", "end_ts", "transfer_bytes"])
dm = dm[dm.end_ts > w0]


def merge(a, gap=0):
    if len(a) == 0: return []
    a = a[np.argsort(a[:, 0])]; out = [list(a[0])]
    for s, t in a[1:]:
        if s <= out[-1][1] + gap: out[-1][1] = max(out[-1][1], t)
        else: out.append([s, t])
    return [[max(int(s), w0) - w0, min(int(t), w1) - w0] for s, t in out if t > w0 and s < w1]


lanes = {}
for eng in ["Tensor", "Vector", "Scalar", "GpSimd", "Sync"]:
    lanes[eng] = merge(ins[ins.engine == eng][["start_ts", "end_ts"]].to_numpy(), gap=50)
lanes["DMA"] = merge(dm[["start_ts", "end_ts"]].to_numpy(), gap=50)
# DMA throughput in 5 us bins
bins = np.arange(w0, w1 + 5000, 5000)
mid = ((dm.start_ts + dm.end_ts) // 2).to_numpy()
tp = np.histogram(mid, bins=bins, weights=dm.transfer_bytes.to_numpy())[0] / 5000  # bytes/ns = GB/s
phases = {"attention": [0, int(ar.start_ts[2 * li + 1]) - w0], "attn_AR": [int(ar.start_ts[2 * li + 1]) - w0, int(ar.end_ts[2 * li + 1]) - w0],
          "moe": [int(ar.end_ts[2 * li + 1]) - w0, int(ar.start_ts[2 * li + 2]) - w0], "moe_AR": [int(ar.start_ts[2 * li + 2]) - w0, w1 - w0]}
json.dump({"span_ns": w1 - w0, "lanes": lanes, "phases": phases, "dma_gbps_5us": tp.round(1).tolist()}, open(out, "w"))
print({k: (len(v), sum(t - s for s, t in v) / 1e3) for k, v in lanes.items()}, phases)
