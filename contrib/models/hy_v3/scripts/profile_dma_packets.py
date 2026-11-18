"""DMA packets that start inside a kernel's instructions (by NKI source file), by size class and
queue type, and the aggregated transfers by source / destination (per layer: 80 layers).
python profile_dma_packets.py <parquet dir> <exec index> <source file stem> [...]"""
import sys, numpy as np, pandas as pd
d, xi = sys.argv[1], int(sys.argv[2])
ex = pd.read_parquet(f"{d}/ExecutionInfo.parquet"); e = ex[ex.execution_index == xi].iloc[0]
F = [("start_ts", ">=", int(e.execution_start_ts)), ("start_ts", "<", int(e.execution_end_ts))]
ins = pd.read_parquet(f"{d}/Instruction.parquet", filters=F, columns=["start_ts", "end_ts", "opcode", "nki_source_location"])
dm = pd.read_parquet(f"{d}/DmaPacket.parquet", filters=F, columns=["start_ts", "end_ts", "transfer_bytes", "queue_name", "queue_type", "read_bytes", "write_bytes"]).sort_values("start_ts")
agg = pd.read_parquet(f"{d}/DmaPacketAggregated.parquet", filters=F, columns=["start_ts", "end_ts", "transfer_bytes", "queue_name", "source", "dest", "read_shape", "op"])
ins["stem"] = ins.nki_source_location.fillna("").str.extract(r"/([\w_]+)\.py:")[0]
def merge(a):
    a = a[np.argsort(a[:, 0])]; out = [list(a[0])]
    for s, t in a[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], t)
        else: out.append([s, t])
    return np.array(out, dtype=np.int64)
bins = [0, 16, 64, 256, 512, 1024, 2048, 4096, 16384, 1 << 40]
for stem in sys.argv[3:]:
    K = merge(ins[(ins.stem == stem) & ~ins.opcode.isin(["EVENT_SEMAPHORE", "DRAIN", "NOP"])][["start_ts", "end_ts"]].to_numpy())
    for name, t in (("packets", dm), ("aggregated transfers", agg)):
        j = np.searchsorted(K[:, 0], t.start_ts.to_numpy(), side="right") - 1
        k = t[(j >= 0) & (t.start_ts.to_numpy() < K[np.clip(j, 0, None), 1])]
        c = pd.cut(k.transfer_bytes, bins, right=False)
        g = k.groupby(c, observed=True).transfer_bytes.agg(["size", "sum"])
        g["MB/layer"] = (g["sum"] / 80 / 1e6).round(2); g["n/layer"] = (g["size"] / 80).round(0)
        print(f"\n{stem} {name}: {k.transfer_bytes.sum() / 80 / 1e6:.2f} MB/layer in {len(k) / 80:.0f}/layer")
        print(g[["n/layer", "MB/layer"]].to_string())
        if name == "packets":
            print(k.groupby("queue_type").transfer_bytes.agg(["size", "sum"]).assign(MB=lambda x: x["sum"] / 80e6).round(2).to_string())
        else:
            print(k.groupby(["source", "dest"]).transfer_bytes.agg(["size", "sum", "median"]).assign(MB_layer=lambda x: x["sum"] / 80e6, n_layer=lambda x: x["size"] / 80).sort_values("sum", ascending=False).head(6).round(2).to_string())
