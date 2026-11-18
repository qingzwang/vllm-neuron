"""Per-DMA-engine busy fraction and rate inside a kernel's instructions (by NKI source file), and the
median software-DGE packet size and duration.
python profile_dma_engines.py <parquet dir> <exec index> <source file stem> [...]"""
import sys, numpy as np, pandas as pd
d, xi = sys.argv[1], int(sys.argv[2])
ex = pd.read_parquet(f"{d}/ExecutionInfo.parquet"); e = ex[ex.execution_index == xi].iloc[0]
F = [("start_ts", ">=", int(e.execution_start_ts)), ("start_ts", "<", int(e.execution_end_ts))]
ins = pd.read_parquet(f"{d}/Instruction.parquet", filters=F, columns=["start_ts", "end_ts", "opcode", "nki_source_location"])
dm = pd.read_parquet(f"{d}/DmaPacket.parquet", filters=F, columns=["start_ts", "end_ts", "transfer_bytes", "engine_idx", "pcore_idx", "queue_type"])
ins["stem"] = ins.nki_source_location.fillna("").str.extract(r"/([\w_]+)\.py:")[0]
def merge(a):
    a = a[np.argsort(a[:, 0])]; out = [list(a[0])]
    for s, t in a[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], t)
        else: out.append([s, t])
    return np.array(out, dtype=np.int64)
for stem in sys.argv[3:]:
    K = merge(ins[(ins.stem == stem) & ~ins.opcode.isin(["EVENT_SEMAPHORE", "DRAIN", "NOP"])][["start_ts", "end_ts"]].to_numpy())
    kt = (K[:, 1] - K[:, 0]).sum()
    j = np.searchsorted(K[:, 0], dm.start_ts.to_numpy(), side="right") - 1
    k = dm[(j >= 0) & (dm.start_ts.to_numpy() < K[np.clip(j, 0, None), 1])]
    rows = []
    for (pc, en), g in k.groupby(["pcore_idx", "engine_idx"]):
        iv = merge(g[["start_ts", "end_ts"]].to_numpy()); b = (iv[:, 1] - iv[:, 0]).sum()
        rows.append((pc, en, b / kt, g.transfer_bytes.sum() / b, g.transfer_bytes.sum() / kt))
    r = pd.DataFrame(rows, columns=["pcore", "engine", "busy_frac", "GBps_while_busy", "GBps_avg"])
    print(f"\n{stem}: {len(r)} DMA engines; kernel time {kt / 1e3 / 80:.0f} us/layer; total {k.transfer_bytes.sum() / kt:.0f} GB/s")
    print(r.describe().loc[["mean", "min", "50%", "max"]].round(2).to_string())
    sw = k[k.queue_type == "software_dynamic"]
    dur = (sw.end_ts - sw.start_ts)
    print("software-DGE packet: median bytes", int(sw.transfer_bytes.median()), "median duration ns", int(dur.median()),
          "-> per-packet rate", round(sw.transfer_bytes.median() / dur.median(), 1), "GB/s")
