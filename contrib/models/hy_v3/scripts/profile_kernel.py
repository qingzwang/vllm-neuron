"""Inside one kernel (by NKI source file) over the 80 layers of one execution: opcodes per engine, DMA
packet sizes, and how compute and DMA overlap in the kernel's time.
python profile_kernel.py <parquet dir> <exec index> <source file stem> [more stems...]"""
import sys, numpy as np, pandas as pd
d, xi = sys.argv[1], int(sys.argv[2])
ex = pd.read_parquet(f"{d}/ExecutionInfo.parquet"); e = ex[ex.execution_index == xi].iloc[0]
F = [("start_ts", ">=", int(e.execution_start_ts)), ("start_ts", "<", int(e.execution_end_ts))]
ins = pd.read_parquet(f"{d}/Instruction.parquet", filters=F, columns=["start_ts", "end_ts", "engine", "opcode", "nki_source_location", "operands"])
dm = pd.read_parquet(f"{d}/DmaPacket.parquet", filters=F, columns=["start_ts", "end_ts", "transfer_bytes", "queue_name"]).sort_values("start_ts")
ins["stem"] = ins.nki_source_location.fillna("").str.extract(r"/([\w_]+)\.py:")[0]
ins["line"] = ins.nki_source_location.fillna("").str.extract(r"/([\w_]+\.py:\d+)")[0]
NL = 80


def merge(a):
    if len(a) == 0: return np.zeros((0, 2), np.int64)
    a = a[np.argsort(a[:, 0])]; out = [list(a[0])]
    for s, t in a[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], t)
        else: out.append([s, t])
    return np.array(out, dtype=np.int64)


def inter(a, b):  # intersection length of two merged interval sets
    i = j = tot = 0
    while i < len(a) and j < len(b):
        s, t = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if s < t: tot += t - s
        if a[i, 1] < b[j, 1]: i += 1
        else: j += 1
    return tot


L = lambda iv: int((iv[:, 1] - iv[:, 0]).sum()) if len(iv) else 0
for stem in sys.argv[3:]:
    g = ins[ins.stem == stem]
    sem = g.opcode.isin(["EVENT_SEMAPHORE", "DRAIN", "NOP"])
    work = g[~sem]
    K = merge(work[["start_ts", "end_ts"]].to_numpy())
    kt = L(K)
    j = np.searchsorted(K[:, 0], dm.start_ts.to_numpy(), side="right") - 1
    m = (j >= 0) & (dm.start_ts.to_numpy() < K[np.clip(j, 0, None), 1])
    dk = dm[m]
    D = merge(dk[["start_ts", "end_ts"]].to_numpy())
    comp = merge(work[work.engine != "Sync"][["start_ts", "end_ts"]].to_numpy())
    print(f"\n=== {stem}: {kt / 1e3 / NL:.1f} us/layer, {len(g) / NL:.0f} instr/layer ({sem.sum() / NL:.0f} semaphore/drain/nop)")
    print(f"DMA: {dk.transfer_bytes.sum() / 1e6 / NL:.2f} MB/layer, {len(dk) / NL:.0f} packets/layer, packet bytes p10/p50/p90 "
          f"{dk.transfer_bytes.quantile([.1, .5, .9]).astype(int).tolist()}, {dk.queue_name.nunique()} queues")
    print(f"time split (us/layer): DMA active {L(D) / 1e3 / NL:.1f} (rate while active {dk.transfer_bytes.sum() / max(L(D), 1):.0f} GB/s), "
          f"compute active {L(comp) / 1e3 / NL:.1f}, both {inter(D, comp) / 1e3 / NL:.1f}, "
          f"neither (only Sync/waits) {(kt - L(merge(np.vstack([D, comp]))) ) / 1e3 / NL:.1f}")
    w = work.assign(dur=work.end_ts - work.start_ts)
    t = w.groupby(["engine", "opcode"]).agg(n=("dur", "size"), tot=("dur", "sum"), mean=("dur", "mean"))
    t["n/layer"] = (t.n / NL).round(0); t["us/layer"] = (t.tot / 1e3 / NL).round(1); t["mean ns"] = t["mean"].round(0)
    print(t.sort_values("tot", ascending=False)[["n/layer", "us/layer", "mean ns"]].head(14).to_string())
    for eng in ["Tensor", "Vector", "Scalar", "GpSimd", "Sync"]:
        ge = work[work.engine == eng]
        print(f"  {eng}: busy {L(merge(ge[['start_ts', 'end_ts']].to_numpy())) / 1e3 / NL:.1f} us/layer", end="")
    print()
    top = w.groupby("line").dur.sum().sort_values(ascending=False).head(8) / 1e3 / NL
    print("top source lines (us/layer, summed durations):", {k: round(v, 1) for k, v in top.items()})
