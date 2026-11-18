"""Per-transformer-block breakdown of one decode target execution (instructions + DMA + collectives).

python profile_block.py <parquet dir> [exec index]
The timeline is cut at the all-reduces: the first one is the embedding's, then every layer issues
two (after attention, after MoE). Layer i = end of the previous layer's MoE all-reduce .. end of its
own. Phases: attention (until the attention all-reduce starts), attn AR, MoE, MoE AR. Inside a phase
instructions are grouped by NKI source file (torch ops: "torch"); DMA bytes are the packets that
start inside a group's instruction intervals; idle = no instruction and no collective.
"""
import sys, numpy as np, pandas as pd

d = sys.argv[1]
ex = pd.read_parquet(f"{d}/ExecutionInfo.parquet")
ex["ms"] = (ex.execution_end_ts - ex.execution_start_ts) / 1e6
cnt = pd.read_parquet(f"{d}/Instruction.parquet", columns=["execution_index"]).execution_index.value_counts()
if len(sys.argv) > 2:
    xi = int(sys.argv[2])
else:
    big = ex.groupby("neff_name").ms.median().idxmax()
    cand = ex[(ex.neff_name == big) & ex.execution_index.isin(cnt[cnt > 0.8 * cnt.max()].index)].sort_values("ms")
    xi = int(cand.iloc[len(cand) // 2].execution_index)
e = ex[ex.execution_index == xi].iloc[0]
t0, t1 = int(e.execution_start_ts), int(e.execution_end_ts)
F = [("start_ts", ">=", t0), ("start_ts", "<", t1)]
ins = pd.read_parquet(f"{d}/Instruction.parquet", filters=F, columns=[
    "start_ts", "end_ts", "engine", "opcode", "nki_source_location", "hlo_name"])
ins = ins[~ins.opcode.isin(["EVENT_SEMAPHORE", "DRAIN", "NOP"])].copy()
src = ins.nki_source_location.fillna("").str.extract(r"/([\w_]+)\.py:")[0]
TORCH = {"model", "nki_hop", "_functional_collectives", "functional"}
ins["part"] = np.where(src.notna() & ~src.isin(TORCH), src, "torch")
dm = pd.read_parquet(f"{d}/DmaPacket.parquet", filters=F, columns=["start_ts", "transfer_bytes"]).sort_values("start_ts")
dts, dby = dm.start_ts.to_numpy(), dm.transfer_bytes.to_numpy()
cc = pd.read_parquet(f"{d}/CcOp.parquet", filters=F).sort_values("start_ts")
ar = cc[cc.operation == "ALLREDUCE"].reset_index(drop=True)
nl = (len(ar) - 1) // 2
print(f"execution {xi}: {(t1 - t0) / 1e6:.2f} ms, {len(ar)} all-reduces of {ar.input_size.median() / 1024:.0f} KB -> {nl} layers")


def merge(a):
    if len(a) == 0: return np.zeros((0, 2), np.int64)
    a = a[np.argsort(a[:, 0])]; out = [list(a[0])]
    for s, t in a[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], t)
        else: out.append([s, t])
    return np.array(out, dtype=np.int64)


def clip(iv, s, t):
    iv = iv[(iv[:, 1] > s) & (iv[:, 0] < t)].copy()
    if len(iv): iv[:, 0] = np.maximum(iv[:, 0], s); iv[:, 1] = np.minimum(iv[:, 1], t)
    return iv


def length(iv): return int((iv[:, 1] - iv[:, 0]).sum()) if len(iv) else 0


def dma_in(iv):
    if not len(iv): return 0
    j = np.searchsorted(iv[:, 0], dts, side="right") - 1
    return int(dby[(j >= 0) & (dts < iv[np.clip(j, 0, None), 1])].sum())


allv = merge(ins[["start_ts", "end_ts"]].to_numpy())
ins = ins.sort_values("start_ts"); st = ins.start_ts.to_numpy()
rows = []
for i in range(nl):
    w0 = int(ar.end_ts[2 * i]); a0, a1 = int(ar.start_ts[2 * i + 1]), int(ar.end_ts[2 * i + 1])
    m0, m1 = int(ar.start_ts[2 * i + 2]), int(ar.end_ts[2 * i + 2])
    r = {"layer": i, "span": m1 - w0, "attn": a0 - w0, "attn_AR": a1 - a0, "moe": m0 - a1, "moe_AR": m1 - m0,
         "DMA": int(dby[(dts >= w0) & (dts < m1)].sum())}
    for ph, (s, t) in {"attn_AR": (a0, a1), "moe_AR": (m0, m1)}.items():
        r[ph + "|busy"] = length(clip(allv, s, t))
        r[ph + "|DMA"] = int(dby[(dts >= s) & (dts < t)].sum())
    for ph, (s, t) in {"attn": (w0, a0), "moe": (a1, m0)}.items():
        g = ins.iloc[np.searchsorted(st, s):np.searchsorted(st, t)]
        r[ph + "|idle"] = (t - s) - length(clip(allv, s, t))
        r[ph + "|DMA"] = int(dby[(dts >= s) & (dts < t)].sum())
        for p, gp in g.groupby("part"):
            iv = clip(merge(gp[["start_ts", "end_ts"]].to_numpy()), s, t)
            k = f"{ph}:{p}"
            r[k] = length(iv); r[k + "|B"] = dma_in(iv); r[k + "|n"] = len(gp)
            for eng, ge in gp.groupby("engine"):
                r[k + "|" + eng] = length(clip(merge(ge[["start_ts", "end_ts"]].to_numpy()), s, t))
        for eng, ge in g.groupby("engine"):
            r[ph + "|eng|" + eng] = length(clip(merge(ge[["start_ts", "end_ts"]].to_numpy()), s, t))
    rows.append(r)
df = pd.DataFrame(rows).fillna(0)
moe_layers = df[df.layer >= 1]
med = moe_layers.median(numeric_only=True)
us = lambda x: x / 1e3
print(f"\nlayers 1-{nl - 1}, median per layer (us): span {us(med.span):.1f} = attention {us(med.attn):.1f} + attn AR {us(med.attn_AR):.1f}"
      f" + MoE {us(med.moe):.1f} + MoE AR {us(med.moe_AR):.1f};  DMA {med.DMA / 1e6:.2f} MB/layer "
      f"({med.DMA / med.span:.0f} GB/s averaged over the span)")
print(f"whole step: {(t1 - t0) / 1e6:.2f} ms; sum of layer spans {df.span.sum() / 1e6:.2f} ms; "
      f"attention {df.attn.sum() / 1e6:.2f}, attn AR {df.attn_AR.sum() / 1e6:.2f}, MoE {df.moe.sum() / 1e6:.2f}, MoE AR {df.moe_AR.sum() / 1e6:.2f}")
for ph in ("attn", "moe"):
    print(f"\n[{ph} phase] {us(med[ph]):.1f} us, DMA {med[ph + '|DMA'] / 1e6:.2f} MB ({med[ph + '|DMA'] / max(med[ph], 1):.0f} GB/s), "
          f"idle {us(med[ph + '|idle']):.1f} us; engine busy: " +
          " ".join(f"{k.split('|')[2]} {us(med[k]):.0f}" for k in med.index if k.startswith(ph + "|eng|")))
    print(f"  {'group':28s} {'us':>7s} {'DMA MB':>7s} {'GB/s':>6s} {'instr':>6s}  engine busy us")
    ks = sorted([k for k in med.index if k.startswith(ph + ":") and "|" not in k], key=lambda k: -med[k])
    for k in ks:
        if med[k] < 300: continue
        eng = " ".join(f"{c.split('|')[1]} {us(med[c]):.0f}" for c in med.index
                       if c.startswith(k + "|") and c.split("|")[1] not in ("B", "n") and med[c] > 300)
        print(f"  {k.split(':')[1]:28s} {us(med[k]):7.1f} {med[k + '|B'] / 1e6:7.2f} {med[k + '|B'] / max(med[k], 1):6.0f} {med[k + '|n']:6.0f}  {eng}")
print("\n[overlap] while an all-reduce runs, any engine busy / DMA moved (median per layer):")
for ph in ("attn_AR", "moe_AR"):
    print(f"  {ph}: {us(med[ph]):.1f} us, engines busy {us(med[ph + '|busy']):.1f} us "
          f"({100 * med[ph + '|busy'] / max(med[ph], 1):.0f}%), DMA {med[ph + '|DMA'] / 1e6:.2f} MB")
df.to_csv(f"block_{xi}.csv", index=False)
