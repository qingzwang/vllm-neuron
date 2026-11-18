import argparse, gzip, json, glob, os, re, statistics as st

CATS = [
    ("comm",        r"allreduce|all_reduce|cross_device|nccl|reduce_scatter|reducescatter|allgather|all_gather|lamport|oneshot|twoshot|custom_ar"),
    ("moe",         r"moe|expert|grouped_gemm|topk_gating|topkgating|fused_moe"),
    ("attention",   r"flash|fmha|attention|attn|paged|batchdecode|batchprefill|decode_kernel|merge_state|mla"),
    ("gemm",        r"gemm|sm90|cutlass|xmma|cublas|nvjet|scaled_mm|matmul"),
    ("sampling",    r"sampl|argmax|top_k|top_p|topk|radix|sort|multinomial|softmax|rejection"),
    ("elementwise", r"norm|rms|rotary|rope|silu|act_and_mul|elementwise|vectorized|reduce_kernel|copy|cat|index|fill|memcpy|memset|gather|scatter"),
]
def cat(name):
    n = name.lower()
    for c, p in CATS:
        if re.search(p, n):
            return c
    return "other"

def load(f):
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        return json.load(fh).get("traceEvents", [])

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--step-regex", default=None,
                help="正则:每个 forward step 只出现一次的 kernel,用来算步数,把总量换算成每步")
ap.add_argument("--top", type=int, default=40)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dir, "**", "*.json*"), recursive=True))
if not files:
    print("no trace files under", a.dir)
for k, f in enumerate(files):
    ev = load(f)
    ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    print(f"\n### {os.path.relpath(f, a.dir)}")
    if not ker:
        print("  no GPU kernels in this file"); continue
    t0 = min(e["ts"] for e in ker); t1 = max(e["ts"] + e["dur"] for e in ker)
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ker)
    busy = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: busy += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    busy += ce - cs
    nsteps = None
    if a.step_regex:
        nsteps = sum(1 for e in ker if re.search(a.step_regex, e["name"], re.I)) or None
    glaunch = sum(1 for e in ev if e.get("name") == "cudaGraphLaunch")
    print(f"  window {(t1-t0)/1e3:.1f} ms | GPU busy {busy/1e3:.1f} ms ({busy/(t1-t0):.0%}) | kernels {len(ker)} | cudaGraphLaunch {glaunch} | steps {nsteps}")
    tot = {}; comm = {}
    for e in ker:
        c = cat(e["name"]); tot[c] = tot.get(c, 0) + e["dur"]
        if c == "comm": comm.setdefault(e["name"][:90], []).append(e["dur"])
    print("  category       total ms" + ("    ms/step" if nsteps else ""))
    for c, v in sorted(tot.items(), key=lambda x: -x[1]):
        line = f"  {c:12s} {v/1e3:10.1f}"
        if nsteps: line += f" {v/1e3/nsteps:10.2f}"
        print(line)
    for n, ds in sorted(comm.items(), key=lambda x: -sum(x[1])):
        q = st.quantiles(ds, n=10) if len(ds) >= 10 else [min(ds), max(ds)]
        per = f" | {len(ds)/nsteps:.0f}/step" if nsteps else ""
        print(f"  comm kernel: n={len(ds)}{per} median {st.median(ds):.1f} us p10 {q[0]:.1f} p90 {q[-1]:.1f} | {n}")
    if k == 0:
        agg = {}
        for e in ker:
            x = agg.setdefault(e["name"][:120], [0, 0]); x[0] += 1; x[1] += e["dur"]
        print(f"  top {a.top} kernels (this file): count, total ms, avg us, category, name")
        for n, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:a.top]:
            print(f"   {c:7d} {d/1e3:9.2f} {d/c:9.1f}  {cat(n):11s} {n}")
