"""Supplement to prof_summary.py (does not replace it). For one torch-profiler trace:
  * splits GPU kernels by the CUDA graph they ran in ('graph id' arg; 0 = eager, not in a graph)
    -> with MTP, the target forward and the draft forwards are separate graphs;
  * per graph: launches, launches/step, ms/launch, ms/step, category split (prof_summary.cat);
  * every kernel: count/step, ms/step, avg us, graph, category, and a 'refined' category
    (see REFINE below; names that cannot be attributed from the name alone are marked 'unclear').
Usage: python prof_detail.py <trace.json.gz> --step-regex '<kernel once per step>' [--top N]
"""
import argparse, gzip, json, re, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("ps_cats", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prof_summary.py"))
src = open(spec.origin).read().split("def load(")[0]  # only CATS and cat()
ns = {}; exec(src, ns); cat = ns["cat"]

# Refinement, by kernel name only. Evidence: vLLM log "Using FLASHINFER_CUTLASS Fp8 MoE backend";
# these kernels run once per MoE layer (79/forward, = number of layers with routed experts).
REFINE = [
    ("moe", r"tensorrt_llm::kernels::cutlass_kernels|GroupProblemShape"),  # flashinfer cutlass fused-MoE (grouped GEMM, expand/permute, activation, finalize)
    ("unclear(fp32 simt sgemm, 1/MoE layer: likely router gate GEMM)", r"cutlass_80_simt_sgemm|splitKreduce_kernel"),
    ("unclear(triton fused, Inductor-generated)", r"^triton_(red|per|poi)_fused_(\d+|add_\d+|.*_[0-9]+)$"),
]
def refine(n):
    for c, p in REFINE:
        if re.search(p, n):
            return c
    return cat(n)

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--step-regex", required=True); ap.add_argument("--top", type=int, default=60)
a = ap.parse_args()
op = gzip.open if a.trace.endswith(".gz") else open
ev = json.load(op(a.trace, "rt"))["traceEvents"]
ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
nsteps = sum(1 for e in ker if re.search(a.step_regex, e["name"], re.I))
t0 = min(e["ts"] for e in ker); t1 = max(e["ts"] + e["dur"] for e in ker)
print(f"trace {os.path.basename(a.trace)}\nwindow {(t1-t0)/1e3:.1f} ms, steps {nsteps} (regex {a.step_regex!r}), "
      f"window/steps {(t1-t0)/1e3/nsteps:.2f} ms, GPU kernel sum {sum(e['dur'] for e in ker)/1e3/nsteps:.2f} ms/step")

gid = lambda e: e.get("args", {}).get("graph id", 0) or 0
corr = lambda e: e.get("args", {}).get("correlation")
by_g = collections.defaultdict(list)
for e in ker: by_g[gid(e)].append(e)
print("\n## per CUDA graph (graph id 0 = eager kernels outside graphs)")
print(f"{'graph':>6} {'launches':>8} {'/step':>6} {'kern/launch':>11} {'ms/launch':>9} {'ms/step':>8}  categories ms/step (refined)")
for g, ks in sorted(by_g.items(), key=lambda x: -sum(e['dur'] for e in x[1])):
    launches = len({corr(e) for e in ks}) if g else 0
    tot = sum(e["dur"] for e in ks)
    cs = collections.Counter()
    for e in ks: cs[refine(e["name"]).split("(")[0]] += e["dur"]
    csx = ", ".join(f"{c} {v/1e3/nsteps:.2f}" for c, v in cs.most_common())
    per_l = f"{len(ks)/launches:11.0f} {tot/1e3/launches:9.3f}" if launches else f"{'-':>11} {'-':>9}"
    print(f"{g:>6} {launches:>8} {launches/nsteps:6.2f} {per_l} {tot/1e3/nsteps:8.3f}  {csx}")

print("\n## categories ms/step, all graphs: prof_summary category vs refined")
c1 = collections.Counter(); c2 = collections.Counter()
for e in ker: c1[cat(e["name"])] += e["dur"]; c2[refine(e["name"])] += e["dur"]
for c, v in c1.most_common(): print(f"  summary  {c:12s} {v/1e3/nsteps:7.3f}")
for c, v in c2.most_common(): print(f"  refined  {c:60s} {v/1e3/nsteps:7.3f}")

print(f"\n## top {a.top} kernels: count/step, ms/step, avg us, graph id(s), summary cat, refined cat, name")
agg = {}
for e in ker:
    x = agg.setdefault(e["name"], [0, 0, set()]); x[0] += 1; x[1] += e["dur"]; x[2].add(gid(e))
for n, (c, d, gs) in sorted(agg.items(), key=lambda x: -x[1][1])[:a.top]:
    print(f"  {c/nsteps:7.2f} {d/1e3/nsteps:7.3f} {d/c:8.1f}  g{','.join(map(str, sorted(gs)))[:12]:12s} {cat(n):11s} {refine(n)[:28]:28s} {n[:150]}")

print("\n## comm kernels by graph: count/step, median us")
cm = collections.defaultdict(list)
for e in ker:
    if cat(e["name"]) == "comm": cm[(gid(e), e["name"][:100])].append(e["dur"])
for (g, n), ds in sorted(cm.items()):
    ds.sort(); print(f"  g{g}: {len(ds)/nsteps:6.1f}/step median {ds[len(ds)//2]:6.1f} us  {n}")
