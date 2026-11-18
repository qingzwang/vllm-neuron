"""Figures for results/h100_profile/README.md.  python plot_figs.py  (matplotlib, a CJK font such as Noto Sans CJK)
H100 numbers are read from data/; Trn2 numbers are quoted from the hy_v3 README (sources noted inline)."""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, FIGS = os.path.join(HERE, "data"), os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

# ---- style: reference palette (light), validated with the dataviz validator (adjacent pairs, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"   # categorical slots 1-5, fixed order
NEUTRAL, NEUTRAL2 = "#c3c2b7", "#e1e0d9"                                      # 'other' / 'idle' (not series hues)
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f.replace(" ", "") and "Regular" in f:
        font_manager.fontManager.addfont(f)
cjk = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
plt.rcParams.update({
    "font.family": cjk[:1] + ["DejaVu Sans"], "font.size": 10.5, "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "legend.frameon": False, "legend.fontsize": 9.5,
    "lines.linewidth": 2, "savefig.dpi": 160,
})

def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight", pad_inches=0.2); plt.close(fig)

def note(fig, text, y=-0.02):
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8.5, color=MUTED, transform=fig.transFigure)

# ---- data
def sweep(cfg):
    out = {}
    for C in (1, 4, 8, 16):
        d = json.load(open(f"{DATA}/decode_sweep/{cfg}/c{C}.json"))
        out[C] = dict(itl=d["median_itl_ms"], tpot=d["mean_tpot_ms"], tps=d["output_throughput"], mtp=d["spec_tokens_per_step"])
    return out
SW = {"A": sweep("A"), "B": sweep("B")}

def nccl():
    txt = open(f"{DATA}/nccl/nccl_h100.txt").read(); secs = re.split(r"=== (\w+) \((\w+)\) ===", txt); r = {}
    for i in range(1, len(secs), 3):
        for l in secs[i + 2].splitlines():
            f = l.split()
            if len(f) >= 13 and f[0].isdigit(): r.setdefault((secs[i], secs[i + 1]), {})[int(f[0])] = float(f[5])
    return r
NC = nccl()

def detail(cfg):  # refined categories, ms/step, from prof_detail.py output (rank0)
    t = open(f"{DATA}/prof_decode/{cfg}/detail_rank0.txt").read()
    ref = {m.group(1).strip(): float(m.group(2)) for m in re.finditer(r"^  refined  (.+?)\s+([0-9.]+)$", t, re.M)}
    win = float(re.search(r"window/steps ([0-9.]+) ms", t).group(1))
    graphs = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"^\s+(\d+)\s+\d+\s+[0-9.]+\s+\S+\s+\S+\s+([0-9.]+)  ", t, re.M)]
    return ref, win, graphs
def busy_idle(cfg):  # GPU idle ms/step from prof_summary rank0 line
    t = open(f"{DATA}/prof_decode/{cfg}/summary.txt").read()
    m = re.search(r"window ([0-9.]+) ms \| GPU busy ([0-9.]+) ms .*? steps (\d+)", t)
    return (float(m.group(1)) - float(m.group(2))) / int(m.group(3))

# ======================= Fig 1: decode step / TPOT vs concurrency
Cs = [1, 4, 8, 16]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
for ax, key, title in [(axs[0], "itl", "decode step(ITL 中位数)"), (axs[1], "tpot", "TPOT 均值")]:
    for cfg, col, lab in [("A", S1, "A:BF16 KV,无 MTP"), ("B", S2, "B:FP8 KV + MTP k=3")]:
        ys = [SW[cfg][c][key] for c in Cs]
        ax.plot(Cs, ys, color=col, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=lab, zorder=3)
        for c, y in zip(Cs, ys):
            below = (cfg == "A") or key == "tpot"
            ax.annotate(f"{y:.1f}", (c, y), textcoords="offset points", xytext=(0, -15 if below else 8), ha="center", fontsize=8.5, color=INK2)
    ax.set_title(title); ax.set_xlabel("并发 C"); ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(Cs)); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticklabels([str(c) for c in Cs])
    ax.set_ylim(0, 48); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("ms")
# Trn2 reference: README "Trn2 vs H100" / MTP section, D-only C=8 TPOT 19.71 ms, step ~44 ms (same benchmark)
axs[0].scatter([8], [44], marker="D", s=60, color=INK2, zorder=4)
axs[0].annotate("Trn2 ≈44", (8, 44), textcoords="offset points", xytext=(-10, -4), ha="right", fontsize=8.5, color=INK2)
axs[1].scatter([8], [19.71], marker="D", s=60, color=INK2, zorder=4, label="Trn2 1P1D,C=8(README)")
axs[1].annotate("Trn2 19.7", (8, 19.71), textcoords="offset points", xytext=(-10, 4), ha="right", fontsize=8.5, color=INK2)
h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = axs[1].get_legend_handles_labels()
fig.legend(h1 + h2[-1:], l1 + l2[-1:], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
note(fig, "H100 p5.48xlarge,TP8,D-only 重放(85k 输入、92% 共享前缀、输出 1200、temperature 0.9);MTP 每步 2.24–2.28 token。"
          "Trn2 点引自 hy_v3 README 同一 benchmark 的 D-only 结果(TPOT 19.71 ms,step 约 44 ms,2.22 token/步)。")
save(fig, "fig1_decode_vs_concurrency.png")

# ======================= Fig 2: per-step parts, Trn2 (C=12) vs H100 B (C=8)
refB, winB, gB = detail("B")
gB = sorted(gB, key=lambda x: -x[1]); target = gB[0][1]
draft = sum(v for g, v in gB[1:] if g != 0); eager = next(v for g, v in gB if g == 0)
# H100 target-graph parts (prof_detail per-graph line for the target graph)
tline = re.search(r"^\s+%d\s+.*?  (moe .*)$" % gB[0][0], open(f"{DATA}/prof_decode/B/detail_rank0.txt").read(), re.M).group(1)
tp = {k: float(v) for k, v in re.findall(r"(\w+) ([0-9.]+)", tline)}
h100 = {"attention": tp["attention"], "MoE": tp["moe"], "层内其他\n(dense GEMM、router、norm…)": tp["gemm"] + tp["unclear"] + tp["other"] + tp["elementwise"],
        "集合通信": tp["comm"], "MTP draft + 采样": draft + eager}
# Trn2: hy_v3 README, "Decode step at C = 12": rank-0 device profile, 48.2 ms/step = target 43.6 + draft 4.6;
# target parts attention 24.3, fused MoE 15.3, other instructions 9.5, collectives 5.0 (same 85k / 92% D-only benchmark).
trn2 = {"attention": 24.3, "MoE": 15.3, "层内其他\n(dense GEMM、router、norm…)": 9.5, "集合通信": 5.0, "MTP draft + 采样": 4.6}
parts = list(h100)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
y = range(len(parts)); hgt = 0.36
for off, vals, col, lab in [(-hgt / 2 - 0.02, trn2, S2, "Trn2(trn2.48xlarge,TP64,C=12,每步 48.2 ms)"),
                            (hgt / 2 + 0.02, h100, S1, f"H100(p5,TP8,配置 B,C=8,每步 {winB:.1f} ms)")]:
    ys = [i + off for i in y]; vs = [vals[p] for p in parts]
    ax.barh(ys, vs, height=hgt, color=col, label=lab, zorder=3)
    for yy, v in zip(ys, vs):
        ax.text(v + 0.3, yy, f"{v:.1f}", va="center", fontsize=9, color=INK2)
for i, p in enumerate(parts):
    r = trn2[p] / h100[p]
    ax.text(26.6, i, f"×{r:.1f}", va="center", ha="left", fontsize=10, color=INK, fontweight="bold")
ax.text(26.6, -0.75, "Trn2/H100", fontsize=8.5, color=MUTED)
ax.set_yticks(list(y)); ax.set_yticklabels(parts); ax.invert_yaxis(); ax.set_xlim(0, 26)
ax.set_xlabel("每个 decode step 的 ms(rank0 设备 profile)"); ax.grid(axis="y", visible=False)
ax.set_title("每一步时间花在哪:Trn2 vs H100(同一 85k D-only benchmark)")
ax.legend(loc="lower right")
note(fig, "Trn2 数字引自 hy_v3 README「Decode step at C = 12」(target 各部分之和不等于 43.6 ms,原文如此);H100 为本次 torch profiler,target 一个 CUDA graph。\n"
          "两边 C 不同(12 vs 8);H100 在 C=16 时每步也只有 33.7 ms。两边每步都产出约 2.2 个 token。")
save(fig, "fig2_step_parts_trn2_vs_h100.png")

# ======================= Fig 3: H100 per-step composition, A vs B
def comp(cfg):
    ref, win, _ = detail(cfg)
    g = lambda k: sum(v for n, v in ref.items() if n.startswith(k))
    return win, [("attention", g("attention"), S1), ("MoE", g("moe"), S2), ("集合通信", g("comm"), S3),
                 ("dense GEMM", g("gemm"), S4),
                 ("其他 kernel", g("unclear") + g("elementwise") + g("other") + g("sampling"), NEUTRAL),
                 ("GPU 空闲", busy_idle(cfg), NEUTRAL2)]
fig, ax = plt.subplots(figsize=(10, 3.4))
rows = [("A  BF16 KV,无 MTP\n每步 8 个 token", comp("A")), ("B  FP8 KV + MTP k=3\ntarget 每步 32 个 token", comp("B"))]
for i, (lab, (win, segs)) in enumerate(rows):
    left = 0
    for name, v, col in segs:
        ax.barh(i, v, left=left, height=0.55, color=col, edgecolor=SURFACE, linewidth=2, zorder=3,
                hatch="////" if name == "GPU 空闲" else None, label=name if i == 0 else None)
        if v >= 1.4:
            ax.text(left + v / 2, i, (f"{name}\n" if v >= 4 else "") + f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if col in (S1, S2) else INK)
        left += v
    ax.plot([win, win], [i - 0.36, i + 0.36], color=INK, linewidth=2, zorder=4)
    ax.text(win + 0.25, i + 0.33, f"实际每步 {win:.1f} ms", ha="left", va="center", fontsize=8.5, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(0, 31); ax.set_xlabel("ms / step(C=8,rank0)"); ax.grid(axis="y", visible=False)
ax.set_title("H100 decode 一步的构成:MoE 和 attention 占七成,通信约一成", pad=30)
ax.legend(loc="lower left", ncol=6, bbox_to_anchor=(0, 1.0), handlelength=1.4, columnspacing=1.2)
note(fig, "分类:MoE 包含 flashinfer CUTLASS 融合 MoE 的全部 kernel(grouped GEMM、permute、activation、topk);"
          "「其他 kernel」含疑似 router 的 FP32 sgemm、Inductor 融合 kernel、norm 等。\n"
          "条长是 kernel 时间之和 + 空闲,略长于实际步长,因为有多 stream 并行;comm 的 AR kernel 已融合 RMSNorm(+FP8 量化)。", y=-0.08)
save(fig, "fig3_h100_step_composition.png")

# ======================= Fig 4: All-Reduce latency vs size
sizes = sorted(NC[("all_reduce", "normal")])
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(sizes, [NC[("all_reduce", "normal")][s] for s in sizes], color=S1, linestyle=(0, (4, 2)), label="H100 8 卡 NCCL(nccl-tests)")
ax.plot(sizes, [NC[("all_reduce", "cudagraph")][s] for s in sizes], color=S1, label="H100 8 卡 NCCL,CUDA Graph(-G 20)")
# Trn2 nccom-test, single trn2.48xlarge, BF16 AR (task brief)
t64 = {16 << 10: 22.2, 32 << 10: 22.4, 64 << 10: 22.8, 256 << 10: 27.2, 512 << 10: 34.8, 2 << 20: 78.0, 16 << 20: 259.6, 64 << 20: 657.2, 256 << 20: 1726.4}
t8 = {16 << 10: 10.3, 32 << 10: 14.0, 64 << 10: 18.0, 256 << 10: 24.8, 512 << 10: 29.2, 2 << 20: 50.6}
ax.plot(list(t64), list(t64.values()), color=S2, label="Trn2 64 rank(nccom-test)")
ax.plot(list(t8), list(t8.values()), color=S2, linestyle=(0, (4, 2)), label="Trn2 8 rank(nccom-test)")
# measured in the decode traces: flashinfer trtllm one-shot lamport AR (+RMSNorm), median us
vl = [(64 << 10, 9.9, "A:64 KB"), (256 << 10, 14.1, "B:256 KB")]
ax.scatter([s for s, _, _ in vl], [v for _, v, _ in vl], s=80, color=S3, edgecolor=SURFACE, linewidth=2, zorder=5,
           label="vLLM decode 实际 AR(trtllm 融合 kernel,trace 中位数)")
for s, v, t in vl:
    ax.annotate(f"{v} µs", (s, v), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=INK)
ax.axvspan(48 << 10, 340 << 10, color=GRID, alpha=0.5, zorder=0)
ax.text(128 << 10, 2700, "decode 实际消息大小", ha="center", va="top", fontsize=8.5, color=INK2)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ticks = [16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20]
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([f"{t >> 20} MB" if t >= 1 << 20 else f"{t >> 10} KB" for t in ticks])
ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 500, 1000, 2000])); ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500", "1000", "2000"]); ax.set_ylim(5, 3000)
ax.set_xlabel("消息大小(BF16)"); ax.set_ylabel("单次 All-Reduce 耗时(µs)")
ax.set_title("单次 All-Reduce:小消息下链路差距不大,软件路径差距大")
ax.legend(loc="lower right", fontsize=8.8)
note(fig, "H100 小消息下,普通启动的 NCCL 约 50 µs(主要是启动和同步开销);decode 里 vLLM 用 flashinfer trtllm one-shot kernel,并把残差加 + RMSNorm(+FP8 量化)融合进同一个 kernel。\n"
          "每步次数:H100 target 161 次 AR + draft 9 次 + 4 次 AllGather;Trn2(任务说明提供的 32k profile)每步 437 次集合通信,256 KB AR 中位数 39.5 µs。")
save(fig, "fig4_allreduce_latency.png")

# ======================= Fig 5: effective HBM bandwidth of the two big kernels
ctx = 85015 + 600              # prompt + about half of the 1200 generated tokens
kv_tok_layer = {"A": 2 * 128 * 2, "B": 2 * 128 * 1}   # bytes per token per layer per GPU (1 KV head per GPU at TP8)
fa_us = {"A": 67.0, "B": 71.7}                       # FA3 decode kernel, target graph, avg us/call (trace)
moe_us = {"A": 66.3, "B": 122.7}                     # grouped GEMMs per MoE layer, target graph (trace)
toks = {"A": 8, "B": 32}
exp_bytes = 3 * 4096 * 1536 / 8                     # FP8 gate_up + down of one expert, 1/8 per GPU (TP8)
def experts(t): return 192 * (1 - (1 - 8 / 192) ** t)  # expected distinct experts, uniform routing
bw = {}
for c in ("A", "B"):
    bw[("attention", c)] = 8 * ctx * kv_tok_layer[c] / (fa_us[c] * 1e-6) / 1e12
    bw[("MoE", c)] = experts(toks[c]) * exp_bytes / (moe_us[c] * 1e-6) / 1e12
hbm = float(re.search(r"GPU0 D2D copy: ([0-9.]+)", open(f"{DATA}/hbm/hbm_h100.txt").read()).group(1))
fig, ax = plt.subplots(figsize=(9, 4))
groups = [("attention", "attention(FA3 decode)"), ("MoE", "MoE grouped GEMM(估算)")]
w = 0.34
for j, (c, col, lab) in enumerate([("A", S1, "A:BF16 KV,8 token/步"), ("B", S2, "B:FP8 KV,32 token/步")]):
    xs = [i + (j - 0.5) * (w + 0.04) for i in range(len(groups))]
    vs = [bw[(g, c)] for g, _ in groups]
    ax.bar(xs, vs, width=w, color=col, label=lab, zorder=3)
    for x, v in zip(xs, vs):
        ax.text(x, v - 0.12, f"{v:.1f}", ha="center", va="top", fontsize=10, color="#ffffff", fontweight="bold")
ax.axhline(hbm, color=INK, linewidth=1.5, linestyle=(0, (4, 2)), zorder=4)
ax.text(1.62, hbm + 0.1, f"实测 HBM 峰值 {hbm:.2f} TB/s(D2D copy)", ha="right", fontsize=9, color=INK)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[1] for g in groups]); ax.set_xlim(-0.6, 1.65)
ax.set_ylabel("等效带宽(TB/s,每卡)"); ax.set_ylim(0, 6); ax.grid(axis="x", visible=False)
ax.set_title("H100 两个大头 kernel 的等效带宽")
ax.legend(loc="upper right")
note(fig, "attention:按每个序列独立读完整 KV 计算(8 序列 × 约 85.6k token,每卡 1 个 KV head)。A 超过 HBM 峰值,"
          "说明 8 个请求共享的 78k 前缀 KV 在 L2 里被复用(推断),\n这个 benchmark 对 H100 的 attention 偏有利。"
          "MoE:按均匀路由估算每层用到的不同专家数(8 token→55 个,32 token→143 个),每个专家 FP8 权重的 1/8 约 2.36 MB。")
save(fig, "fig5_effective_bandwidth.png")
print({k: round(v, 2) for k, v in bw.items()}, "hbm", hbm, "target", target, "draft", draft, "eager", eager)
