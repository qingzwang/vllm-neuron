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

# ======================= Trn2 dense decode block profile, per layer (quoted)
# Source: docs/DECODE_BLOCK_PROFILE.md on branch hy3-decode-block-profile (commit 2fa3433), trn2.48xlarge TP64/EP64,
# FP8 KV, MTP k=3, fused decode MoE, dense attention, c8 (8 requests x 4 verify tokens), rank 0, layers 1-79 median.
TRN2_LAYER = {"attention": 509, "ar1": 28, "moe": 171, "ar2": 29}      # us per layer, measured (medians)
TRN2_LAYER_TOTAL = 739                                                 # us, the report's layer median
TRN2_BW = 716                                                          # GB/s, one logical core's HBM stack
TRN2_KV_MB = 8 * 86016 * 256 / 1e6                                     # 8 requests x 22 MB (1 replicated KV head)
TRN2_MOE_MB = 56.6 + 2.4 + 1.6                                         # 3 routed experts + shared-expert shard + router
TRN2_ATT_KERNEL_US, TRN2_MOE_US = 429, 171                             # attention_tkg kernel; MoE block
H100_BW = hbm * 1000                                                   # GB/s, measured D2D
H100_KV_MB = 8 * ctx * 256 / 1e6                                       # 8 requests, 1 KV head per GPU, FP8 K+V
H100_MOE_MB = experts(32) * exp_bytes / 1e6                            # estimate, uniform routing
h_att = tp["attention"] / 80 * 1e3; h_moe = tp["moe"] / 79 * 1e3
h_comm = tp["comm"] / 80 * 1e3; h_oth = (tp["gemm"] + tp["unclear"] + tp["other"] + tp["elementwise"]) / 80 * 1e3

# Fig 6: one layer, stacked in execution order
fig, ax = plt.subplots(figsize=(10, 3.3))
rows = [("Trn2 每个 rank\n(dense attention)", [("attention block", TRN2_LAYER["attention"], S1), ("AR", TRN2_LAYER["ar1"], S3),
                                                 ("MoE", TRN2_LAYER["moe"], S2), ("AR ", TRN2_LAYER["ar2"], S3)]),
        ("H100 每张 GPU\n(配置 B target)", [("attention", h_att, S1), ("层内其他", h_oth, S4), ("MoE", h_moe, S2), ("AR(2 次,融合 norm)", h_comm, S3)])]
for i, (lab, segs) in enumerate(rows):
    left = 0
    for name, v, col in segs:
        ax.barh(i, v, left=left, height=0.55, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        if v >= 45:
            ax.text(left + v / 2, i, (f"{name.strip()}\n" if v >= 100 else "") + f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if col in (S1, S2) else INK)
        left += v
    ax.text(left + 8, i, f"{TRN2_LAYER_TOTAL if i == 0 else left:.0f} µs", va="center", fontsize=10, fontweight="bold", color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(0, 820); ax.set_xlabel("µs / 层(C=8,每个请求每步 4 个 token)"); ax.grid(axis="y", visible=False)
ax.set_title("一层 decode 的时间:Trn2 慢在 attention,MoE 只差 1.2 倍")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=S1, label="attention"), Patch(color=S2, label="MoE"), Patch(color=S3, label="All-Reduce"),
                   Patch(color=S4, label="H100 层内其他(QKV/O GEMM、router、norm…)")], loc="lower right", fontsize=8.8)
note(fig, "Trn2:DECODE_BLOCK_PROFILE.md(分支 hy3-decode-block-profile @ 2fa3433),rank0,layers 1-79 中位数;attention block 含 norm、QKV、RoPE、写 KV、O-proj。\n"
          "各部分为中位数,和报告的整层中位数 739 µs 不严格相加。H100:本次 trace 的 target graph 按类别除以层数;QKV/O 等 GEMM 计在「层内其他」里。", y=-0.09)
save(fig, "fig6_layer_trn2_vs_h100.png")

# Fig 7: bytes each device must read per layer
fig, ax = plt.subplots(figsize=(9, 3.6))
cats = ["KV(8 个请求)", "MoE 权重"]
vals = {"Trn2 每个 rank": (TRN2_KV_MB, TRN2_MOE_MB), "H100 每张 GPU": (H100_KV_MB, H100_MOE_MB)}
for j, (lab, col) in enumerate([("Trn2 每个 rank", S2), ("H100 每张 GPU", S1)]):
    ys = [i + (j - 0.5) * 0.4 for i in range(2)]
    ax.barh(ys, vals[lab], height=0.36, color=col, label=lab + (f"(HBM {TRN2_BW} GB/s)" if j == 0 else f"(HBM {H100_BW:.0f} GB/s)"), zorder=3)
    for y, v in zip(ys, vals[lab]):
        ax.text(v + 4, y, f"{v:.0f} MB", va="center", fontsize=9.5, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels(cats); ax.invert_yaxis(); ax.set_xlim(0, 400)
ax.set_xlabel("每层每个设备要从 HBM 读的 MB(C=8)"); ax.grid(axis="y", visible=False)
ax.set_title("每个设备读的字节:KV 一样多,MoE 差 5.6 倍")
ax.legend(loc="upper right")
note(fig, "8 个 KV head:H100 TP8 每卡正好 1 个;Trn2 TP64 每个 rank 也是 1 个,但每个 head 被 8 个 rank 重复读(KV 复制 8 份)。MoE 在 EP64 下按专家切分,每个 rank 只有 3 个专家。\n"
          "H100 MoE 为均匀路由估算(每层约 143 个专家 × 2.36 MB);Trn2 为 DECODE_BLOCK_PROFILE.md 的 bound 口径(融合 kernel 读全部 3 个专家 + 共享专家分片 + router)。", y=-0.09)
save(fig, "fig7_bytes_per_device.png")

# Fig 8: HBM-bandwidth bound vs measured, per kernel
rows = [("Trn2 attention_tkg", TRN2_KV_MB / TRN2_BW * 1e3, TRN2_ATT_KERNEL_US, S2),
        ("H100 FA3 decode", H100_KV_MB / H100_BW * 1e3, fa_us["B"], S1),
        ("Trn2 MoE block", TRN2_MOE_MB / TRN2_BW * 1e3, TRN2_MOE_US, S2),
        ("H100 MoE grouped GEMM", H100_MOE_MB / H100_BW * 1e3, moe_us["B"], S1)]
fig, ax = plt.subplots(figsize=(9.5, 3.9))
for i, (lab, bound, meas, col) in enumerate(rows):
    ax.barh(i, meas, height=0.5, color=col, zorder=3)
    ax.plot([bound, bound], [i - 0.33, i + 0.33], color=INK, linewidth=2, zorder=4)
    ax.text(meas + 6, i, f"{meas:.0f} µs,带宽利用 {bound / meas:.0%}", va="center", fontsize=9.5, color=INK)
ax.plot([], [], color=INK, linewidth=2, label="HBM 带宽下限(字节 ÷ 带宽)")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(0, 560); ax.set_xlabel("µs / 层(C=8)"); ax.grid(axis="y", visible=False)
ax.set_title("离带宽下限有多远:H100 两个 kernel 在 81–91%,Trn2 在 49–57%")
ax.legend(loc="lower right")
note(fig, "Trn2 的损失(DECODE_BLOCK_PROFILE.md):paged KV 以 2 KB 行 gather,microbenchmark 上限 348 GB/s;descriptor 由 GpSimd 运行时生成;"
          "attention 每层约 126 µs 固定开销;MoE 的 DMA 引擎只有一半时间在工作。\nH100 的 KV block 同样是 16 token × 128 × FP8 = 每块每 head 2 KB(data/cache_config_info_B.txt),FA3 仍达 81%。")
save(fig, "fig8_bound_vs_measured.png")
print("layer us", round(h_att, 1), round(h_oth, 1), round(h_moe, 1), round(h_comm, 1), "sum", round(h_att + h_oth + h_moe + h_comm, 1),
      "| MB", round(TRN2_KV_MB, 1), round(H100_KV_MB, 1), round(TRN2_MOE_MB, 1), round(H100_MOE_MB, 1),
      "| bound us", [round(r[1], 1) for r in rows])

# ======================= Fig 9 / 10: MFU, HFU, MBU (utilization.py -> data/utilization.json)
U = json.load(open(f"{DATA}/utilization.json"))["rows"]
def urow(name, C): return next(r for r in U if r["name"] == name and r["C"] == C)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.1))
ax = axs[0]
for cfg, col, lab in [("H100 A", S1, "A:BF16 KV,无 MTP"), ("H100 B", S2, "B:FP8 KV + MTP k=3")]:
    lo = [urow(cfg, c)["MBU_dedup"] * 100 for c in Cs]; hi = [urow(cfg, c)["MBU"] * 100 for c in Cs]
    ax.fill_between(Cs, lo, hi, color=col, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(Cs, hi, color=col, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab + "(每请求独立读 KV)", zorder=3)
    ax.plot(Cs, lo, color=col, linestyle=(0, (4, 2)), linewidth=1.5, label=lab + "(共享前缀只算一次)", zorder=3)
    ax.annotate(f"{hi[2]:.0f}%", (8, hi[2]), textcoords="offset points", xytext=(-6, 7), ha="right", fontsize=8.5, color=INK2)
ax.axhline(hbm / 3.35 * 100, color=INK, linewidth=1.2, linestyle=(0, (1, 2)))
ax.text(1, hbm / 3.35 * 100 + 1.5, f"实测 D2D copy 带宽 = 规格的 {hbm / 3.35:.0%}", fontsize=8.5, color=INK2)
ax.set_title("MBU(对 3.35 TB/s 规格)"); ax.set_ylabel("%"); ax.set_ylim(0, 105)
ax2 = axs[1]
ax2.plot(Cs, [urow("H100 A", c)["MFU"] * 100 for c in Cs], color=S1, marker="o", markersize=6, markeredgecolor=SURFACE, label="A:MFU(= HFU)")
ax2.plot(Cs, [urow("H100 B", c)["MFU"] * 100 for c in Cs], color=S2, marker="o", markersize=6, markeredgecolor=SURFACE, label="B:MFU(只算被接受的 token)")
ax2.plot(Cs, [urow("H100 B", c)["HFU"] * 100 for c in Cs], color=S2, linestyle=(0, (4, 2)), marker="s", markersize=5, label="B:HFU(含被拒的 verify token 和 draft)")
for name, key in (("H100 B", "HFU"), ("H100 B", "MFU"), ("H100 A", "MFU")):
    v = urow(name, 16)[key] * 100
    ax2.annotate(f"{v:.1f}%", (16, v), textcoords="offset points", xytext=(6, -3), fontsize=8.5, color=INK2)
ax2.set_title("MFU / HFU(对 FP8 dense 1979 TFLOPS 规格)"); ax2.set_ylim(0, 4)
for a in axs:
    a.set_xscale("log", base=2); a.xaxis.set_major_locator(FixedLocator(Cs)); a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels([str(c) for c in Cs]); a.set_xlabel("并发 C"); a.grid(axis="x", visible=False)
axs[0].legend(loc="lower right", fontsize=8.0); axs[1].legend(loc="upper left", fontsize=8.3)
note(fig, "由实测 step(ITL 中位数)和 MTP 接受数,按 Hy3 config 的 FLOPs / 字节数计算(utilization.py)。每个 token 的 FLOPs:linear 40.4 G + attention 224.4 G(85.6k 上下文)。\n"
          "MBU 的两条线是区间:实际 HBM 流量在「每请求独立读 KV」和「共享前缀只读一次」之间;A 的上界接近甚至超过实测 copy 带宽,说明共享前缀的 KV 在 L2 里被复用。", y=-0.05)
save(fig, "fig9_h100_utilization_vs_c.png")

# Fig 10: C=8, H100 vs Trn2
plats = [("H100 B", "H100\n配置 B"), ("Trn2 dense profile", "Trn2\ndense profile"), ("Trn2 1P1D sparse", "Trn2 1P1D\nsparse(线上)")]
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
w = 0.36
for j, (key, col, lab) in enumerate([("MFU", S1, "MFU"), ("HFU", S2, "HFU(Trn2 为下限)")]):
    xs = [i + (j - 0.5) * (w + 0.03) for i in range(3)]; vs = [urow(n, 8)[key] * 100 for n, _ in plats]
    axs[0].bar(xs, vs, width=w, color=col, label=lab, zorder=3)
    for x, v in zip(xs, vs): axs[0].text(x, v + 0.04, f"{v:.2f}%", ha="center", fontsize=8.8, color=INK)
axs[0].set_title("算力利用率(C=8)"); axs[0].set_ylim(0, 2.6); axs[0].set_ylabel("%")
bw_items = [("MBU", S1, "MBU(模型需要的字节)"), ("hw", S2, "硬件实际搬运的字节")]
for j, (key, col, lab) in enumerate(bw_items):
    xs, vs = [], []
    for i, (n, _) in enumerate(plats):
        r = urow(n, 8)
        v = r["MBU"] if key == "MBU" else r.get("hw_util")
        if "sparse" in n or v is None: continue
        xs.append(i + (j - 0.5) * (w + 0.03)); vs.append(v * 100)
    axs[1].bar(xs, vs, width=w, color=col, label=lab, zorder=3)
    for x, v in zip(xs, vs): axs[1].text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=8.8, color=INK)
axs[1].text(2, 4, "sparse 只读约 1/8 的 KV,\n不适用 dense 口径", ha="center", fontsize=8.5, color=MUTED)
axs[1].set_title("带宽利用率(C=8,每请求独立读 KV 口径)"); axs[1].set_ylim(0, 70); axs[1].set_xlim(-0.6, 2.6)
for a in axs:
    a.set_xticks(range(3)); a.set_xticklabels([p[1] for p in plats]); a.grid(axis="x", visible=False); a.legend(loc="upper right", fontsize=8.5)
note(fig, "峰值:H100 FP8 1979 TFLOPS、3.35 TB/s;Trn2 每颗芯片 FP8 1299 TFLOPS,每个逻辑核 716 GB/s(DECODE_BLOCK_PROFILE.md 的口径)。Trn2 dense profile 的每步 token 数未报告,按 2.2 估算;"
          "\nTrn2 HFU 加上了文档写明的重复计算(router ×64、共享专家 ×8);硬件实际搬运 = DECODE_BLOCK_PROFILE.md 的 DMA 字节(含 KV 8 倍复制、3 个专家全读)。H100 没有复制,实际搬运 ≤ 模型需要。", y=-0.05)
save(fig, "fig10_utilization_trn2_vs_h100.png")
