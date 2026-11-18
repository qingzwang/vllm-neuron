"""Figures for docs/DECODE_BLOCK_PROFILE.md (matplotlib, Chinese labels, PNG).

python plot_decode_block.py [out dir, default docs/decode_block_profile]

Needs matplotlib and a CJK font (Noto Sans CJK SC: `apt install fonts-noto-cjk`; WenQuanYi Zen Hei
is the fallback). Inputs: the per-layer medians below (results/profile_decode_block/block_c*.txt,
kernel_c*.txt, dma_engines_c8.txt) and the one-layer timelines
results/profile_decode_block/timeline_c*_layer40.json (scripts/profile_timeline.py).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "profile_decode_block")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "docs", "decode_block_profile")

# Light palette (validated categorical slots 1-3; chrome and ink tokens).
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, BAND = "#2a78d6", "#eb6834", "#1baf7a", "#f0efec"
COL = {"attention": BLUE, "all-reduce": ORANGE, "MoE": AQUA}

plt.rcParams.update({
    "font.family": ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
    "font.size": 10.5, "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 10,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})


def titles(fig, title, sub):
    fig.text(0.012, 0.975, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.915, sub, fontsize=10, color=INK2, va="top")


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def legend(fig, labels, colors, x, y):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y), ncol=len(labels), frameon=False,
               fontsize=10, handlelength=1.0, handleheight=1.0)


# ---- 1: one layer, measured vs the HBM bound -----------------------------------------------------
LAYER = [  # results/profile_decode_block/block_c*.txt; bounds as in the doc
    ("1 个请求  实测", [("attention", 173.9), ("all-reduce", 23.6), ("MoE", 172.0), ("all-reduce", 24.8)]),
    ("1 个请求  带宽下限", [("attention", 34), ("all-reduce", 22), ("MoE", 85), ("all-reduce", 22)]),
    ("8 个请求  实测", [("attention", 508.8), ("all-reduce", 27.9), ("MoE", 171.3), ("all-reduce", 29.0)]),
    ("8 个请求  带宽下限", [("attention", 249), ("all-reduce", 27), ("MoE", 85), ("all-reduce", 27)]),
]


def fig_layer():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.subplots_adjust(left=0.17, right=0.93, top=0.72, bottom=0.14)
    titles(fig, "一层 decode：实测 vs HBM 带宽下限",
           "第 1-79 层中位数，rank 0。下限 = 这一层必须搬的字节 ÷ 716 GB/s；all-reduce 下限 = nccom-test 实测延迟")
    ys = [3.3, 2.5, 1.0, 0.2]
    for (label, parts), y in zip(LAYER, ys):
        x = 0
        for p, v in parts:
            ax.barh(y, v - 2, left=x, height=0.5, color=COL[p], linewidth=0)
            if v > 40:
                ax.text(x + v / 2, y, f"{v:.0f}", ha="center", va="center", color="white", fontsize=9.5)
            x += v
        ax.text(x + 8, y, f"{x:.0f} µs", va="center", color=INK, fontsize=10)
    ax.set_yticks(ys, [l for l, _ in LAYER])
    for t, (l, _) in zip(ax.get_yticklabels(), LAYER):
        t.set_color(INK if "实测" in l else INK2)
    ax.set_xlim(0, 820); ax.set_ylim(-0.4, 3.8)
    ax.set_xlabel("每层 µs")
    ax.tick_params(axis="y", length=0)
    xgrid(ax)
    legend(fig, ["attention", "all-reduce", "MoE"], [BLUE, ORANGE, AQUA], 0.008, 0.85)
    save(fig, "layer_vs_bound.png")


# ---- 2: one layer's engine timeline --------------------------------------------------------------
LANES = [("Tensor", "TensorE"), ("Vector", "VectorE"), ("Scalar", "ScalarE"), ("GpSimd", "GpSimd"),
         ("Sync", "Sync")]  # no DMA lane: tiny completion packets keep "any queue active" true almost always


def fig_timeline(tag, who):
    tl = json.load(open(os.path.join(RES, f"timeline_{tag}_layer40.json")))
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.10, hspace=0.12)
    titles(fig, f"一层在设备上的时间线：{who}（{tl['span_ns'] / 1e3:.0f} µs）",
           "第 40 层，rank 0。色块 = 该引擎在执行指令（<50 ns 的空隙合并）；下图 = DMA 吞吐（5 µs 一格）")
    us = lambda ns: ns / 1e3
    ph = tl["phases"]
    for k in ("attn_AR", "moe_AR"):
        for a in (ax, bx):
            a.axvspan(us(ph[k][0]), us(ph[k][1]), color=BAND, zorder=0, linewidth=0)
    rows = LANES + [("AR", "all-reduce")]
    for k, n in (("attention", "attention"), ("attn_AR", "all-reduce"), ("moe", "MoE"), ("moe_AR", "all-reduce")):
        ax.text(us(sum(ph[k]) / 2), len(rows) - 0.45, n, ha="center", va="bottom", color=MUTED, fontsize=9)
    for i, (key, _) in enumerate(rows):
        y = len(rows) - 1 - i
        if key == "AR":
            iv, col = [ph["attn_AR"], ph["moe_AR"]], ORANGE
        else:
            iv, col = tl["lanes"][key], BLUE
        ax.broken_barh([(us(a), max(us(b - a), 0.4)) for a, b in iv], (y - 0.32, 0.64), color=col, linewidth=0)
    ax.set_yticks(range(len(rows)), [l for _, l in rows][::-1])
    ax.set_ylim(-0.6, len(rows) + 0.1)
    ax.tick_params(axis="both", length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    pts = tl["dma_gbps_5us"]
    bx.plot([i * 5 + 2.5 for i in range(len(pts))], pts, color=AQUA, linewidth=1.8, solid_joinstyle="round")
    bx.axhline(716, color=AXIS, linewidth=1.2)
    bx.text(745, 730, "HBM 716 GB/s", ha="right", va="bottom", color=MUTED, fontsize=9)
    bx.set_ylim(0, 820); bx.set_yticks([0, 200, 400, 600, 800])
    bx.set_ylabel("DMA GB/s", fontsize=9.5)
    bx.yaxis.grid(True, color=GRID, linewidth=0.8); bx.set_axisbelow(True)
    bx.set_xlim(0, 750); bx.set_xlabel("距上一层结束的时间（µs）")
    legend(fig, ["引擎在执行指令", "all-reduce", "DMA 吞吐"], [BLUE, ORANGE, AQUA], 0.14, 0.87)
    save(fig, f"timeline_{tag}.png")


# ---- 3: attention time vs requests ---------------------------------------------------------------
def fig_attention():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.subplots_adjust(left=0.10, right=0.80, top=0.80, bottom=0.14)
    titles(fig, "attention 每层：约 126 µs 固定 + 每个请求 48 µs",
           "attention 阶段（norm、QKV、RoPE、写 KV、attention、O 投影），在 1 和 8 个请求处实测，直线为拟合")
    r = [0, 8]
    fit = lambda x: 126.1 + 47.8 * x
    bound = lambda x: 2.9 + 30.8 * x  # QKV/O weights 2.1 MB + 22 MB KV per request at 716 GB/s
    ax.plot(r, [fit(x) for x in r], color=BLUE, linewidth=2, solid_capstyle="round")
    ax.plot(r, [bound(x) for x in r], color=MUTED, linewidth=2, solid_capstyle="round")
    ax.scatter([1, 8], [173.9, 508.8], s=55, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3, clip_on=False)
    ax.text(8.15, fit(8), "实测 509 µs", va="center", color=INK)
    ax.text(8.15, bound(8), "带宽下限 249 µs", va="center", color=INK)
    ax.text(1.15, 140, "1 个请求 174 µs（下限 34）", color=INK)
    ax.text(0.1, 100, "固定 126 µs", color=MUTED, fontsize=9.5)
    ax.set_xlim(0, 8); ax.set_ylim(0, 600); ax.set_xticks(range(9))
    ax.set_xlabel("同时 decode 的请求数（每个请求每步验证 4 个 token）")
    ax.set_ylabel("每层 µs")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    save(fig, "attention_vs_requests.png")


# ---- 4: why the DMA stops near 400 GB/s ----------------------------------------------------------
def fig_dma():
    rows = [  # dma_engines_c8.txt
        ("attention_tkg", "HBM 峰值", 716), ("attention_tkg", "33 个 DMA 引擎一直忙（每个 17.5 GB/s）", 578),
        ("attention_tkg", "实测：引擎只有 66% 的时间在忙", 384),
        ("moe_tkg_fused", "HBM 峰值", 716), ("moe_tkg_fused", "33 个 DMA 引擎一直忙（每个 19.9 GB/s）", 657),
        ("moe_tkg_fused", "实测：引擎只有 51% 的时间在忙", 339)]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    fig.subplots_adjust(left=0.40, right=0.93, top=0.76, bottom=0.13)
    titles(fig, "两个大 kernel 为什么只有约 350-400 GB/s，而不是 716",
           "8 个请求，rank 0。数据都是 2-4 KB 的小包，描述符由 GpSimd 在运行时生成（software DGE）")
    ys = [5.6, 4.8, 4.0, 2.2, 1.4, 0.6]
    for (k, step, v), y in zip(rows, ys):
        meas = step.startswith("实测")
        ax.barh(y, v, height=0.5, color=BLUE if meas else AXIS, linewidth=0)
        ax.text(v + 8, y, f"{v}", va="center", color=INK)
    ax.set_yticks(ys, [s for _, s, _ in rows])
    for t, (_, s, _) in zip(ax.get_yticklabels(), rows):
        t.set_color(INK if s.startswith("实测") else INK2)
    for k, y in (("attention_tkg", 6.3), ("moe_tkg_fused", 2.9)):
        ax.text(-0.72, y, k, transform=ax.get_yaxis_transform(), fontweight="bold", color=INK, va="center")
    ax.set_xlim(0, 800); ax.set_ylim(0.1, 6.7)
    ax.set_xlabel("每个逻辑核 GB/s")
    ax.tick_params(axis="y", length=0)
    xgrid(ax)
    save(fig, "dma_ceiling.png")


# ---- 5: DMA microbenchmark (scripts/dma_bench.py) ------------------------------------------------
def fig_dma_bench():
    rows = {}
    for line in open(os.path.join(RES, "..", "dma_bench", "hbm_to_sbuf_lnc2.txt")):
        f = line.split()
        if len(f) == 6 and f[0] in ("none", "hwdge", "swdge", "gather"):
            rows.setdefault(f[0], []).append((int(f[1]), float(f[4])))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.74, bottom=0.14)
    titles(fig, "DMA 带宽 vs 每行字节数（HBM → SBUF，一个逻辑核）",
           "每次 DMA 搬 [128 个 partition × 每行 F 字节]；gather = 每个 partition 按 SBUF 里的索引取一行（paged KV 的方式）")
    style = {"none": (AXIS, "静态描述符"), "hwdge": (ORANGE, "硬件 DGE"), "swdge": (AQUA, "GpSimd 软件 DGE"),
             "gather": (BLUE, "GpSimd gather")}
    for mode, (col, name) in style.items():
        xs, ys = zip(*rows[mode])
        ax.plot(xs, ys, color=col, linewidth=2.2 if mode == "gather" else 1.8, marker="o", markersize=5,
                markeredgecolor=SURF, markeredgewidth=1.5)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=c, linewidth=2, marker="o", markersize=5) for c, _ in style.values()],
               [n for _, n in style.values()], loc="upper left", bbox_to_anchor=(0.09, 0.85), ncol=4, frameon=False, fontsize=10)
    ax.axhline(716, color=AXIS, linewidth=1)
    ax.text(512, 724, "HBM 716 GB/s", color=MUTED, fontsize=9, va="bottom")
    ax.axvspan(1800, 4400, color=BAND, zorder=0, linewidth=0)
    ax.text(2800, 40, "Hy3 现在的\n2-4 KB 包", ha="center", color=MUTED, fontsize=9)
    ax.annotate("gather 2 KB：348 GB/s", xy=(2048, 347.6), xytext=(5200, 330), color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192, 16384, 32768], ["512", "1K", "2K", "4K", "8K", "16K", "32K"])
    ax.set_xlim(420, 40000); ax.set_ylim(0, 800)
    ax.set_xlabel("每行（每个 partition）连续字节数"); ax.set_ylabel("GB/s")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    save(fig, "dma_bench.png")


if __name__ == "__main__":
    fig_dma_bench()
    fig_layer()
    fig_timeline("c1", "1 个请求")
    fig_timeline("c8", "8 个请求")
    fig_attention()
    fig_dma()
