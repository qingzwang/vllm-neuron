"""MFU / HFU / MBU of Hy3-FP8 decode on H100 (this run) and Trn2 (quoted), from measured step times.
python utilization.py  ->  data/utilization.json + a markdown table on stdout.

Definitions (per decode step, whole node):
  MFU = useful model FLOPs / (step time x peak FLOPs).  Useful = output tokens (accepted with MTP) x the dense
        model's FLOPs per token (linear 2 x active params + attention 4 x n_heads x head_dim x context x layers).
  HFU = executed FLOPs / (step time x peak FLOPs).  Executed = every token the target verifies (C x (1+k) with MTP)
        + the MTP draft passes + work replicated across ranks (Trn2: router x64, shared expert x8; lower bound).
  MBU = model-required HBM bytes / (step time x peak HBM bandwidth).  Required = weights the step must read once
        (attention, dense layer, lm_head, router, the routed experts some token picked + shared expert) + every
        request's KV for its own full context, without counting replication or cache reuse.
  MBU_dedup = the same with the 78,336-token shared prefix's KV counted once (the benchmark's requests share it, and at
        BF16 one layer's prefix KV per GPU, 40 MB, fits in H100's 50 MB L2). Real HBM traffic lies between the two.
  intensity = executed FLOPs / required bytes; the ridge point is peak FLOPs / peak bandwidth.
  Trn2 'hardware bytes' (DECODE_BLOCK_PROFILE.md DMA bytes) is the bandwidth analogue of HFU: bytes actually moved.
Peaks are spec values: H100 SXM FP8 dense 1979 TFLOPS, HBM3 3.35 TB/s; Trn2 chip FP8 dense 1299 TFLOPS,
716 GB/s per logical core (64 per trn2.48xlarge, the figure DECODE_BLOCK_PROFILE.md uses).
Assumptions: uniform routing for the distinct experts (real routing is more concentrated -> fewer bytes, so MBU is an
upper estimate); context 85,615 (85,015 prompt + ~600 generated, the mean over a 1200-token decode)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
cfg = json.load(open(os.path.join(DATA, "hy3_config_subset.json")))
H, L, NH, NKV, HD = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
E, TOPK, I, ID, V = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"], cfg["intermediate_size"], cfg["vocab_size"]
LM = L - cfg["first_k_dense_replace"]                    # 79 MoE layers
P_ATT = H * (NH * HD + 2 * NKV * HD) + NH * HD * H        # QKV + O
P_EXP = 3 * H * I; P_ROUTER = H * E; P_DENSE = 3 * H * ID; P_LM = V * H; P_EH = 2 * H * H
CTX = 85015 + 600
PREFIX = 78336

def distinct(t):                                          # expected distinct routed experts for t tokens, uniform routing
    return E * (1 - (1 - TOPK / E) ** t)

def flops_tok(layers_att, layers_moe, dense, lm, eh=0):   # FLOPs for one token through the given parts
    lin = 2 * (layers_att * P_ATT + layers_moe * ((TOPK + 1) * P_EXP + P_ROUTER) + dense * P_DENSE + lm * P_LM + eh * P_EH)
    att = layers_att * 4 * NH * HD * CTX
    return lin, att
F_LIN, F_ATT = flops_tok(L, LM, 1, 1)                     # target model, per token
D_LIN, D_ATT = flops_tok(1, 1, 0, 1, eh=1)                # one MTP draft pass, per token

def kv_bytes(C, layers, kvb, dedup=False):
    toks = (PREFIX + C * (CTX - PREFIX)) if dedup else C * CTX
    return toks * layers * 2 * NKV * HD * kvb
def w_bytes_target(t):                                    # FP8 weights 1 B, lm_head / router BF16 2 B
    return L * P_ATT + P_DENSE + 2 * P_LM + LM * (2 * P_ROUTER + (distinct(t) + 1) * P_EXP)
def w_bytes_draft(t): return P_ATT + 2 * P_ROUTER + (distinct(t) + 1) * P_EXP + P_EH + 2 * P_LM

def step(C, k, kvb, dedup=False):                         # FLOPs and required bytes of one decode step at concurrency C, MTP k
    T = C * (1 + k)
    passes = [T] + [C] * (k - 1) if k else []             # draft: one pass over all verified positions, then k-1 of C tokens
    exe = T * (F_LIN + F_ATT) + sum(p * (D_LIN + D_ATT) for p in passes)
    byt = w_bytes_target(T) + kv_bytes(C, L, kvb, dedup) + sum(w_bytes_draft(p) + kv_bytes(C, 1, kvb, dedup) for p in passes)
    return exe, byt

H100 = dict(n=8, flops=1979e12, bw=3.35e12)
TRN2 = dict(n=64, flops=1299e12 / 4, bw=716e9)            # per logical core
rows = []
def add(name, plat, C, k, kvb, step_s, useful_tok, extra_flops=0.0, hw_bytes=None, note=""):
    exe, byt = step(C, k, kvb); exe += extra_flops; byt_d = step(C, k, kvb, True)[1]
    pf, pb = plat["n"] * plat["flops"], plat["n"] * plat["bw"]
    r = dict(name=name, C=C, mtp_k=k, step_ms=step_s * 1e3, useful_tokens_per_step=useful_tok,
             useful_tflops=useful_tok * (F_LIN + F_ATT) / step_s / 1e12, executed_tflops=exe / step_s / 1e12,
             required_GB=byt / 1e9, required_TBps=byt / step_s / 1e12,
             MFU=useful_tok * (F_LIN + F_ATT) / step_s / pf, HFU=exe / step_s / pf, MBU=byt / step_s / pb,
             MBU_dedup=byt_d / step_s / pb, required_GB_dedup=byt_d / 1e9, intensity=exe / byt, ridge=plat["flops"] / plat["bw"], note=note)
    if hw_bytes is not None: r.update(hw_GB=hw_bytes / 1e9, hw_util=hw_bytes / step_s / pb)
    rows.append(r)

# ---- H100, this run: step = ITL median; useful tokens per step = C x MTP tokens/step (bench JSON + /metrics)
for cfgname, k, kvb in (("A", 0, 2), ("B", 3, 1)):
    for C in (1, 4, 8, 16):
        d = json.load(open(os.path.join(DATA, "decode_sweep", cfgname, f"c{C}.json")))
        add(f"H100 {cfgname}", H100, C, k, kvb, d["median_itl_ms"] / 1e3, C * d["spec_tokens_per_step"])

# ---- Trn2, quoted. Replicated work (docs/PROFILE_85K.md): router on every rank (x64), decode shared expert on each
# group of 8 ranks (x8). Extra executed FLOPs per verified token, a lower bound (the MoE kernel's own padding unknown).
def trn2_extra(T): return T * LM * (63 * 2 * P_ROUTER + 7 * 2 * P_EXP)
# (a) dense attention profile, DECODE_BLOCK_PROFILE.md c8: target 60.0 + draft 3.9 + gap 0.07 ms; tokens/step not reported,
#     2.2 assumed (the D-only chat replay accepts 2.19-2.23). DMA bytes: (199 MB attention + 51 MB MoE) x 80 layers per rank.
add("Trn2 dense profile", TRN2, 8, 3, 1, 63.97e-3, 8 * 2.2, trn2_extra(32), hw_bytes=64 * 80 * 250e6,
    note="DECODE_BLOCK_PROFILE.md c8; 2.2 tokens/step assumed; hw bytes = target DMA bytes only")
# (b) production 1P1D D-only (README 'Trn2 vs H100'): TPOT 19.71 ms, 2.22 tokens/step -> step 43.8 ms; sparse decode
#     reads ~1/8 of the KV, so the dense 'required bytes' do not apply: MBU not reported.
add("Trn2 1P1D sparse", TRN2, 8, 3, 1, 19.71e-3 * 2.22, 8 * 2.22, trn2_extra(32),
    note="README 1P1D D-only, sparse decode; MBU not meaningful (sparse reads ~1/8 of the KV)")

out = dict(definitions=__doc__, model=dict(active_params_per_token=(L * P_ATT + LM * ((TOPK + 1) * P_EXP + P_ROUTER) + P_DENSE + P_LM),
           flops_per_token_linear=F_LIN, flops_per_token_attention=F_ATT, context=CTX), rows=rows)
json.dump(out, open(os.path.join(DATA, "utilization.json"), "w"), indent=1)
print(f"FLOPs/token: linear {F_LIN/1e9:.1f} G + attention {F_ATT/1e9:.1f} G (context {CTX})")
print("| 配置 | C | step (ms) | 有效 token/步 | MFU | HFU | MBU(共享前缀只算一次 ~ 每请求独立) | 需读字节/步 (GB) | 算术强度 FLOP/B(拐点) | 硬件实际搬运 |\n|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    mbu = "—" if "sparse" in r["name"] else f"{r['MBU_dedup']:.0%} ~ {r['MBU']:.0%}"
    gb = "—" if "sparse" in r["name"] else f"{r['required_GB_dedup']:.0f} ~ {r['required_GB']:.0f}"
    hw = f"{r['hw_GB']:.0f} GB,{r['hw_util']:.0%}" if "hw_GB" in r else "—"
    print(f"| {r['name']} | {r['C']} | {r['step_ms']:.1f} | {r['useful_tokens_per_step']:.1f} | {r['MFU']:.2%} | {r['HFU']:.2%} | {mbu} | {gb} | {'—' if 'sparse' in r['name'] else format(r['intensity'], '.0f')}({r['ridge']:.0f}) | {hw} |")
