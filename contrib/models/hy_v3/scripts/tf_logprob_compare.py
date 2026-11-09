"""Neuron vs GPU target log-probs on the same teacher-forced sequences.

Neuron: captured last-layer outputs (rank shards of each prefill), final norm + lm_head on CPU (fp32).
GPU: vLLM prompt_logprobs (top-5 + the actual token). Per completion position: top-1 agreement, the
log-prob of the actual next token on both, and the GPU top-1 margin where they disagree.
Usage: python -I tf_compare.py <fp8 ckpt> <capture dir> <tf_gpu.json>
"""
import glob, json, os, re, sys
from pathlib import Path
import torch
from safetensors import safe_open
ckpt, cap, gj = sys.argv[1:4]
cfg = json.loads(Path(f"{ckpt}/config.json").read_text()); eps = cfg["rms_norm_eps"]
wm = json.loads(Path(f"{ckpt}/model.safetensors.index.json").read_text())["weight_map"]
def get(k):
    with safe_open(f"{ckpt}/{wm[k]}", "pt") as f: return f.get_tensor(k)
head = get("lm_head.weight").float(); nw = get("model.norm.weight").float()
passes = []
for m in glob.glob(os.path.join(cap, "dp*", "*", "*_meta.json")):
    d = os.path.dirname(m); meta = json.load(open(m))
    if not os.path.basename(d).startswith("prefill") or not meta.get("req_ids"): continue
    ld = glob.glob(os.path.join(d, "manual.final_hidden*"))
    if not ld: continue
    fs = sorted(glob.glob(os.path.join(ld[0], "rank*.pt")), key=lambda f: int(re.search(r"rank(\d+)", f).group(1)))
    passes.append((os.path.getmtime(m), torch.cat([torch.load(f) for f in fs])))
passes.sort(key=lambda x: x[0])
G = json.load(open(gj)); print(len(passes), "captures,", len(G), "sequences")
agree = n = 0; dl = []; flips = []
for (_, h), g in zip(passes, G):
    ids = g["ids"]; N, P = len(ids), g["prompt_len"]; h = h[:N].float()
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * nw
    lp = torch.log_softmax(hn @ head.T, -1)  # lp[t]: next-token distribution after t
    for t in range(P - 1, N - 1):
        gl = g["prompt_logprobs"][t + 1]  # GPU: distribution for token t+1
        if not gl: continue
        gtop = max(gl.items(), key=lambda kv: kv[1]["logprob"]); g1 = int(gtop[0])
        n1 = int(lp[t].argmax()); n += 1; agree += (g1 == n1)
        a = ids[t + 1]; ga = gl.get(str(a), {}).get("logprob")
        if ga is not None: dl.append(abs(float(lp[t, a]) - ga))
        if g1 != n1:
            s = sorted((v["logprob"] for v in gl.values()), reverse=True)
            flips.append(s[0] - s[1] if len(s) > 1 else 99)
dl = torch.tensor(dl)
print(f"top-1 agreement {agree / n:.4f} over {n} positions; |dlogprob(actual)| mean {dl.mean():.4f}, p90 {dl.quantile(0.9):.4f}, max {dl.max():.3f}")
if flips:
    f = torch.tensor(flips); print(f"disagreements: {len(flips)}, GPU top-1 vs top-2 margin median {f.median():.3f}, max {f.max():.3f}")
