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
    fs = sorted(glob.glob(os.path.join(ld[0], "rank*.pt")), key=lambda f: int(re.search(r"rank(\d+)", f).group(1)))
    passes.append((os.path.getmtime(m), torch.cat([torch.load(f) for f in fs])))
passes.sort(key=lambda x: x[0]); G = json.load(open(gj))
rows = []
for (_, h), g in zip(passes, G):
    ids = g["ids"]; N, P = len(ids), g["prompt_len"]; h = h[:N].float()
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * nw
    lp = torch.log_softmax(hn @ head.T, -1)
    for t in range(0, N - 1):
        gl = g["prompt_logprobs"][t + 1]
        if not gl: continue
        s = sorted(gl.items(), key=lambda kv: -kv[1]["logprob"]); g1 = int(s[0][0]); gm = s[0][1]["logprob"] - s[1][1]["logprob"]
        n1 = int(lp[t].argmax()); a = ids[t + 1]; ga = gl.get(str(a), {}).get("logprob")
        rows.append((g["id"], t, t >= P - 1, g1 == n1, gm, abs(float(lp[t, a]) - ga) if ga is not None else None))
import collections
for region in (False, True):
    r = [x for x in rows if x[2] == region]
    print("completion" if region else "prompt", f"positions {len(r)} top-1 agree {sum(x[3] for x in r)/len(r):.4f}",
          f"mean|dlp| {sum(x[5] for x in r if x[5] is not None)/sum(1 for x in r if x[5] is not None):.4f}")
r = [x for x in rows if x[2]]
dis = [x for x in r if not x[3]]
print("disagreements by GPU margin:", collections.Counter(min(int(x[4] / 0.5), 6) * 0.5 for x in dis))
for lo, hi in ((0, 128), (128, 256), (256, 512), (512, 1024)):
    b = [x for x in r if lo <= x[1] < hi]
    if b: print(f"pos {lo}-{hi}: n {len(b)} agree {sum(x[3] for x in b)/len(b):.4f} mean|dlp| {sum(x[5] for x in b if x[5] is not None)/len(b):.4f}")
per = collections.defaultdict(list)
for x in r: per[x[0]].append(x)
print("worst sequences:", sorted(((round(1 - sum(y[3] for y in v)/len(v), 3), k, len(v)) for k, v in per.items()), reverse=True)[:5])
