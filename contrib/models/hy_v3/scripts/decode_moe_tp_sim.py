"""Usage: python decode_moe_tp_sim.py <expert_load.sh capture dir>

Decode MoE load under EP64 vs TP-G groups, from the expert_load.sh captures (decode passes only).

EP64: rank r owns experts 3r..3r+2 whole; a step's cost on r ~ its distinct hit experts.
TP-G: ranks in groups of G consecutive ranks; each group's 3G experts are split in G chunks
(I / G columns); each member computes one chunk of every distinct hit expert of its group.
Prints the mean over steps and layers of the busiest rank's hits (EP64, in whole experts) and
the busiest group's hits (TP-G, in chunks per member), for the T-token step modes of expert_load.py.
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_load import decode_steps

def load_decode(capture_dir):
    decode = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(os.path.join(capture_dir, "dp*", "*", "*_meta.json"))):
        fwd_dir = os.path.dirname(d)
        if os.path.basename(fwd_dir).startswith("prefill"):
            continue
        meta = json.load(open(d)); req_ids = meta.get("req_ids") or []
        if not req_ids:
            continue
        for ld in glob.glob(os.path.join(fwd_dir, "manual.moe_route.*")):
            L = int(re.search(r"moe_route\.(\d+)", ld).group(1))
            t = torch.load(os.path.join(ld, "rank0.pt")).numpy()[: len(req_ids)]
            for i, rid in enumerate(req_ids):
                decode[rid][L].append(t[i])
    return {r: {L: np.stack(v) for L, v in ls.items()} for r, ls in decode.items()}

dec = load_decode(sys.argv[1])
layers = sorted({L for v in dec.values() for L in v})
print(len(dec), "requests,", sum(len(next(iter(v.values()))) for v in dec.values()), "decode tokens, layers", layers[0], "-", layers[-1])
rng = np.random.default_rng(0)
E, R = 192, 64
for T, mode in [(4, "mtp"), (8, "batch_mtp"), (16, "batch_mtp"), (32, "batch_mtp")]:
    res = defaultdict(list)
    for L in layers[::4]:
        for s in decode_steps(dec, L, T, mode, rng):
            hit = np.unique(s.ravel())
            ep = np.bincount(hit // 3, minlength=R)
            res["ep64"].append(ep.max()); res["mean"].append(ep.mean())
            for G in (2, 4, 8):
                g = np.bincount(hit // (3 * G), minlength=R // G)
                res[f"tp{G}"].append(g.max())
    m = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"T={T:2d}: EP64 busiest {m['ep64']:.2f} experts (mean {m['mean']:.2f}); "
          + "; ".join(f"TP{G} busiest {m[f'tp{G}']:.2f} chunks = {m[f'tp{G}']/G:.2f} expert-equiv" for G in (2, 4, 8)))
