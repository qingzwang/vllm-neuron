"""One logical core: moe_tkg_hits_batch time vs the number of hit experts, for whole experts
(I = 1536, 3 local) and TP-G chunks (I = 1536 / G, 3G local), T tokens."""
import os, sys, time
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62182")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
import torch, torch._dynamo
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.common_types import ActFnType
from vllm_neuron.envs import get_compile_backend_name
from hy_v3_neuron.kernels.moe_tkg_hits_batch import moe_tkg_hits_batch
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024
DEV, H = "neuron:0", 4096
T = int(os.environ.get("T", "4"))

def run(e_l, I, n_list, reps=48):
    g = torch.Generator().manual_seed(e_l)
    gu = (torch.randn(e_l, H, 2, I, generator=g) * 8).to(torch.float8_e4m3fn).to(DEV)
    dn = (torch.randn(e_l, I, H, generator=g) * 8).to(torch.float8_e4m3fn).to(DEV)
    gus = torch.full((e_l, 2, 1), 0.01).to(DEV); dns = torch.full((e_l, 1), 0.01).to(DEV)
    x = torch.randn(T, H, generator=g).to(torch.bfloat16).to(DEV)
    aff = torch.rand(T, e_l, generator=g).to(DEV)
    idx = torch.arange(e_l, dtype=torch.int32).view(1, e_l).to(DEV)
    ins = torch.ones((e_l, 1)).to(DEV)
    def f(x, gu, dn, aff, idx, n, gus, dns, ins):
        return wrap_nki(moe_tkg_hits_batch)[2](hidden_input=x, expert_gate_up_weights=gu, expert_down_weights=dn,
            expert_affinities=aff, expert_index=idx, num_hits=n, expert_gate_up_weights_scale=gus,
            expert_down_weights_scale=dns, expert_gate_up_input_scale=ins, expert_down_input_scale=ins,
            activation_fn=ActFnType.SiLU)
    c = torch.compile(f, backend=get_compile_backend_name(), fullgraph=True)
    out = {}
    with torch.no_grad():
        for n in n_list:
            nn_ = torch.tensor([[n]], dtype=torch.int32).to(DEV)
            args = (x, gu, dn, aff, idx, nn_, gus, dns, ins)
            for _ in range(5): c(*args).cpu()
            best = 1e9
            for _ in range(3):
                t0 = time.time()
                for i in range(reps):
                    o = c(*args)
                    if i % 8 == 7: o.cpu()
                o.cpu(); best = min(best, (time.time() - t0) / reps * 1e3)
            out[n] = best
    return out

for G, e_l, I, ns in [(1, 3, 1536, [0, 1, 2, 3]), (4, 12, 384, [0, 1, 2, 4, 8, 12]), (8, 24, 192, [0, 1, 2, 4, 8, 16, 24])]:
    r = run(e_l, I, ns)
    n_max = ns[-1]; per = (r[n_max] - r[ns[1]]) / (n_max - ns[1])
    print(f"G={G} I={I}: " + ", ".join(f"n={n}: {v:.3f}" for n, v in r.items()) + f" ms; per hit {per*1e3:.1f} us", flush=True)
