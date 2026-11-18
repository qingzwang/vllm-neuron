"""One logical NeuronCore: HBM -> SBUF DMA bandwidth by row size and descriptor mode.

Each DMA moves a [128 partitions, F bytes] tile; every partition's row is F contiguous bytes in HBM,
so F is the size of one descriptor element. Modes:
  none    descriptors built by the runtime before execution (static)
  hwdge   descriptors built by the DGE hardware block (Sync / Activation queues)
  swdge   descriptors built by GpSimd at run time
  gather  swdge with a row index per partition from SBUF (random rows: the paged-KV access)
Both physical cores of the logical core (LNC 2) read their own half of the bytes. Bandwidth is the
slope between two pass counts over the same source, so launch costs cancel. Host timing is too noisy
at these sizes: run with PROF=<dir> and time on the device.

Usage: NKI_ENABLE_TRACE_CACHE=0 PROF=<dir> python dma_bench.py [mode ...]
       PY=<python with pandas> dma_bench_export.sh <dir>
       (env: SIZES=512,2048,..., MB=64 source per logical core, REPS=2,10 passes)
"""
import os
import sys
import time

os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62182")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch._dynamo
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.envs import get_compile_backend_name

DEV = "neuron:0"
torch._dynamo.config.cache_size_limit = 512  # one compiled graph per variant
torch._dynamo.config.accumulated_cache_size_limit = 4096
P = 128
DGE = {"none": nisa.dge_mode.none, "hwdge": nisa.dge_mode.hwdge, "swdge": nisa.dge_mode.swdge,
       "gather": nisa.dge_mode.swdge}


def make(mode, F, n_dma, reps):
    nb = max(2, min(8, (160 * 1024) // F))  # SBUF ring: up to 160 KB of each partition

    @nki.jit
    def k(src, idx, tag):
        # src: [2 * n_dma * P, F] fp8 in HBM (one half per physical core); idx: [P, 1] int32 rows
        # One byte per partition of every DMA's destination is copied into `log`, so no DMA is dead
        # code (the compiler drops DMA writes nothing reads); the copies are tiny Vector ops.
        out = nl.ndarray((P, reps * n_dma), dtype=src.dtype, buffer=nl.shared_hbm)
        log = nl.ndarray((P, reps * n_dma), dtype=src.dtype, buffer=nl.sbuf)
        pid = nl.program_id(0)
        bufs = [nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf) for _ in range(nb)]
        if mode == "gather":
            idx_sb = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx_sb, src=idx)
        for j in range(reps * n_dma):
            i = j % n_dma
            row0 = (pid * n_dma + i) * P
            if mode == "gather":
                s = src.ap(pattern=[[F, P], [1, F]], offset=row0 * F,
                           vector_offset=idx_sb.ap(pattern=[[1, P], [1, 1]], offset=0), indirect_dim=0)
            else:
                s = src[nl.ds(row0, P), 0:F]
            nisa.dma_copy(dst=bufs[j % nb], src=s, dge_mode=DGE[mode])
            nisa.tensor_copy(dst=log[0:P, j:j + 1], src=bufs[j % nb][0:P, 0:1])
        nisa.dma_copy(dst=out, src=log)
        return out
    return k


def timeit(c, args, n=10):
    with torch.no_grad():
        for _ in range(3):
            c(*args).cpu()
        best = 1e9
        for _ in range(3):
            t0 = time.time()
            for _ in range(n):
                c(*args).cpu()
            best = min(best, (time.time() - t0) / n)
    return best * 1e3


sizes = [int(x) for x in os.environ.get("SIZES", "512,1024,2048,4096,8192,16384,32768").split(",")]
MB = int(os.environ.get("MB", "64"))  # source size per logical core
REPS = [int(x) for x in os.environ.get("REPS", "2,10").split(",")]  # passes over it
PROF = os.environ.get("PROF")  # device profile directory: exact device times (analyze with dma_bench_analyze.py)
modes = sys.argv[1:] or ["none", "hwdge", "swdge", "gather"]
g = torch.Generator().manual_seed(0)
runs, order = [], []
for mode in modes:
    for F in sizes:
        n_dma = max(1, MB * 2 ** 20 // 2 // (P * F))  # per physical core
        rows = 2 * n_dma * P
        src = torch.randint(-100, 100, (rows, F), dtype=torch.int8, generator=g).view(torch.float8_e4m3fn).to(DEV)
        # gather: each partition reads a random row of the current window (rows of a paged block table)
        idx = torch.randint(0, P, (P, 1), dtype=torch.int32, generator=g).to(DEV)
        for r in REPS:
            tag = torch.zeros(modes.index(mode) + 1, F // 512 + 97 * r, device=DEV)  # distinct shapes per variant
            kern = make(mode, F, n_dma, r)
            c = torch.compile(lambda a, b, t, kern=kern: wrap_nki(kern)[2](a, b, t),
                              backend=get_compile_backend_name(), fullgraph=True)
            with torch.no_grad():
                c(src, idx, tag).cpu()
            runs.append((mode, F, r, r * rows * F, c, (src, idx, tag)))
        print(f"compiled {mode} F={F}", flush=True)
if PROF:
    import json
    import libtorch_neuronx_lite.envs as lenvs
    from vllm_neuron.vllm.worker.neuron_profiler import _DEFAULT_ACTIVITIES
    rt = torch.classes.neuron.Runtime()
    rt.start_profiling(PROF, _DEFAULT_ACTIVITIES, None, None, lenvs.get_neff_cache_dir(backend=get_compile_backend_name()))
with torch.no_grad():
    for mode, F, r, nbytes, c, args in runs:
        for _ in range(3):
            c(*args).cpu()
            order.append([mode, F, r, nbytes])
if PROF:
    rt.stop_profiling()
    json.dump(order, open(os.path.join(PROF, "order.json"), "w"))
    print("profiled", len(order), "executions; order in", os.path.join(PROF, "order.json"))
else:
    for mode, F, r, nbytes, c, args in runs:
        print(mode, F, r, f"{timeit(c, args):.3f} ms host")
