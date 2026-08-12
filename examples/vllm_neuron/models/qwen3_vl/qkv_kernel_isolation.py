# SPDX-License-Identifier: Apache-2.0
"""Isolate the NKI QKV kernel: compare kernel output against the PyTorch path.

Reproducer for the bug documented in TP2_WRONG_OUTPUT.md: the kernel silently
computes wrong results once fused_qkv_dim reaches 3072, which is what makes
Qwen3-VL-8B produce garbage at TP=2 (per-rank fused_qkv_dim 3072) while TP=4
(1536) is fine. Runs in about a minute; edit CASES to sweep.

Mirrors the exact call Qwen3-VL's text attention makes (fused QKV + per-head QK
RMSNorm + M-RoPE), sweeping the per-rank head counts that different TP degrees
produce. The reference is the same qkv_proj function run on CPU, where
can_run_kernel() returns False and it takes the PyTorch branch — so any
divergence is the kernel alone.

Qwen3-VL-8B text config: hidden 4096, 32 Q heads, 8 KV heads, head_dim 128,
so fused_qkv_dim per rank = (32/TP + 2*8/TP) * 128.
"""

import torch

import vllm_neuron as vllm_neuron  # noqa: F401
from vllm_neuron import envs
from vllm_neuron.envs import get_compile_backend_name
import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.qkv import NormType

DEVICE = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"
D_HEAD = 128
T = 2048           # tokens; CTE path (T > 96), value does not enter the guard
EPS = 1e-6
DTYPE = torch.bfloat16

# (label, hidden, q_heads_per_rank, kv_heads_per_rank)
# Boundary hunt at H=4096, plus two cases that separate "ratio to H" from
# "absolute fused_qkv_dim", plus a same-fused/different-split control.
CASES = [
    ("f=1536", 4096, 8, 2),
    ("f=2048", 4096, 8, 4),
    ("f=2560", 4096, 16, 2),
    ("f=2944", 4096, 17, 3),
    ("f=3072", 4096, 16, 4),
]


class QKV(torch.nn.Module):
    def __init__(self, nq, nkv):
        super().__init__()
        self.nq, self.nkv = nq, nkv

    def forward(self, hidden, w, cos, sin, qg, kg):
        return NF.qkv_proj(
            hidden=hidden,
            qkv_weights=w,
            cos_cache=cos,
            sin_cache=sin,
            num_q_heads=self.nq,
            num_kv_heads=self.nkv,
            d_head=D_HEAD,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=EPS,
            qk_norm_pre_rope_q_gamma=qg,
            qk_norm_pre_rope_k_gamma=kg,
        )


def build(H, nq, nkv, seed=0):
    g = torch.Generator().manual_seed(seed)
    fused = (nq + 2 * nkv) * D_HEAD
    hidden = torch.randn(1, T, H, generator=g).to(DTYPE)
    w = (torch.randn(H, fused, generator=g) * (H**-0.5)).to(DTYPE)
    half = torch.randn(T, D_HEAD // 2, generator=g).to(DTYPE)
    cos = torch.cat((half, half), dim=-1).unsqueeze(0)
    sin = torch.cat((half, half), dim=-1).unsqueeze(0)
    qg = torch.randn(1, D_HEAD, generator=g).to(DTYPE)
    kg = torch.randn(1, D_HEAD, generator=g).to(DTYPE)
    return fused, hidden, w, cos, sin, qg, kg


def main():
    print(f"{'case':>11} {'H':>5} {'q/kv':>7} {'fused':>6} {'fused/H':>8} {'guard':>8} "
          f"{'max|diff|':>10} {'rel':>9}  verdict")
    for label, H, nq, nkv in CASES:
        fused, hidden, w, cos, sin, qg, kg = build(H, nq, nkv)
        mod = QKV(nq, nkv)

        # Reference: CPU → can_run_kernel() False → PyTorch branch
        ref = mod(hidden, w, cos, sin, qg, kg).float()

        # Kernel: on device via the Neuron backend
        dev = [t.to(DEVICE) for t in (hidden, w, cos, sin, qg, kg)]
        compiled = torch.compile(mod.to(DEVICE), backend=get_compile_backend_name())
        out = compiled(*dev).to("cpu").float()

        diff = (out - ref).abs().max().item()
        scale = ref.abs().max().item()
        rel = diff / scale if scale else float("nan")
        # bf16 matmul on different hardware paths: ~1e-2 relative is normal,
        # anything near or above 1.0 means the kernel computed something else.
        verdict = "OK" if rel < 0.05 else "*** WRONG ***"
        print(f"{label:>11} {H:>5} {f'{nq}/{nkv}':>7} {fused:>6} {fused / H:>8.3f} "
              f"{'pass' if fused <= H else 'FALLBACK':>7} "
              f"{diff:>10.4f} {rel:>9.4f}  {verdict}")


if __name__ == "__main__":
    main()
