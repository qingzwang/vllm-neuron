# SPDX-License-Identifier: Apache-2.0
"""Sinkhorn-normalized hyper-connection mixing weights (DeepSeek-V4 mHC).

DeepSeek-V4 replaces the usual residual connection with Manifold-Constrained
Hyper-Connections: the residual stream carries ``hc_mult`` copies of the hidden
state, and each sublayer collapses them, runs, and mixes its output back. The
mixing weights come from a projection split into three parts — ``pre`` (collapse
weights), ``post`` (broadcast weights) and ``comb`` (an ``hc_mult x hc_mult``
combination matrix) — where ``comb`` is driven toward doubly stochastic by
alternating row/column normalization. That is what keeps the stream's scale
stable across 43 layers.

The 20 Sinkhorn rounds are the problem for compilation: expressed in torch they
unroll into the graph, and at ~900 HLO ops per hyper-connection with two per
layer they dominate the 43-layer instruction count. The NKI kernel iterates on
the hardware instead, so the cost is one round's worth of instructions
regardless of ``sinkhorn_iters``.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

# Tokens per kernel tile: SBUF partitions cap at 128.
PMAX = 128

# Fewest tokens worth dispatching to the kernel, mirroring the sparse-attention
# op: the win is a bounded instruction count over many tokens, and at one token
# per sequence the HOP-call overhead dominates. Decode stays in torch.
MIN_KERNEL_TOKENS = 8


@nki.jit
def _hc_sinkhorn_nki(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """Split the mix projection and Sinkhorn-normalize the combination matrix.

    Args:
        mixes: ``[T, (2 + hc_mult) * hc_mult]`` fp32 projection output.
        hc_scale: ``[1, 3]`` fp32 per-part scale.
        hc_base: ``[1, (2 + hc_mult) * hc_mult]`` fp32 per-element bias.
        hc_mult: Residual-stream copies (compile-time constant).
        sinkhorn_iters: Total normalization rounds (compile-time constant).
        eps: Stabilizer added after each normalization.

    Returns:
        ``(pre, post, comb)`` — ``[T, hc_mult]``, ``[T, hc_mult]`` and
        ``[T, hc_mult * hc_mult]`` (row-major), all fp32.
    """
    tokens = mixes.shape[0]
    hc = hc_mult
    comb_width = hc * hc

    pre_out = nl.ndarray((tokens, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    post_out = nl.ndarray((tokens, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    comb_out = nl.ndarray(
        (tokens, comb_width), dtype=nl.float32, buffer=nl.shared_hbm
    )

    scale_sb = nl.load(hc_scale)  # [1, 3]
    base_sb = nl.load(hc_base)    # [1, mix_hc]

    num_tiles = (tokens + PMAX - 1) // PMAX
    for tile in nl.static_range(num_tiles):
        t_lo = tile * PMAX
        t_hi = min(t_lo + PMAX, tokens)

        t_len = t_hi - t_lo
        m = nl.load(mixes[t_lo:t_hi])  # [t_len, mix_hc]

        # scale and base arrive with a single partition. Elementwise ops require
        # operands to match the destination's partition count (the CPU simulator
        # broadcasts implicitly; the compiler rejects it), so widen them here.
        scale_b = nl.broadcast_to(scale_sb, (t_len, scale_sb.shape[1]))
        base_b = nl.broadcast_to(base_sb, (t_len, base_sb.shape[1]))

        # ── pre = sigmoid(m[:hc] * scale[0] + base[:hc]) + eps ──
        pre = nl.add(
            nl.sigmoid(
                nl.add(
                    nl.multiply(m[:, 0:hc], scale_b[:, 0:1]),
                    base_b[:, 0:hc],
                )
            ),
            eps,
        )
        nl.store(pre_out[t_lo:t_hi], value=pre)

        # ── post = 2 * sigmoid(m[hc:2hc] * scale[1] + base[hc:2hc]) ──
        post = nl.multiply(
            nl.sigmoid(
                nl.add(
                    nl.multiply(m[:, hc : 2 * hc], scale_b[:, 1:2]),
                    base_b[:, hc : 2 * hc],
                )
            ),
            2.0,
        )
        nl.store(post_out[t_lo:t_hi], value=post)

        # ── comb: row softmax, then alternating row/column normalization ──
        comb = nl.ndarray((t_len, comb_width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            comb,
            nl.add(
                nl.multiply(m[:, 2 * hc :], scale_b[:, 2:3]),
                base_b[:, 2 * hc :],
            ),
        )

        # Row softmax. Rows of the hc x hc matrix are contiguous column ranges,
        # so each is its own free-axis slice.
        for row in nl.static_range(hc):
            lo = row * hc
            block = comb[:, lo : lo + hc]
            shifted = nl.subtract(block, nl.max(block, axis=1, keepdims=True))
            exps = nl.exp(shifted)
            normalized = nl.add(
                nl.multiply(exps, nl.reciprocal(nl.sum(exps, axis=1, keepdims=True))),
                eps,
            )
            nisa.tensor_copy(comb[:, lo : lo + hc], normalized)

        # The reference normalizes columns first, then alternates starting with
        # rows, for a total of sinkhorn_iters column passes.
        _normalize_columns(comb, hc, eps)
        # Iterating on the hardware keeps the instruction count independent of
        # sinkhorn_iters; unrolling this in torch is what made the 43-layer
        # graph exceed the compiler's budget.
        for _ in nl.sequential_range(sinkhorn_iters - 1):
            _normalize_rows(comb, hc, eps)
            _normalize_columns(comb, hc, eps)

        nl.store(comb_out[t_lo:t_hi], value=comb)

    return pre_out, post_out, comb_out


def _normalize_rows(comb, hc: int, eps: float) -> None:
    """Divide each row of the flattened hc x hc matrix by its sum, in place."""
    for row in nl.static_range(hc):
        lo = row * hc
        block = comb[:, lo : lo + hc]
        total = nl.add(nl.sum(block, axis=1, keepdims=True), eps)
        nisa.tensor_copy(
            comb[:, lo : lo + hc], nl.multiply(block, nl.reciprocal(total))
        )


def _normalize_columns(comb, hc: int, eps: float) -> None:
    """Divide each column of the flattened hc x hc matrix by its sum, in place.

    Column ``j`` is strided: elements ``j, j + hc, j + 2 * hc, ...``.
    """
    for col in nl.static_range(hc):
        total = comb[:, col : col + 1]
        for row in nl.static_range(1, hc):
            total = nl.add(total, comb[:, row * hc + col : row * hc + col + 1])
        inv = nl.reciprocal(nl.add(total, eps))
        for row in nl.static_range(hc):
            idx = row * hc + col
            nisa.tensor_copy(
                comb[:, idx : idx + 1], nl.multiply(comb[:, idx : idx + 1], inv)
            )


def hc_split_sinkhorn(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split the mix projection into pre / post / combination weights.

    Dispatches to the NKI kernel on Neuron and falls back to PyTorch otherwise.

    Args:
        mixes: ``[T, (2 + hc_mult) * hc_mult]`` projection output.
        hc_scale: ``[3]`` per-part scale.
        hc_base: ``[(2 + hc_mult) * hc_mult]`` per-element bias.
        hc_mult: Number of residual-stream copies.
        sinkhorn_iters: Total normalization rounds.
        eps: Stabilizer added after each normalization.

    Returns:
        ``(pre, post, comb)`` with shapes ``[T, hc]``, ``[T, hc]`` and
        ``[T, hc, hc]``, all fp32.
    """
    if mixes.ndim != 2:
        raise ValueError(f"mixes must be 2-D [T, mix_hc], got {tuple(mixes.shape)}")

    if mixes.shape[0] < MIN_KERNEL_TOKENS or not can_run_kernel(mixes):
        return _torch_hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )

    wrapped = wrap_nki(_hc_sinkhorn_nki)
    pre, post, comb = wrapped[1](
        mixes.float().contiguous(),
        hc_scale.float().reshape(1, -1).contiguous(),
        hc_base.float().reshape(1, -1).contiguous(),
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=float(eps),
    )
    return pre, post, comb.view(-1, hc_mult, hc_mult)


def _torch_hc_split_sinkhorn(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """PyTorch fallback, matching the reference implementation exactly."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(
        m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.unflatten(-1, (hc, hc))

    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
