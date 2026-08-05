# SPDX-License-Identifier: Apache-2.0
"""Sparse attention over a latent KV shared by every query head (MLA).

DeepSeek-V4's attention selects a subset of latent KV slots per query — a
sliding window plus, on most layers, compressed slots chosen by an indexer — and
attends over the union. Because the latent KV is a *single* head shared by all Q
heads, a slot contributes the same vector to both the logit and the weighted sum.

The plugin's flash-attention kernel caps ``head_dim`` at 128; MLA needs 512, so
this module provides a purpose-built NKI kernel plus a PyTorch fallback.

The kernel tiles both dimensions to the Tensor Engine's 128 limit — ``head_dim``
over PSUM-accumulated blocks (it is the logits' contraction axis) and the
selection over blocks folded by an online, FlashAttention-style softmax. That
keeps the gathered KV bounded at ``[128, 128]`` per step, which is what lets the
43-layer graph fit the compiler's instruction budget: a single-pass torch gather
materializes ``[tokens, selection, 512]`` and blows past it.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch import Tensor

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

# Tensor Engine is a 128x128 systolic array: a matmul's contraction dim (the
# stationary tile's partition axis) and its stationary free axis both cap at 128.
PMAX = 128

# Sentinel driving padding logits below any real one before the exponential.
_MASK_BIAS = 1.0e30


def _transpose(src, rows: int, cols: int):
    """Transpose an SBUF tile into a new SBUF tile.

    The Tensor Engine implements transpose as a matmul, so it writes PSUM (the
    Vector Engine path is capped at 32x32). Route through PSUM and copy back.
    """
    staging = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(staging, src, engine=nisa.engine.tensor)
    dst = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst, staging)
    return dst


@nki.jit
def _sparse_latent_attention_nki(kv_t, q_t, idx, valid, sink, scale):
    """NKI sparse latent attention.

    Args:
        kv_t:  ``[D, S]`` fp32 — latent KV, transposed so slots sit on the free
               axis, which is the axis ``gather_flattened`` indexes.
        q_t:   ``[T, D, H]`` fp32 — queries per token, transposed to match.
        idx:   ``[T, D, K]`` int32 — selected slot per token, broadcast down the
               D axis so every partition gathers the same slots. Values must be
               in range; the caller clamps.
        valid: ``[T, H, K]`` fp32 — 1.0 for a real selection, 0.0 for padding.
               The gather needs in-range indices, so the caller clamps and marks
               the clamped entries here instead.
        sink:  ``[H, 1]`` fp32 — per-head sink logit: a softmax column that
               carries no value, which is how a head attends to "nothing".
        scale: fp32 scalar — softmax scale.

    Returns:
        ``[T, H, D]`` fp32 attention output.
    """
    D, S = kv_t.shape
    T = q_t.shape[0]
    H = q_t.shape[2]
    K = idx.shape[2]

    d_blocks = (D + PMAX - 1) // PMAX
    k_blocks = (K + PMAX - 1) // PMAX

    out = nl.ndarray((T, H, D), dtype=nl.float32, buffer=nl.shared_hbm)
    sink_sb = nl.load(sink)  # [H, 1]

    # Tokens iterate on the hardware rather than unrolling into the graph, so
    # the instruction count stays independent of the sequence length.
    for t in nl.sequential_range(T):
        # Running online-softmax state. Written in place with tensor_copy:
        # nl.* expressions are lazy, so rebinding a Python name would leave the
        # tile the next iteration reads unchanged.
        running_max = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        denom = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(running_max, -_MASK_BIAS)
        nisa.memset(denom, 0.0)

        # Weighted-KV accumulator, laid out 2-D as [H, d_blocks * PMAX] so each
        # D block owns a contiguous column range. The natural 3-D shape
        # [d_blocks, H, PMAX] misaligns the partition axis when a [H, 1] tile is
        # broadcast against a partition slice of it.
        acc = nl.ndarray((H, d_blocks * PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc, 0.0)

        for kb in nl.static_range(k_blocks):
            k_lo = kb * PMAX
            k_hi = min(k_lo + PMAX, K)
            k_len = k_hi - k_lo

            # ── logits[k_len, H] = sum over D blocks of g_d^T @ q_d ──
            # The contraction is over D, which lives on the partition axis, so
            # the D blocks accumulate straight into PSUM: block 0 initializes
            # it, the rest add.
            logits_ps = nl.ndarray((k_len, H), dtype=nl.float32, buffer=nl.psum)
            for db in nl.static_range(d_blocks):
                d_lo = db * PMAX
                d_hi = min(d_lo + PMAX, D)
                gathered = nl.gather_flattened(
                    nl.load(kv_t[d_lo:d_hi]),
                    nl.load(idx[t, d_lo:d_hi, k_lo:k_hi]),
                )
                q_d = nl.load(q_t[t, d_lo:d_hi])
                nisa.nc_matmul(logits_ps, gathered, q_d, accumulate=(db > 0))

            logits = nl.ndarray((k_len, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(logits, logits_ps)

            # ── online softmax over this selection block ──
            # Transpose to [H, k_len] so the reductions run along the free axis.
            block_logits = nl.multiply(_transpose(logits, H, k_len), scale)

            # Padding slots gathered a clamped (arbitrary) row, so their logits
            # are meaningless: push them below any real logit for the max, and
            # zero their weight after the exponential.
            valid_blk = nl.load(valid[t, :, k_lo:k_hi])       # [H, k_len]
            masked = nl.add(
                nl.multiply(block_logits, valid_blk),
                nl.multiply(nl.subtract(valid_blk, 1.0), _MASK_BIAS),
            )

            block_max = nl.max(masked, axis=1, keepdims=True)  # [H, 1]
            new_max = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(new_max, nl.maximum(running_max, block_max))
            rescale = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                rescale, nl.exp(nl.subtract(running_max, new_max))
            )

            weights = nl.ndarray((H, k_len), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                weights,
                nl.multiply(nl.exp(nl.subtract(masked, new_max)), valid_blk),
            )

            new_denom = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                new_denom,
                nl.add(
                    nl.multiply(denom, rescale),
                    nl.sum(weights, axis=1, keepdims=True),
                ),
            )
            nisa.tensor_copy(denom, new_denom)
            nisa.tensor_copy(running_max, new_max)

            # ── fold weights @ gathered into the accumulator, per D block ──
            weights_k = _transpose(weights, k_len, H)          # [k_len, H]
            for db in nl.static_range(d_blocks):
                d_lo = db * PMAX
                d_hi = min(d_lo + PMAX, D)
                d_len = d_hi - d_lo
                gathered = nl.gather_flattened(
                    nl.load(kv_t[d_lo:d_hi]),
                    nl.load(idx[t, d_lo:d_hi, k_lo:k_hi]),
                )
                g_t = _transpose(gathered, k_len, d_len)       # [k_len, d_len]
                # weights_k^T[H, k_len] @ g_t[k_len, d_len] -> [H, d_len]
                contrib_ps = nl.matmul(weights_k, g_t, transpose_x=True)
                contrib = nl.ndarray((H, d_len), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(contrib, contrib_ps)
                a_lo = db * PMAX
                previous = nl.multiply(acc[:, a_lo : a_lo + d_len], rescale)
                nisa.tensor_copy(
                    acc[:, a_lo : a_lo + d_len], nl.add(previous, contrib)
                )

        # ── fold in the sink and normalize ──
        final_denom = nl.add(
            denom, nl.exp(nl.subtract(sink_sb, running_max))
        )
        for db in nl.static_range(d_blocks):
            d_lo = db * PMAX
            d_hi = min(d_lo + PMAX, D)
            d_len = d_hi - d_lo
            a_lo = db * PMAX
            nl.store(
                out[t, :, d_lo:d_hi],
                value=nl.divide(acc[:, a_lo : a_lo + d_len], final_denom),
            )

    return out


def _can_use_kernel(kv: Tensor, head_dim: int, selection: int) -> bool:
    """Check the NKI kernel's constraints.

    The kernel tiles freely over both dimensions, so the only hard requirements
    are hardware availability and fp32 compute.
    """
    if not can_run_kernel(kv):
        return False
    if head_dim < 1 or selection < 1:
        return False
    return True


def sparse_latent_attention(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    scale: float,
    chunk_size: int = PMAX,
) -> Tensor:
    """Sparse attention over a latent KV shared by all heads.

    Dispatches to the NKI kernel on Neuron and falls back to a chunked PyTorch
    implementation otherwise (CPU mode, or when a constraint is unmet).

    Args:
        q: ``[T, heads, head_dim]`` queries.
        kv: ``[S, head_dim]`` latent KV slots.
        attn_sink: ``[heads]`` sink logits (fp32).
        topk_idxs: ``[T, topk]`` int32 slot indices into ``kv``; ``-1`` (or any
            out-of-range value) marks an unused slot and is masked out.
        scale: Softmax scale, normally ``head_dim ** -0.5``.
        chunk_size: Selected slots per pass in the PyTorch fallback.

    Returns:
        ``[T, heads, head_dim]`` attention output in ``q``'s dtype.
    """
    tokens, heads, head_dim = q.shape
    num_slots, kv_dim = kv.shape
    if kv_dim != head_dim:
        raise ValueError(
            f"kv head_dim ({kv_dim}) must match q head_dim ({head_dim})"
        )
    selection = topk_idxs.shape[-1]

    if not _can_use_kernel(kv, head_dim, selection):
        return _torch_sparse_latent_attention(
            q, kv, attn_sink, topk_idxs, scale, chunk_size
        )

    idx = topk_idxs.long()
    valid = ((idx >= 0) & (idx < num_slots)).to(torch.float32)
    safe_idx = idx.clamp(0, num_slots - 1).to(torch.int32)

    # The kernel gathers along the free axis, so the KV and queries arrive
    # transposed, and the indices are broadcast down the head_dim axis so every
    # partition gathers the same slots.
    kv_t = kv.to(torch.float32).transpose(0, 1).contiguous()      # [D, S]
    q_t = q.to(torch.float32).transpose(1, 2).contiguous()        # [T, D, H]
    idx_b = safe_idx.unsqueeze(1).expand(tokens, head_dim, selection).contiguous()
    valid_b = valid.unsqueeze(1).expand(tokens, heads, selection).contiguous()
    sink = attn_sink.to(torch.float32).reshape(heads, 1).contiguous()

    wrapped = wrap_nki(_sparse_latent_attention_nki)
    out = wrapped[1](
        kv_t,
        q_t,
        idx_b,
        valid_b,
        sink,
        float(scale),
    )
    return out.to(q.dtype)


def _torch_sparse_latent_attention(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    scale: float,
    chunk_size: int,
) -> Tensor:
    """PyTorch fallback: the same online-softmax algorithm, chunked."""
    tokens, heads, head_dim = q.shape
    selection = topk_idxs.shape[-1]
    num_slots = kv.shape[0]
    q_f32 = q.float()

    lowest = torch.finfo(torch.float32).min
    running_max = torch.full(
        (tokens, heads), lowest, device=q.device, dtype=torch.float32
    )
    denominator = torch.zeros(
        tokens, heads, device=q.device, dtype=torch.float32
    )
    accumulator = torch.zeros(
        tokens, heads, head_dim, device=q.device, dtype=torch.float32
    )

    for start in range(0, selection, chunk_size):
        stop = min(start + chunk_size, selection)
        idx = topk_idxs[:, start:stop].long()
        valid = (idx >= 0) & (idx < num_slots)
        width = stop - start

        gathered = kv.index_select(
            0, idx.clamp(0, num_slots - 1).reshape(-1)
        ).view(tokens, width, head_dim)

        logits = torch.einsum("thd,tkd->thk", q_f32, gathered.float())
        logits = logits * torch.full_like(logits, scale)
        logits = torch.where(
            valid.unsqueeze(1), logits, torch.full_like(logits, lowest)
        )

        chunk_max = torch.maximum(running_max, logits.amax(dim=-1))
        rescale = torch.exp(running_max - chunk_max)
        weights = torch.exp(logits - chunk_max.unsqueeze(-1))

        accumulator = accumulator * rescale.unsqueeze(-1) + torch.einsum(
            "thk,tkd->thd", weights, gathered.float()
        )
        denominator = denominator * rescale + weights.sum(dim=-1)
        running_max = chunk_max

    denominator = denominator + torch.exp(
        attn_sink.float().view(1, heads) - running_max
    )
    return (accumulator / denominator.unsqueeze(-1)).to(q.dtype)
