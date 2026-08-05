# SPDX-License-Identifier: Apache-2.0
"""Shared building blocks for DeepSeek-V4.

These pieces are architecture-specific but parallelism-agnostic: they take
already-sharded tensors and do pure math, so they are shared by the attention,
MoE and hyper-connection code and are directly comparable against the reference
implementation shipped in the checkpoint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.functional.hc_sinkhorn import hc_split_sinkhorn


class DeepseekV4RMSNorm(nn.Module):
    """RMSNorm that normalizes in fp32 and returns the input dtype.

    The checkpoint stores these weights in bf16; the reference implementation
    upcasts the weight to fp32 and does the whole reduction there, so we keep an
    fp32 parameter to match bit-for-bit.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        variance = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Unweighted RMS normalization, applied to Q heads before RoPE."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


# ── Rotary embeddings (YaRN, interleaved pairs) ───────────────────────────────


def _find_correction_dim(
    num_rotations: float, dim: int, base: float, max_seq_len: int
) -> float:
    return (
        dim
        * math.log(max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp(
    low: float, high: float, dim: int, device: torch.device | None = None
) -> torch.Tensor:
    if low == high:
        high += 0.001
    ramp = (
        torch.arange(dim, dtype=torch.float32, device=device) - low
    ) / (high - low)
    return ramp.clamp(0, 1)


def compute_yarn_inv_freq(
    rotary_dim: int,
    base: float,
    original_max_position_embeddings: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Inverse frequencies with YaRN interpolation.

    When ``original_max_position_embeddings`` is 0 the scaling is disabled and
    plain RoPE frequencies are returned — the reference implementation uses this
    for the sliding-window-only stream.
    """
    freqs = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    if original_max_position_embeddings > 0:
        low, high = _find_correction_range(
            beta_fast, beta_slow, rotary_dim, base, original_max_position_embeddings
        )
        smooth = 1 - _linear_ramp(low, high, rotary_dim // 2, device=device)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


class DeepseekV4RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for the interleaved-pair RoPE.

    DeepSeek-V4 rotates *adjacent* element pairs (GPT-J style), not split halves,
    and applies the rotation only to the trailing ``rotary_dim`` elements of the
    latent KV / Q head. Getting this wrong produces plausible-looking but wrong
    attention output, so the pairing is asserted by the accuracy tests.

    Each layer needs up to two of these: the sliding-window stream uses
    ``rope_theta`` with YaRN disabled, and the compressed streams use
    ``compress_rope_theta`` with YaRN enabled.
    """

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        original_max_position_embeddings: int = 0,
        factor: float = 1.0,
        beta_fast: float = 32,
        beta_slow: float = 1,
    ):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.factor = factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``[T, rotary_dim // 2]``.

        Half-width tables: with interleaved pairs each ``(cos, sin)`` entry is
        applied to one element pair, so no doubling is needed.

        The frequencies are recomputed here rather than cached in a buffer.
        Models are built on the meta device, so a buffer created in ``__init__``
        would have no storage; deriving them from ``positions.device`` keeps
        everything on the traced device.
        """
        inv_freq = compute_yarn_inv_freq(
            self.rotary_dim,
            self.base,
            self.original_max_position_embeddings,
            self.factor,
            self.beta_fast,
            self.beta_slow,
            device=positions.device,
        )
        freqs = torch.outer(positions.float(), inv_freq)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate adjacent element pairs of ``x`` by the given angles.

    Follows the runner's flat-token convention: the token dim is dim 0, with any
    head dims in between, and the trailing dim is rotated. The reference
    implementation's leading batch dim has no equivalent here.

    Args:
        x: ``[T, ..., rotary_dim]`` — trailing dim is rotated and must be even.
        cos: ``[T, rotary_dim // 2]``.
        sin: ``[T, rotary_dim // 2]``.
        inverse: Apply the conjugate rotation (used to de-rotate attention
            output before the O projection).

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    dtype = x.dtype
    pairs = x.float().unflatten(-1, (-1, 2))
    x_even = pairs[..., 0]
    x_odd = pairs[..., 1]

    # Insert singleton axes for any head dims between the token dim and the
    # rotated dim, so the [T, rotary_dim // 2] tables broadcast over them.
    for _ in range(x_even.ndim - cos.ndim):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    cos = cos.to(x_even.dtype)
    sin = sin.to(x_even.dtype)
    if inverse:
        sin = -sin

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2).to(dtype)


# ── Manifold-Constrained Hyper-Connections (mHC) ─────────────────────────────


class HyperConnection(nn.Module):
    """One hyper-connection mix: reduces ``hc_mult`` streams to 1 and back.

    ``pre`` collapses the residual streams into the single tensor a sublayer
    (attention or MoE) consumes; ``post`` re-expands the sublayer output and
    mixes it back into the streams via the Sinkhorn-normalized combination
    matrix.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

        mix_hc = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.empty(mix_hc, hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def pre(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse ``[T, hc, H]`` streams into ``[T, H]``.

        Returns ``(collapsed, post, comb)`` — the latter two are threaded into
        :meth:`post` after the sublayer runs.
        """
        shape = streams.shape
        dtype = streams.dtype
        flat = streams.flatten(-2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
        )
        collapsed = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=-2)
        return collapsed.to(dtype), post, comb

    def post(
        self,
        sublayer_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Mix ``[T, H]`` sublayer output back into the ``[T, hc, H]`` streams.

        The combination matrix contracts over its *first* index:
        ``out[j] = post[j] * sublayer_out + sum_i comb[i, j] * residual[i]``.
        """
        broadcast = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2)
        mixed = torch.einsum("...ij,...id->...jd", comb, residual.float())
        return (broadcast + mixed).type_as(sublayer_out)


class HyperConnectionHead(nn.Module):
    """Final stream collapse before the LM head.

    Unlike :class:`HyperConnection` this has no Sinkhorn step — the head only
    needs the ``pre`` weights, so the projection is ``hc_mult`` wide instead of
    ``(2 + hc_mult) * hc_mult``.
    """

    def __init__(
        self, hidden_size: int, hc_mult: int, norm_eps: float, hc_eps: float
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.fn = nn.Parameter(
            torch.empty(hc_mult, hc_mult * hidden_size, dtype=torch.float32)
        )
        self.base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        shape = streams.shape
        dtype = streams.dtype
        flat = streams.flatten(-2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.fn) * rsqrt
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        collapsed = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=-2)
        return collapsed.to(dtype)


# ── MoE expert math ──────────────────────────────────────────────────────────


FP8_E4M3_MAX = 448.0


def fake_quant_fp8(
    x: torch.Tensor, block_size: int = 64, round_scale_to_pow2: bool = True
) -> torch.Tensor:
    """Quantize to FP8 (e4m3) per block along the last dim, then dequantize.

    DeepSeek-V4 was trained quantization-aware: the reference implementation
    runs this on the latent KV's content dims and on the compressor output
    before they enter attention. Skipping it shifts activations by ~2-3%
    relative, which compounds across 43 layers — so the port reproduces it even
    though Neuron computes in bf16.

    The RoPE dims are deliberately excluded by the callers: they stay bf16 to
    preserve positional precision.

    Args:
        x: Tensor whose last dim is a multiple of ``block_size``.
        block_size: Elements sharing one scale.
        round_scale_to_pow2: Round scales up to a power of two, matching the
            checkpoint's ``ue8m0`` scale format.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    width = x.shape[-1]
    if width % block_size:
        raise ValueError(
            f"fake_quant_fp8 needs the last dim ({width}) to be a multiple of "
            f"block_size ({block_size})"
        )
    dtype = x.dtype
    blocks = x.float().unflatten(-1, (width // block_size, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = amax / FP8_E4M3_MAX
    if round_scale_to_pow2:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    quantized = (blocks / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = quantized.to(torch.float8_e4m3fn).to(torch.float32)
    return (quantized * scale).flatten(-2).to(dtype)


FP4_E2M1_MAX = 6.0

# float4_e2m1 representable magnitudes, ascending, and the midpoints between
# consecutive levels. Kept as Python constants: building them with
# torch.tensor() inside forward() breaks Dynamo's fake-tensor tracing.
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MIDPOINTS = tuple(
    (_FP4_LEVELS[i] + _FP4_LEVELS[i + 1]) / 2 for i in range(len(_FP4_LEVELS) - 1)
)


def _round_to_e2m1(magnitude: torch.Tensor) -> torch.Tensor:
    """Round non-negative magnitudes to the nearest e2m1 level.

    Built as a sum of thresholded increments — ``level_0 + sum_i step_i *
    (level_i - level_{i-1})`` — rather than a chain of selects or a gather.
    Both alternatives are worse here: a lookup table needs a device-side
    constant tensor (which breaks Dynamo's fake-tensor tracing), and a
    ``torch.where`` chain feeding downstream indexing trips an internal
    neuronx-cc error (NCC_ILSA902) during legalization.

    The threshold is strict (``>``) so exact midpoints round down, matching
    ``torch.bucketize``'s right-open interval and hence the reference kernel.
    """
    result = torch.full_like(magnitude, _FP4_LEVELS[0])
    previous = _FP4_LEVELS[0]
    for level, midpoint in zip(_FP4_LEVELS[1:], _FP4_MIDPOINTS):
        step = (magnitude > torch.full_like(magnitude, midpoint)).to(magnitude.dtype)
        result = result + step * (level - previous)
        previous = level
    return result


def fake_quant_fp4(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Quantize to FP4 (e2m1) per block along the last dim, then dequantize.

    Used on the indexer's compressed stream and query, which the reference
    implementation keeps on an FP4 grid with power-of-two block scales.

    Args:
        x: Tensor whose last dim is a multiple of ``block_size``.
        block_size: Elements sharing one scale.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    width = x.shape[-1]
    if width % block_size:
        raise ValueError(
            f"fake_quant_fp4 needs the last dim ({width}) to be a multiple of "
            f"block_size ({block_size})"
        )
    dtype = x.dtype
    blocks = x.float().unflatten(-1, (width // block_size, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(FP4_E2M1_MAX * 2.0**-126)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / FP4_E2M1_MAX)))
    scaled = (blocks / scale).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    quantized = torch.sign(scaled) * _round_to_e2m1(scaled.abs())
    return (quantized * scale).flatten(-2).to(dtype)


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply a normalized Walsh-Hadamard transform over the last dim.

    Spreads per-element magnitude across the vector so a shared block scale
    loses less precision. The reference uses ``fast_hadamard_transform`` with
    ``scale = dim ** -0.5``; this is the same transform written in torch so it
    traces on Neuron.

    Args:
        x: Tensor whose last dim is a power of two.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    dtype = x.dtype
    width = x.shape[-1]
    if width & (width - 1):
        raise ValueError(f"hadamard_rotate needs a power-of-two last dim, got {width}")

    y = x.float()
    stride = 1
    while stride < width:
        y = y.unflatten(-1, (width // (2 * stride), 2, stride))
        low = y[..., 0, :]
        high = y[..., 1, :]
        y = torch.stack([low + high, low - high], dim=-2)
        y = y.flatten(-3)
        stride *= 2
    return (y * width**-0.5).to(dtype)


def swiglu(
    gate: torch.Tensor, up: torch.Tensor, limit: float = 0.0
) -> torch.Tensor:
    """SwiGLU with the asymmetric clamping the reference applies.

    ``up`` is clamped on both sides but ``gate`` only from above — matching
    ``Expert.forward`` in the reference implementation.
    """
    gate = gate.float()
    up = up.float()
    if limit > 0:
        up = up.clamp(min=-limit, max=limit)
        gate = gate.clamp(max=limit)
    return F.silu(gate) * up


def topk_mask(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Boolean mask marking the ``top_k`` largest entries of each row.

    Avoids ``torch.topk``: XLA lowers top-k over a small trailing dim to
    ``sort``, which neuronx-cc rejects on Trainium. Instead the k-th largest
    value is found by iterated max-and-mask, which is ``top_k`` elementwise
    passes — cheap for the k=6 that DeepSeek-V4 uses, and fully traceable.

    Ties are broken by position (lower index wins), so exactly ``top_k``
    entries are selected even when scores are equal — which matters here,
    because DeepSeek-V4's expert scores are near-degenerate.

    Args:
        scores: ``[T, E]`` scores.
        top_k: Number of entries to select per row.

    Returns:
        ``[T, E]`` boolean mask with exactly ``top_k`` True entries per row.
    """
    num_experts = scores.shape[-1]
    if top_k >= num_experts:
        return torch.ones_like(scores, dtype=torch.bool)

    # Break ties by index so the running max picks a unique winner each round.
    rank_bias = torch.arange(
        num_experts, device=scores.device, dtype=scores.dtype
    ) * -1e-6
    keyed = scores + rank_bias

    remaining = keyed
    selected = torch.zeros_like(scores, dtype=torch.bool)
    neg_inf = torch.finfo(keyed.dtype).min
    for _ in range(top_k):
        winner = remaining.argmax(dim=-1, keepdim=True)
        selected = selected.scatter(-1, winner, True)
        remaining = remaining.scatter(-1, winner, neg_inf)
    return selected


def compute_router_scores(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor | None,
    top_k: int,
    routed_scaling_factor: float,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score-based MoE routing with the sqrt-softplus activation.

    The bias shifts scores for *selection* only; the returned affinities come
    from the unbiased scores (``topk_method="noaux_tc"``).

    Args:
        hidden_states: ``[T, H]``.
        router_weight: ``[E, H]``.
        router_bias: ``[E]`` selection bias, or None for the hash layers.
        top_k: Experts per token.
        routed_scaling_factor: Final multiplier on the routing weights.
        normalize: L1-normalize the selected weights before scaling.

    Returns:
        ``(affinities, selected)``: ``[T, E]`` fp32 affinities, zero for
        unselected experts, and the ``[T, E]`` bool selection mask.
    """
    scores = F.linear(hidden_states.float(), router_weight.float())
    scores = F.softplus(scores).sqrt()
    unbiased = scores
    if router_bias is not None:
        scores = scores + router_bias.float()

    selected = topk_mask(scores, top_k)
    affinities = torch.where(selected, unbiased, torch.zeros_like(unbiased))
    if normalize:
        affinities = affinities / affinities.sum(dim=-1, keepdim=True)
    return affinities * routed_scaling_factor, selected


def compute_hash_router_scores(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    routed_scaling_factor: float,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hash routing: expert selection comes from a fixed token-id table.

    The first ``num_hash_layers`` layers use this. Scores still come from the
    router weight and provide the mixing affinities, but *which* experts a token
    uses is a table lookup on its id — so routing is independent of the hidden
    state.

    Args:
        hidden_states: ``[T, H]``.
        router_weight: ``[E, H]``.
        tid2eid: ``[vocab_size, top_k]`` token id to expert id table.
        input_ids: ``[T]`` token ids.
        routed_scaling_factor: Final multiplier on the routing weights.
        normalize: L1-normalize the selected weights before scaling.

    Returns:
        ``(affinities, selected)``: ``[T, E]`` fp32 affinities, zero for
        unselected experts, and the ``[T, E]`` bool selection mask.

    Note:
        A token id may appear more than once in its ``tid2eid`` row. The
        reference implementation then adds that expert's contribution twice; the
        mask-based form here counts it once. The shipped checkpoints have no
        duplicate rows, so the two agree.
    """
    scores = F.linear(hidden_states.float(), router_weight.float())
    scores = F.softplus(scores).sqrt()

    # Clamp into the table: padding and warmup-synthetic token ids are not
    # guaranteed to be inside the vocabulary, and an out-of-range index faults
    # the hardware gather. Padding tokens' routing is discarded downstream.
    token_ids = input_ids.long().clamp(0, tid2eid.shape[0] - 1)
    indices = tid2eid[token_ids].long()
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected = selected.scatter(-1, indices, True)

    affinities = torch.where(selected, scores, torch.zeros_like(scores))
    if normalize:
        affinities = affinities / affinities.sum(dim=-1, keepdim=True)
    return affinities * routed_scaling_factor, selected
