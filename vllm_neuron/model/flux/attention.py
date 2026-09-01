# SPDX-License-Identifier: Apache-2.0
"""Neuron joint-attention processor for the FLUX transformer.

Drop-in replacement for diffusers' ``FluxAttnProcessor``. It keeps the upstream
projection / QK-norm / RoPE math verbatim and only swaps the attention core:
instead of going through diffusers' backend dispatch to
``F.scaled_dot_product_attention``, it calls this package's NKI flash-attention
kernel, which never materializes the S x S score matrix.

That matters at FLUX resolutions. For 1024x1024 with a 512-token prompt the
joint sequence is 4608, so a materialized BF16 score matrix is 24 heads x 4608 x
4608 x 2 B = 1.0 GiB per attention -- and there are 46 of them per denoising
step. Measured on trn2 at those shapes: 4.9 ms/call for the kernel vs 22.1 ms
for SDPA.

FLUX attention is fully bidirectional (text and image tokens attend to each
other in both directions), so the kernel is called with ``causal_mask=False``.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.attention.attention_cte import flash_attention


def apply_rotary_emb(
    x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Apply FLUX rotary embeddings to a ``[batch, seq, heads, head_dim]`` tensor.

    Same expression as diffusers'
    ``apply_rotary_emb(..., use_real_unbind_dim=-1, sequence_dim=1)`` -- pairs of
    adjacent features are treated as one complex number. Reimplemented because
    the upstream helper ends with ``cos.to(x.device)``, and an explicit device
    copy inside a traced graph lowers to an unimplemented
    ``_copy_from xla:0 -> neuron:0``. The tables already arrive on the right
    device, so the copy has nothing to do anyway.

    One numerical difference from upstream: it carries float64 RoPE tables and so
    rotates in float64, while these tables are float32 (Neuron has no float64).
    The rotation itself is still promoted out of BF16.

    Args:
        x: Query or key tensor, ``[batch, seq, heads, head_dim]``.
        freqs: ``(cos, sin)``, each ``[seq, head_dim]``.

    Returns:
        Rotated tensor, same shape and dtype as ``x``.
    """
    cos, sin = freqs
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def neuron_joint_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Full bidirectional attention over the joint text+image sequence.

    Args:
        query: ``[batch, seq, heads, head_dim]``.
        key: ``[batch, seq, heads, head_dim]``.
        value: ``[batch, seq, heads, head_dim]``.

    Returns:
        ``[batch, seq, heads, head_dim]`` attention output.
    """
    batch, seq, heads, head_dim = query.shape

    # The kernel takes 3D tensors and treats dim 0 as batch, so heads fold into
    # it. Layout per the kernel contract: q/v are [B, S, D], k is [B, D, S].
    q = query.permute(0, 2, 1, 3).reshape(batch * heads, seq, head_dim)
    k = key.permute(0, 2, 3, 1).reshape(batch * heads, head_dim, seq)
    v = value.permute(0, 2, 1, 3).reshape(batch * heads, seq, head_dim)

    out = flash_attention(q, k, v, causal_mask=False)

    return out.reshape(batch, heads, seq, head_dim).permute(0, 2, 1, 3)


class NeuronFluxAttnProcessor:
    """Joint attention for both ``FluxTransformerBlock`` variants.

    The double-stream blocks pass ``encoder_hidden_states`` and expect the text
    and image halves back separately; the single-stream blocks pass a
    pre-concatenated sequence and expect one tensor. Which mode applies is
    determined by ``attn.added_kv_proj_dim``, matching upstream.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            raise NotImplementedError(
                "NeuronFluxAttnProcessor does not support attention_mask; FLUX "
                "pads prompts to a fixed length and attends over everything."
            )

        query = attn.to_q(hidden_states).unflatten(-1, (-1, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (-1, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (-1, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(
                -1, (-1, attn.head_dim)
            )
            encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(
                -1, (-1, attn.head_dim)
            )
            encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(
                -1, (-1, attn.head_dim)
            )
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # Text tokens come first; the RoPE tables are built in the same
            # order (see NeuronFluxTransformer.build_rotary_embedding).
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = neuron_joint_attention(query, key, value)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is None:
            return hidden_states

        text_seq_len = encoder_hidden_states.shape[1]
        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [text_seq_len, hidden_states.shape[1] - text_seq_len], dim=1
        )
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.contiguous()))
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
        return hidden_states, encoder_hidden_states
