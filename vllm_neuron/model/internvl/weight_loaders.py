# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for the InternViT vision encoder.

The checkpoint stores attention QKV fused as ``[3*H, H]`` laid out
``[Q_all | K_all | V_all]``. A contiguous row slice would give rank 0 all of Q
and none of K/V, so the loaders below take rank r's head slice out of each of
the three blocks and re-fuse them.

Same structure as the Qwen3-VL vision loaders; kept separate because InternViT's
QKV also has a bias and its head_dim is 64 rather than 128.
"""

from __future__ import annotations

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def vis_qkv_weight_loader(
    num_heads_per_rank: int,
    head_dim: int,
    vis_hidden_size: int,
) -> SafetensorsWeightLoader:
    """Shard the fused QKV weight ``[3*H, H]`` by head, returning ``[H, 3*hd]``.

    Args:
        num_heads_per_rank: Attention heads assigned to this TP rank.
        head_dim: Dimension per head (64 for InternViT-300M).
        vis_hidden_size: Vision hidden size H (= num_heads * head_dim).

    Returns:
        Loader producing ``[H, 3 * num_heads_per_rank * head_dim]``, transposed
        to match the matmul layout used by InternVisionAttention.
    """
    hd = num_heads_per_rank * head_dim
    H = vis_hidden_size

    def transform(slices, rank):
        w = slices[0][:]  # [3*H, H]
        q = w[rank * hd : (rank + 1) * hd]
        k = w[H + rank * hd : H + (rank + 1) * hd]
        v = w[2 * H + rank * hd : 2 * H + (rank + 1) * hd]
        return torch.cat([q, k, v], dim=0).T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def vis_qkv_bias_loader(
    num_heads_per_rank: int,
    head_dim: int,
    vis_hidden_size: int,
) -> SafetensorsWeightLoader:
    """Shard the fused QKV bias ``[3*H]`` the same way as the weight."""
    hd = num_heads_per_rank * head_dim
    H = vis_hidden_size

    def transform(slices, rank):
        b = slices[0][:]  # [3*H]
        bq = b[rank * hd : (rank + 1) * hd]
        bk = b[H + rank * hd : H + (rank + 1) * hd]
        bv = b[2 * H + rank * hd : 2 * H + (rank + 1) * hd]
        return torch.cat([bq, bk, bv], dim=0)

    return SafetensorsWeightLoader(transform=transform)


def patch_embed_weight_loader() -> SafetensorsWeightLoader:
    """Flatten the Conv2d patch-embed weight for the matmul formulation.

    Checkpoint holds ``[hidden, 3, 14, 14]``; InternVisionPatchEmbed wants
    ``[3*14*14, hidden]`` so a folded patch can be multiplied directly.
    Not sharded — every rank runs the full patch embedding.
    """

    def transform(slices, rank):
        w = slices[0][:]  # [hidden, 3, p, p]
        return w.reshape(w.shape[0], -1).T.contiguous()

    return SafetensorsWeightLoader(transform=transform)
