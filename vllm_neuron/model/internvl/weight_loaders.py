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


def fused_qkv_bias_loader(
    q_size: int,
    kv_size: int,
    num_shards: int,
    num_kv_replicas: int = 1,
) -> SafetensorsWeightLoader:
    """Fuse and shard the separate 1-D q/k/v biases of a Qwen2 attention layer.

    ``fused_qkv_weight_loader`` from the shared utils cannot be reused here: it
    asserts every checkpoint slice is 2-dimensional ("Q slice must be 2
    dimensional"), which holds for weights but not for biases.

    Args:
        q_size: Q bias entries per rank (num_q_heads_per_rank * head_dim).
        kv_size: K (and V) bias entries per rank.
        num_shards: TP world size.
        num_kv_replicas: How many consecutive ranks share one KV head, used when
            TP exceeds the KV head count.

    Returns:
        Loader producing ``[q_size + 2 * kv_size]`` for the calling rank.
    """

    def transform(slices, rank):
        assert len(slices) == 3, "expects [q_bias, k_bias, v_bias] in order"
        local_rank = rank % num_shards
        kv_rank = local_rank // num_kv_replicas
        parts = []
        for sl, size, shard_rank in zip(
            slices, (q_size, kv_size, kv_size), (local_rank, kv_rank, kv_rank)
        ):
            start = shard_rank * size
            parts.append(sl[start : start + size])
        return torch.cat(parts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def replicated_transposed_loader() -> SafetensorsWeightLoader:
    """Transpose a 2-D checkpoint tensor, no sharding.

    HF ``nn.Linear`` stores ``[out_features, in_features]``; this plugin's model
    code keeps ``[in, out]`` so the forward is a plain matmul. Used for the
    projector, which is replicated on every rank rather than sharded.
    """

    def transform(slices, rank):
        return slices[0][:].T.contiguous()

    return SafetensorsWeightLoader(transform=transform)
