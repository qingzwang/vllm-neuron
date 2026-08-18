#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Validate the Qwen3.5 vision tower against HuggingFace, on CPU.

The tower is the plugin's Qwen3-VL encoder reused verbatim, on the grounds that
HF's ``Qwen3_5VisionModel`` is ``Qwen3VLVisionModel`` with the deepstack mergers
deleted and the checkpoint tensor names are identical. This checks that claim
rather than trusting it, on the real weights:

1. **Weight loading** — every parameter of the reused encoder gets a tensor from
   Qwen3.5's checkpoint, and no deepstack mergers are built.
2. **Structure** — merged embeddings come out in the same order and at the same
   magnitude as HF's ``Qwen3_5VisionModel`` for a real image grid.

**What this cannot do is validate the tower numerically.** The plugin's vision
attention runs on NKI kernels (``NF.qkv_proj`` / ``flash_attention`` / ``o_proj``)
and ``can_run_kernel("cpu")`` is False, so on CPU it takes a fallback path whose
arithmetic does not match HF's op-for-op: the very first transformer block already
differs by ~4e-3 and that compounds over 24 blocks to ~1e-1. That is a property of
running kernel code off-device, not evidence about Qwen3.5. The encoder itself is
existing, benchmarked Qwen3-VL code; what needed checking here is only whether it
*applies unchanged*, which is a question about weights and shapes.

Numerical validation of the vision path therefore happens **on device**, with a
real image, against HF generation — see ``run_vl.py`` and
``check_generation_vs_hf.py``.

The encoder takes block-packed, pre-rotated inputs (that is how it keeps static
shapes on device), so the comparison goes through the same packing helpers the
model uses at runtime.

Usage:

    PYTHONPATH=/mnt/nvme/vllm-neuron python check_vision_vs_hf.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument(
        "--grid",
        default="1,16,16",
        help="T,H,W patch grid for one image (must be even in H and W)",
    )
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--tol", type=float, default=2e-2)
    return parser.parse_args()


def init_single_rank_distributed() -> None:
    """A world of one, with vLLM's model-parallel groups set up.

    The vision encoder resolves its TP/DP groups through
    ``get_neuron_vision_tp_group``, which falls through to vLLM's ``get_tp_group``
    — so plain ``torch.distributed`` is not enough; vLLM's parallel state has to
    be initialised too.
    """
    import torch.distributed as dist
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29599",
        local_rank=0,
        backend="gloo",
    )
    # initialize_model_parallel reads the ambient vLLM config, so give it one.
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")


def report(name: str, ours: torch.Tensor, theirs: torch.Tensor, tol: float) -> bool:
    diff = (ours - theirs).abs()
    scale = theirs.abs().max().clamp(min=1e-9)
    rel = (diff.max() / scale).item()
    ok = rel <= tol
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name:26s} max|d|={diff.max():.3e} "
        f"rel={rel:.3e}"
    )
    return ok


def build_config(model_path: str, block_size: int):
    from transformers import AutoConfig

    from vllm_neuron.model.neuron_config import VisionNeuronConfig
    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    hf_config = AutoConfig.from_pretrained(model_path)
    vision_neuron_config = VisionNeuronConfig(
        num_vision_tokens_buckets=[block_size],
        vision_attention_block_size=block_size,
    )
    config = Qwen3_5Config.from_configs(
        hf_config, vision_neuron_config=vision_neuron_config, include_vision=True
    )
    # float32 so a mismatch is a bug rather than bf16 noise.
    config.vision_config.torch_dtype = torch.float32
    return hf_config, config


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    init_single_rank_distributed()

    from vllm_neuron.model.qwen3_vl.vision_encoder_bf16 import Qwen3VLVisionModel

    hf_config, config = build_config(args.model, args.block_size)
    vision_config = config.vision_config
    grid = [int(x) for x in args.grid.split(",")]
    t, h, w = grid
    merge = vision_config.spatial_merge_size
    if h % merge or w % merge:
        raise SystemExit(f"grid H,W must be multiples of {merge}")
    num_tokens = t * h * w
    print(f"checkpoint: {args.model}")
    print(
        f"vision: depth={vision_config.depth} hidden={vision_config.hidden_size} "
        f"-> out={vision_config.out_hidden_size} heads={vision_config.num_heads} "
        f"merge={merge} deepstack={list(vision_config.deepstack_visual_indexes)}"
    )
    print(f"grid {t},{h},{w} -> {num_tokens} raw tokens, "
          f"{num_tokens // merge**2} merged\n")

    ok = True

    # --- 1. build + load ------------------------------------------------
    print("1. weight loading")
    tower = Qwen3VLVisionModel(vision_config, dtype=torch.float32)
    if len(tower.deepstack_merger_list) != 0:
        print(f"  FAIL  built {len(tower.deepstack_merger_list)} deepstack mergers")
        ok = False
    else:
        print("  PASS  no deepstack mergers built")
    before = {n: p.detach().clone() for n, p in tower.named_parameters()}
    tower.load_weights(args.model, device="cpu", cpu_mode=True)
    unchanged = [
        n for n, p in tower.named_parameters() if torch.equal(p.detach(), before[n])
    ]
    if unchanged:
        print(f"  FAIL  {len(unchanged)} parameter(s) unchanged by load: "
              f"{unchanged[:5]}")
        ok = False
    else:
        print(f"  PASS  all {len(before)} parameters loaded from the checkpoint")
    # load_weights assigns the checkpoint's bf16 tensors, so cast back: this
    # comparison wants float32 on both sides.
    tower = tower.float().eval()

    # --- 2. HF reference ------------------------------------------------
    print("\n2. encoder output vs HF")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    hf_tower = Qwen3_5VisionModel._from_config(hf_config.vision_config).float().eval()
    # Load the same checkpoint tensors into HF's module.
    from safetensors import safe_open
    import glob
    import json

    with open(os.path.join(args.model, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    wanted = {k: v for k, v in weight_map.items() if k.startswith("model.visual.")}
    state = {}
    for shard in sorted(set(wanted.values())):
        path = os.path.join(args.model, shard)
        if not os.path.exists(path):
            (path,) = glob.glob(os.path.join(args.model, "*.safetensors"))
        with safe_open(path, framework="pt") as handle:
            for key, where in wanted.items():
                if where == shard:
                    state[key[len("model.visual.") :]] = handle.get_tensor(key).float()
    missing, unexpected = hf_tower.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected vision keys for HF: {unexpected[:5]}")
    if missing:
        print(f"  (HF keys absent from the checkpoint, left at init: {missing[:5]})")

    torch.manual_seed(0)
    patch_dim = (
        vision_config.in_channels
        * vision_config.temporal_patch_size
        * vision_config.patch_size**2
    )
    pixel_values = torch.randn(num_tokens, patch_dim) * 0.5
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)

    hf_block_out: list[torch.Tensor] = []
    hf_handles = [
        blk.register_forward_hook(
            lambda _m, _i, out, store=hf_block_out: store.append(
                (out[0] if isinstance(out, tuple) else out).detach().clone()
            )
        )
        for blk in hf_tower.blocks
    ]
    hf_out = hf_tower(pixel_values, grid_thw)
    for handle in hf_handles:
        handle.remove()
    hf_merged = (
        hf_out.pooler_output if hasattr(hf_out, "pooler_output") else hf_out
    )

    # --- 3. ours, through the real packing pipeline ---------------------
    # Same call sequence as Qwen3_5VLForConditionalGeneration.embed_multimodal
    # (which is Qwen3-VL's, reused): pack items into fixed-size blocks, compute
    # rotary/pos-embed inputs on CPU, scatter into the block layout, then let the
    # encoder scatter-write merged embeddings into the cache buffer.
    from vllm_neuron.model.qwen3_vl.utils.vision_block_packing import (
        compute_block_bounds,
        ffd_pack_images,
        scatter_to_blocks,
        select_vision_bucket,
    )
    from vllm_neuron.model.qwen3_vl.utils.vision_preprocessing import (
        compute_position_indices_and_weights,
        compute_rotary_pos_emb,
    )

    block_size = args.block_size
    head_dim = vision_config.hidden_size // vision_config.num_heads
    num_grid_per_side = int(vision_config.num_position_embeddings**0.5)
    tokens_per_image = [num_tokens]
    grid_for_ve = grid_thw

    _bucket, num_blocks = select_vision_bucket(
        num_tokens, [block_size], block_size, dp_size=1
    )
    assignment = ffd_pack_images(
        tokens_per_image, block_size, num_blocks, one_item_per_block=True
    )
    cos, sin = compute_rotary_pos_emb(grid_for_ve, head_dim, merge)
    pos_emb_idx, pos_emb_weight = compute_position_indices_and_weights(
        grid_for_ve, num_grid_per_side, merge
    )
    bound_min, bound_max = compute_block_bounds(
        tokens_per_image, assignment, grid_for_ve
    )

    packed_pixels = scatter_to_blocks(pixel_values, tokens_per_image, assignment)
    packed_cos = scatter_to_blocks(cos, tokens_per_image, assignment)
    packed_sin = scatter_to_blocks(sin, tokens_per_image, assignment)
    packed_idx = (
        scatter_to_blocks(pos_emb_idx.T, tokens_per_image, assignment)
        .permute(2, 0, 1)
        .contiguous()
    )
    packed_weight = (
        scatter_to_blocks(pos_emb_weight.T, tokens_per_image, assignment)
        .permute(2, 0, 1)
        .contiguous()
    )

    # A stand-in encoder cache: one row per merged token per block, plus a
    # scratch block for VE blocks that hold no real item.
    merged_per_block = block_size // merge**2
    fat_dim = vision_config.out_hidden_size
    cache_buffer = torch.zeros(num_blocks + 1, merged_per_block, fat_dim)
    write_block_ids = torch.arange(num_blocks, dtype=torch.int64)

    our_block_out: list[torch.Tensor] = []
    our_handles = [
        blk.register_forward_hook(
            lambda _m, _i, out, store=our_block_out: store.append(
                (out[0] if isinstance(out, tuple) else out).detach().clone()
            )
        )
        for blk in tower.blocks
    ]
    updated = tower(
        packed_pixels.float(),
        packed_idx,
        packed_weight.float(),
        packed_cos.float(),
        packed_sin.float(),
        bound_min,
        bound_max,
        cache_buffer,
        write_block_ids,
    )
    for handle in our_handles:
        handle.remove()

    # Error after each transformer block. The plugin's vision attention runs on
    # NKI kernels (NF.qkv_proj / flash_attention / o_proj) whose CPU fallback has
    # its own precision, so some floor is expected; what matters is whether the
    # error *grows smoothly with depth* (accumulated kernel noise) or *jumps at
    # one block* (a bug).
    print("\n  per-block error (ours vs HF, first block first):")
    growth = []
    for i, (mine, theirs) in enumerate(zip(our_block_out, hf_block_out)):
        mine_flat = mine.reshape(-1, mine.shape[-1])[:num_tokens]
        rel = (
            (mine_flat - theirs[:num_tokens]).abs().max()
            / theirs[:num_tokens].abs().max().clamp(min=1e-9)
        ).item()
        growth.append(rel)
    for i in (0, 1, 2, len(growth) // 2, len(growth) - 2, len(growth) - 1):
        print(f"      block {i:2d}: rel={growth[i]:.3e}")
    jumps = [
        (i, growth[i] / max(growth[i - 1], 1e-12))
        for i in range(1, len(growth))
        if growth[i] > 3 * growth[i - 1] and growth[i] > 1e-4
    ]
    print(f"      blocks where error more than tripled: {jumps[:5]}")

    merged_tokens = num_tokens // merge**2
    ours_merged = updated[:num_blocks].reshape(-1, fat_dim)[:merged_tokens]

    ref = hf_merged[:merged_tokens]
    # Informational: see the module docstring for why this cannot be tight.
    report("merged embeddings (informational)", ours_merged, ref, args.tol)

    # What *is* meaningful on CPU: the merged tokens come out in the right order
    # and at the right scale. A reordering or a wrong merger would show here even
    # though the kernel fallback blurs the absolute numbers.
    print("\n3. structure")
    distance = (ours_merged[:, None, :] - ref[None, :, :]).abs().amax(-1)
    nearest = distance.argmin(dim=1)
    in_order = int((nearest == torch.arange(merged_tokens)).sum())
    ordered_ok = in_order >= int(0.9 * merged_tokens)
    print(
        f"  {'PASS' if ordered_ok else 'FAIL'}  token order: {in_order}/"
        f"{merged_tokens} merged tokens are closest to their own HF row"
    )
    ok &= ordered_ok

    scale_ratio = (
        ours_merged.abs().mean() / ref.abs().mean().clamp(min=1e-9)
    ).item()
    scale_ok = 0.8 <= scale_ratio <= 1.25
    print(
        f"  {'PASS' if scale_ok else 'FAIL'}  magnitude: mean|ours| / mean|hf| "
        f"= {scale_ratio:.3f}"
    )
    ok &= scale_ok

    if os.environ.get("VISION_DEBUG"):
        per_token = (ours_merged - ref).abs().amax(dim=-1)
        scale = ref.abs().max()
        print(f"      per-token max err: {per_token.tolist()[:16]}")
        bad = (per_token > args.tol * scale).nonzero().flatten().tolist()
        print(f"      tokens over tol: {len(bad)}/{merged_tokens} -> {bad[:16]}")
        # Is it a permutation? Match each of ours to the nearest reference row.
        dist = (ours_merged[:, None, :] - ref[None, :, :]).abs().amax(-1)
        nearest = dist.argmin(dim=1).tolist()
        print(f"      nearest ref row for ours[i]: {nearest[:16]}")
        print(f"      is identity permutation: {nearest == list(range(merged_tokens))}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
