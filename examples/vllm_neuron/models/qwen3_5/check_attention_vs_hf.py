#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Numerically validate the Qwen3.5 attention half against HuggingFace, on CPU.

Companion to ``check_deltanet_vs_hf.py``, covering the pieces of the 6
full-attention layers. Each check targets something that would produce fluent
but wrong output rather than an error:

1. **RMSNorm** — Qwen3.5 scales by ``1 + weight`` with a zero-centred
   checkpoint tensor. The usual ``weight * x`` form would multiply activations
   by roughly zero.
2. **Rotary** — rotary covers only 64 of each head's 256 dims, and the three
   mRoPE axes are interleaved ``THWTHW...`` rather than laid out in chunks.
3. **Attention prefill** — the whole layer, on layer 3's real weights,
   including the ``sigmoid(gate)`` on the attention output.
4. **MLP** — SwiGLU.
5. **Attention decode** — replayed token by token against prefill of the same
   sequence, which is what exercises the paged KV cache reads and writes.
6. **TP=4** — through the real weight loaders, against TP=1. This is where a
   wrong KV-replication shard shows up: with 2 KV heads and 4 ranks, ranks 0-1
   must share KV head 0 and ranks 2-3 KV head 1, matching their query heads.

Usage:

    python check_attention_vs_hf.py [--model /mnt/nvme/models/Qwen3.5-2B]
"""

from __future__ import annotations

import argparse
import sys
from unittest import mock

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument(
        "--layer", type=int, default=3, help="full-attention layer to load"
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--tol", type=float, default=2e-5)
    return parser.parse_args()


def report(name: str, ours: torch.Tensor, theirs: torch.Tensor, tol: float) -> bool:
    diff = (ours - theirs).abs()
    scale = theirs.abs().max().clamp(min=1e-12)
    rel = (diff.max() / scale).item()
    ok = rel <= tol
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name}: max|d|={diff.max():.3e} "
        f"rel={rel:.3e} (ref max |x|={scale:.3e})"
    )
    return ok


def make_config(text_config):
    from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig.from_hf(text_config)
    cfg.torch_dtype = torch.float32
    return cfg


def build_attention(cfg, layer_idx: int, tp_size: int = 1, rank: int = 0):
    from vllm_neuron.model.qwen3_5 import model as mod

    group = type(
        "_G", (), {"world_size": tp_size, "rank_in_group": rank, "device_group": None}
    )()
    with mock.patch.object(mod, "get_tp_group", lambda: group):
        return mod.Qwen3_5Attention(cfg, layer_idx)


def load_layer_weights(model_path: str, layer_idx: int, submodule: str):
    from safetensors import safe_open
    import json
    import os

    with open(os.path.join(model_path, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]

    prefix = f"model.language_model.layers.{layer_idx}.{submodule}."
    wanted = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not wanted:
        raise SystemExit(f"no tensors under {prefix!r}")
    out = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(os.path.join(model_path, shard), framework="pt") as handle:
            for key, where in wanted.items():
                if where == shard:
                    out[key[len(prefix) :]] = handle.get_tensor(key).float()
    return out


def fake_metadata(
    num_tokens: int, num_reqs: int, is_decode: bool, block_size: int, num_blocks: int
):
    """Block tables and slot mappings shaped the way the runner builds them."""
    blocks_per_req = num_blocks // max(num_reqs, 1)
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(
        num_reqs, blocks_per_req
    )
    return {
        "block_table_tensor": block_table,
        "slot_mapping": torch.zeros(num_tokens, dtype=torch.int64),
        "max_query_len": 1 if is_decode else num_tokens,
        "decode_token_threshold": 1,
        "block_size": block_size,
    }


def check_norm(args, text_config) -> bool:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm as HFNorm
    from vllm_neuron.model.qwen3_5.model import Qwen3_5RMSNorm

    print("1. RMSNorm (zero-centred weight)")
    torch.manual_seed(0)
    dim = text_config.hidden_size
    weight = torch.randn(dim) * 0.1

    hf = HFNorm(dim, eps=text_config.rms_norm_eps)
    ours = Qwen3_5RMSNorm(dim, text_config.rms_norm_eps, torch.float32)
    with torch.no_grad():
        hf.weight.copy_(weight)
        ours.weight.copy_(weight)
        x = torch.randn(args.seq_len, dim)
        ok = report("norm output", ours(x), hf(x), args.tol)
    # Guard against the mistake this check exists for: with a zero weight the
    # correct form is the identity, the `weight * x` form annihilates.
    with torch.no_grad():
        ours.weight.zero_()
        identity = ours(x)
    ok &= report("zero weight is identity", identity, x / x.pow(2).mean(-1, keepdim=True)
                 .add(text_config.rms_norm_eps).sqrt(), args.tol)
    return ok


def check_rotary(args, cfg, text_config) -> bool:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from vllm_neuron.model.qwen3_5.model import Qwen3_5RotaryEmbedding

    print("2. partial interleaved mRoPE")
    hf = Qwen3_5TextRotaryEmbedding(text_config)
    ours = Qwen3_5RotaryEmbedding(cfg)

    # 3D positions with the axes genuinely differing, so an axis mix-up shows.
    torch.manual_seed(1)
    t = torch.arange(args.seq_len)
    positions_3d = torch.stack([t, t // 2, t // 3])

    with torch.no_grad():
        hf_cos, hf_sin = hf(torch.zeros(1, args.seq_len, 1), positions_3d.unsqueeze(1))
        our_cos, our_sin = ours(positions_3d, dtype=torch.float32)
    # HF returns the doubled cat((freqs, freqs)); ours returns the half that
    # apply_partial_rotary doubles itself.
    half = hf_cos.shape[-1] // 2
    ok = report("cos", our_cos, hf_cos[0, :, :half], args.tol)
    ok &= report("sin", our_sin, hf_sin[0, :, :half], args.tol)
    print(f"    rotary_dim={cfg.rotary_dim} of head_dim={cfg.head_dim}, "
          f"mrope_section={list(cfg.mrope_section)}")
    return ok


def check_attention(args, cfg, text_config, hf_weights) -> bool:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention as HFAttention,
        Qwen3_5TextRotaryEmbedding,
    )

    print("3. attention prefill vs HF module")
    hf = HFAttention(text_config, args.layer).float().eval()
    missing, unexpected = hf.load_state_dict(hf_weights, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected}")
    if missing:
        print(f"    (left at init: {missing})")

    ours = build_attention(cfg, args.layer)
    with torch.no_grad():
        ours.q_proj_weight.copy_(hf_weights["q_proj.weight"].T)
        ours.k_proj_weight.copy_(hf_weights["k_proj.weight"].T)
        ours.v_proj_weight.copy_(hf_weights["v_proj.weight"].T)
        ours.o_proj_weight.copy_(hf_weights["o_proj.weight"].T)
        ours.q_norm.weight.copy_(hf_weights["q_norm.weight"])
        ours.k_norm.weight.copy_(hf_weights["k_norm.weight"])

    block_size = 32
    num_blocks = -(-args.seq_len // block_size) + 1
    ours.k_cache = torch.zeros(
        num_blocks, ours.num_kv_heads_per_rank, block_size, ours.head_dim
    )
    ours.v_cache = torch.zeros_like(ours.k_cache)

    torch.manual_seed(2)
    hidden = torch.randn(args.seq_len, cfg.hidden_size) * 0.05
    t = torch.arange(args.seq_len)
    positions_3d = torch.stack([t, t, t])

    rotary = Qwen3_5TextRotaryEmbedding(text_config)
    with torch.no_grad():
        cos, sin = rotary(hidden.unsqueeze(0), positions_3d.unsqueeze(1))
        hf_out = hf(
            hidden.unsqueeze(0),
            position_embeddings=(cos, sin),
            attention_mask=torch.triu(
                torch.full((args.seq_len, args.seq_len), float("-inf")), diagonal=1
            ).view(1, 1, args.seq_len, args.seq_len),
        )[0].squeeze(0)

    metadata = fake_metadata(
        args.seq_len, 1, False, block_size, num_blocks
    )
    metadata["slot_mapping"] = t.to(torch.int64)
    half = cos.shape[-1] // 2
    with torch.no_grad():
        our_out = ours.forward_prefill(
            hidden, (cos[0, :, :half], sin[0, :, :half]), metadata
        )
    return report("prefill output", our_out, hf_out, args.tol)


def check_mlp(args, cfg, text_config) -> bool:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP as HFMLP
    from vllm_neuron.model.qwen3_5 import model as mod

    print("4. MLP")
    hf_weights = load_layer_weights(args.model, args.layer, "mlp")
    hf = HFMLP(text_config, text_config.intermediate_size).float().eval()
    hf.load_state_dict(hf_weights)

    group = type("_G", (), {"world_size": 1, "rank_in_group": 0})()
    with mock.patch.object(mod, "get_tp_group", lambda: group):
        ours = mod.Qwen3_5MLP(cfg)
    with torch.no_grad():
        ours.gate_proj_weight.copy_(hf_weights["gate_proj.weight"].T)
        ours.up_proj_weight.copy_(hf_weights["up_proj.weight"].T)
        ours.down_proj_weight.copy_(hf_weights["down_proj.weight"].T)

    torch.manual_seed(3)
    hidden = torch.randn(args.seq_len, cfg.hidden_size) * 0.05
    with torch.no_grad():
        return report(
            "mlp output",
            ours(hidden, is_prefill=True),
            hf(hidden.unsqueeze(0)).squeeze(0),
            args.tol,
        )


def check_decode(args, cfg, text_config, hf_weights) -> bool:
    """Decode step by step must reproduce prefill over the same tokens.

    Prefill is checked against HF above, so this transitively covers decode and
    additionally exercises the paged KV cache: prefill scatters through
    ``slot_mapping`` while decode gathers through the block table.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    print("5. attention decode replay vs prefill")
    steps = 64
    block_size = 32
    num_blocks = steps // block_size

    ours = build_attention(cfg, args.layer)
    with torch.no_grad():
        ours.q_proj_weight.copy_(hf_weights["q_proj.weight"].T)
        ours.k_proj_weight.copy_(hf_weights["k_proj.weight"].T)
        ours.v_proj_weight.copy_(hf_weights["v_proj.weight"].T)
        ours.o_proj_weight.copy_(hf_weights["o_proj.weight"].T)
        ours.q_norm.weight.copy_(hf_weights["q_norm.weight"])
        ours.k_norm.weight.copy_(hf_weights["k_norm.weight"])

    torch.manual_seed(4)
    hidden = torch.randn(steps, cfg.hidden_size) * 0.05
    t = torch.arange(steps)
    positions_3d = torch.stack([t, t, t])
    rotary = Qwen3_5TextRotaryEmbedding(text_config)
    with torch.no_grad():
        cos, sin = rotary(hidden.unsqueeze(0), positions_3d.unsqueeze(1))
    half = cos.shape[-1] // 2
    cos, sin = cos[0, :, :half], sin[0, :, :half]

    ours.k_cache = torch.zeros(
        num_blocks, ours.num_kv_heads_per_rank, block_size, ours.head_dim
    )
    ours.v_cache = torch.zeros_like(ours.k_cache)
    prefill_meta = fake_metadata(steps, 1, False, block_size, num_blocks)
    prefill_meta["slot_mapping"] = t.to(torch.int64)
    with torch.no_grad():
        expected = ours.forward_prefill(hidden, (cos, sin), prefill_meta)

    ours.k_cache.zero_()
    ours.v_cache.zero_()
    outs = []
    with torch.no_grad():
        for step in range(steps):
            meta = fake_metadata(1, 1, True, block_size, num_blocks)
            meta["slot_mapping"] = torch.tensor([step], dtype=torch.int64)
            outs.append(
                ours.forward_decode(
                    hidden[step : step + 1],
                    torch.tensor([step]),
                    (cos[step : step + 1], sin[step : step + 1]),
                    meta,
                )
            )
    return report("decode outputs", torch.cat(outs, dim=0), expected, args.tol)


def check_tp(args, cfg, text_config, hf_weights, tp_size: int = 4) -> bool:
    from vllm_neuron.utils.weight_loader import get_weight_loader

    print(f"6. TP={tp_size} sharding vs TP=1")

    class _Slice:
        def __init__(self, tensor):
            self._t = tensor

        def get_shape(self):
            return tuple(self._t.shape)

        def __getitem__(self, key):
            return self._t[key]

    sources = {
        "q_proj_weight": ["q_proj.weight"],
        "k_proj_weight": ["k_proj.weight"],
        "v_proj_weight": ["v_proj.weight"],
        "o_proj_weight": ["o_proj.weight"],
        "q_norm.weight": ["q_norm.weight"],
        "k_norm.weight": ["k_norm.weight"],
    }

    block_size = 32
    num_blocks = -(-args.seq_len // block_size)
    torch.manual_seed(5)
    hidden = torch.randn(args.seq_len, cfg.hidden_size) * 0.05
    slot_mapping = torch.arange(args.seq_len, dtype=torch.int64)

    def run(module):
        module.k_cache = torch.zeros(
            num_blocks, module.num_kv_heads_per_rank, block_size, module.head_dim
        )
        module.v_cache = torch.zeros_like(module.k_cache)
        meta = fake_metadata(args.seq_len, 1, False, block_size, num_blocks)
        meta["slot_mapping"] = slot_mapping
        cos = torch.zeros(args.seq_len, cfg.rotary_dim // 2)
        sin = torch.zeros_like(cos)
        # Zero cos/sin would scale the rotated half to zero; use a real angle so
        # the rotary path stays exercised.
        angles = torch.arange(args.seq_len).float().unsqueeze(-1) * 0.01
        cos, sin = torch.cos(angles + cos), torch.sin(angles + sin)
        with torch.no_grad():
            return module.forward_prefill(hidden, (cos, sin), meta)

    reference = build_attention(cfg, args.layer)
    with torch.no_grad():
        for name, keys in sources.items():
            param = reference.get_parameter(name)
            param.copy_(get_weight_loader(param).load(
                [_Slice(hf_weights[k]) for k in keys], 0
            ))
    expected = run(reference)

    total = torch.zeros_like(expected)
    for rank in range(tp_size):
        shard = build_attention(cfg, args.layer, tp_size=tp_size, rank=rank)
        shard.world_size = 1  # combine here rather than through collectives
        with torch.no_grad():
            for name, keys in sources.items():
                param = shard.get_parameter(name)
                param.copy_(get_weight_loader(param).load(
                    [_Slice(hf_weights[k]) for k in keys], rank
                ))
        total = total + run(shard)
    return report("summed rank outputs", total, expected, args.tol)


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)

    from transformers import AutoConfig

    text_config = AutoConfig.from_pretrained(args.model).text_config
    layer_type = text_config.layer_types[args.layer]
    print(f"checkpoint: {args.model}")
    print(f"layer {args.layer} is {layer_type!r}\n")
    if layer_type != "full_attention":
        raise SystemExit(
            f"layer {args.layer} is {layer_type}; full attention lives at "
            f"{[i for i, t in enumerate(text_config.layer_types) if t == 'full_attention']}"
        )

    cfg = make_config(text_config)
    hf_weights = load_layer_weights(args.model, args.layer, "self_attn")

    ok = check_norm(args, text_config)
    ok &= check_rotary(args, cfg, text_config)
    ok &= check_attention(args, cfg, text_config, hf_weights)
    ok &= check_mlp(args, cfg, text_config)
    ok &= check_decode(args, cfg, text_config, hf_weights)
    ok &= check_tp(args, cfg, text_config, hf_weights)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
