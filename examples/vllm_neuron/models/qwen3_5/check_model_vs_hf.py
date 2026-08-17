#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Diff the whole Qwen3.5 text stack against HuggingFace, layer by layer, on CPU.

``check_deltanet_vs_hf.py`` and ``check_attention_vs_hf.py`` validate the two
mixer kinds in isolation. This validates everything *around* them — the
embedding, the residual structure, the 24-layer composition, the final norm and
the LM head — by running a prefill through both implementations and reporting the
first layer whose hidden state diverges.

That is much cheaper than bisecting on device: a device run costs ~8 minutes,
this costs seconds, and it names the offending layer directly instead of leaving
a guess.

TP=1 on CPU, float32. What it therefore cannot catch is anything that only
appears under tensor/sequence parallelism or in the runner's own input
preparation — if this passes, those are what is left.

Usage:

    PYTHONPATH=/mnt/nvme/vllm-neuron python check_model_vs_hf.py
"""

from __future__ import annotations

import argparse
import sys
from unittest import mock

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--bucket",
        type=int,
        default=64,
        help="pad the prompt to this many tokens, as the runner would",
    )
    parser.add_argument("--tol", type=float, default=1e-3)
    return parser.parse_args()


class _SingleRankGroup:
    world_size = 1
    rank_in_group = 0
    device_group = None


class _GlooTPGroup:
    """The slice of vLLM's ``GroupCoordinator`` the model uses, over gloo.

    Semantics deliberately match vLLM's: ``all_gather`` and ``reduce_scatter``
    return new tensors, ``all_reduce`` sums in place (the plugin's models rely on
    that and ignore the return value).
    """

    def __init__(self, world_size: int, rank: int):
        import torch.distributed as dist

        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = dist.group.WORLD

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        import torch.distributed as dist

        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.cat(parts, dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        import torch.distributed as dist

        summed = tensor.contiguous().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        return summed.chunk(self.world_size, dim=dim)[self.rank_in_group].contiguous()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor


def init_single_rank_distributed() -> None:
    """A world of one, on gloo.

    The checkpoint loader reaches for the default process group's store even at
    TP=1, and ``VocabDimShardedEmbedding`` keys its sharding off
    ``dist.is_initialized()``. A single-rank gloo group satisfies both without
    changing any arithmetic.
    """
    import os

    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def build_ours(model_path: str, group=None):
    """Instantiate the full text model in float32, with real weights."""
    from transformers import AutoConfig
    from vllm_neuron.model.qwen3_5 import deltanet as dn
    from vllm_neuron.model.qwen3_5 import model as mod
    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    hf_config = AutoConfig.from_pretrained(model_path)
    config = Qwen3_5Config.from_hf(hf_config, include_vision=False)
    # float32 so any mismatch is a bug rather than bf16 noise.
    config.text_config.torch_dtype = torch.float32

    group = group if group is not None else _SingleRankGroup()
    with (
        mock.patch.object(mod, "get_tp_group", lambda: group),
        mock.patch.object(dn, "get_tp_group", lambda: group),
    ):
        model = mod.Qwen3_5ForCausalLM(config)
    model.load_weights(model_path, torch.device("cpu"), None)
    return model.eval(), config


def build_hf(model_path: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, config=hf_config
    )
    # ``AutoModelForCausalLM`` on this checkpoint gives the causal-LM head over
    # the text stack directly, but the same weights are also reachable through
    # the VL wrapper as ``model.model.language_model``. Accept either.
    inner = model.model
    text_model = getattr(inner, "language_model", inner)
    return model.eval(), text_model


def report(name: str, ours: torch.Tensor, theirs: torch.Tensor, tol: float) -> bool:
    diff = (ours - theirs).abs()
    scale = theirs.abs().max().clamp(min=1e-12)
    rel = (diff.max() / scale).item()
    ok = rel <= tol
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name:28s} max|d|={diff.max():.3e} "
        f"rel={rel:.3e}"
    )
    return ok


def fake_metadata(text_config, num_tokens: int, block_size: int, mamba_blocks: int):
    """Prefill metadata for one sequence, keyed per layer like the runner's."""
    blocks_per_seq = -(-num_tokens // block_size)
    attn_meta = {
        "block_table_tensor": torch.arange(blocks_per_seq, dtype=torch.int32).view(
            1, blocks_per_seq
        ),
        "slot_mapping": torch.arange(num_tokens, dtype=torch.int64),
        "max_query_len": num_tokens,
        "decode_token_threshold": 1,
        "block_size": block_size,
    }
    state_meta = dict(attn_meta)
    state_meta["block_table_tensor"] = torch.zeros(1, 1, dtype=torch.int32)
    metadata = {}
    for i, kind in enumerate(text_config.layer_types):
        if kind == "linear_attention":
            metadata[f"layers.{i}.linear_attn"] = state_meta
        else:
            metadata[f"layers.{i}.self_attn"] = attn_meta
    return metadata


def bind_caches(model, config, block_size: int, num_blocks: int, mamba_blocks: int = 2):
    caches = {}
    tc = config.text_config
    for layer in model.language_model.layers:
        if layer.is_linear_attention:
            conv_shape, rec_shape = tc.state_shapes(1)
            caches[layer.linear_attn.layer_name] = [
                torch.zeros(mamba_blocks, *conv_shape, dtype=torch.float32),
                torch.zeros(mamba_blocks, *rec_shape, dtype=torch.float32),
            ]
        else:
            attn = layer.self_attn
            shape = (num_blocks, attn.num_kv_heads_per_rank, block_size, attn.head_dim)
            caches[attn.layer_name] = [torch.zeros(shape), torch.zeros(shape)]
    model.bind_kv_cache(caches)


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    init_single_rank_distributed()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    real = len(ids)
    num_tokens = max(args.bucket, real)

    # Pad the way the runner does: token id 0 appended, positions frozen at the
    # last real value.
    input_ids = torch.cat([ids, torch.zeros(num_tokens - real, dtype=ids.dtype)])
    positions = torch.cat(
        [torch.arange(real), torch.full((num_tokens - real,), real - 1)]
    )
    rotary_position_ids = torch.cat(
        [
            torch.arange(real).unsqueeze(0).expand(3, -1),
            torch.zeros(3, num_tokens - real, dtype=torch.int64),
        ],
        dim=1,
    )

    print(f"checkpoint: {args.model}")
    print(f"prompt: {args.prompt!r} -> {real} tokens, padded to {num_tokens}\n")

    ours, config = build_ours(args.model)
    hf_full, hf_text = build_hf(args.model)

    block_size = 288  # what the hybrid page alignment picks at runtime
    num_blocks = -(-num_tokens // block_size) + 1
    bind_caches(ours, config, block_size, num_blocks)
    metadata = fake_metadata(config.text_config, num_tokens, block_size, 2)

    # ── HF reference, capturing every layer's output ─────────────────────
    hf_states = []
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out, store=hf_states: store.append(
                (out[0] if isinstance(out, tuple) else out).squeeze(0).clone()
            )
        )
        for layer in hf_text.layers
    ]
    hf_out = hf_text(input_ids=ids.unsqueeze(0))
    for handle in handles:
        handle.remove()
    hf_final = hf_out.last_hidden_state.squeeze(0)

    # ── Ours, same capture ───────────────────────────────────────────────
    our_states = []
    lm = ours.language_model
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out, store=our_states: store.append(out.clone())
        )
        for layer in lm.layers
    ]
    hidden = lm(
        input_ids,
        positions,
        rotary_position_ids,
        metadata,
        rank=None,
    )
    for handle in handles:
        handle.remove()

    print("per-layer hidden states (first divergence is the bug):")
    ok = True
    layer_types = config.text_config.layer_types
    for i, (mine, theirs) in enumerate(zip(our_states, hf_states)):
        kind = "delta" if layer_types[i] == "linear_attention" else "attn "
        passed = report(f"layer {i:2d} ({kind})", mine[:real], theirs[:real], args.tol)
        ok &= passed
        if not passed:
            print(f"\n  --> first divergence at layer {i} ({layer_types[i]})")
            break

    if ok:
        ok &= report("final norm", hidden[:real], hf_final[:real], args.tol)
        our_logits = ours.lm_head(hidden[:real])
        hf_logits = hf_full.lm_head(hf_final[:real].unsqueeze(0)).squeeze(0)
        ok &= report("logits", our_logits, hf_logits, args.tol)
        mine = our_logits[-1].argmax().item()
        theirs = hf_logits[-1].argmax().item()
        print(
            f"\n  greedy next token: ours={tokenizer.decode([mine])!r} "
            f"hf={tokenizer.decode([theirs])!r}"
        )
        ok &= mine == theirs

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
