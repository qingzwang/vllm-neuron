#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run the Qwen3.5 text stack at TP=4 on CPU over gloo, and diff against HF.

``check_model_vs_hf.py`` proves the stack is correct at TP=1, which leaves
tensor/sequence parallelism as the untested half: every ``all_gather`` /
``reduce_scatter`` around the mixers and the MLP, the sequence-parallel token
scatter in the embedding, and the vocab-sharded LM head.

This spawns four real gloo ranks on CPU and runs the same prefill, so those
collectives execute for real. Rank 0 reports the per-layer diff against HF, and
the four ranks' logit shards are concatenated to check the vocab sharding.

Costs seconds; the equivalent on device costs ~8 minutes per attempt.

Usage:

    PYTHONPATH=/mnt/nvme/vllm-neuron python check_tp_vs_hf.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.multiprocessing as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument(
        "--bucket",
        type=int,
        default=64,
        help="pad the prompt to this many tokens; must be divisible by --tp",
    )
    parser.add_argument("--tol", type=float, default=1e-3)
    return parser.parse_args()


def _worker(rank: int, world_size: int, args: argparse.Namespace, out_queue) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29593"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.set_grad_enabled(False)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_model_vs_hf import _GlooTPGroup, bind_caches, build_ours, fake_metadata

    group = _GlooTPGroup(world_size, rank)
    model, config = build_ours(args.model, group=group)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    real = len(ids)
    num_tokens = max(args.bucket, real)
    if num_tokens % world_size:
        raise SystemExit(f"--bucket {num_tokens} must be divisible by tp {world_size}")

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

    block_size = 288  # what the hybrid page alignment picks at runtime
    num_blocks = -(-num_tokens // block_size) + 1
    bind_caches(model, config, block_size, num_blocks)
    metadata = fake_metadata(config.text_config, num_tokens, block_size, 2)

    # Capture each layer's output. Under SP a layer returns this rank's token
    # shard, so gather to full sequence order before comparing.
    captured: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out, store=captured: store.append(
                group.all_gather(out.clone(), dim=0)
            )
        )
        for layer in model.language_model.layers
    ]
    hidden = model.language_model(
        input_ids,
        positions.to(torch.int32),
        rotary_position_ids,
        metadata,
        rank=torch.tensor([rank]),
    )
    for handle in handles:
        handle.remove()

    # lm_head is column-parallel with gather_output=False, so each rank holds a
    # vocab shard. Concatenate them in rank order to rebuild full logits.
    logits_shard = model.lm_head(hidden[:real])
    full_logits = group.all_gather(logits_shard.contiguous(), dim=-1)

    if rank == 0:
        out_queue.put(
            {
                "layers": [t.clone() for t in captured],
                "final": hidden.clone(),
                "logits": full_logits.clone(),
                "real": real,
            }
        )
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_model_vs_hf import build_hf, report

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]

    print(f"checkpoint: {args.model}")
    print(f"prompt: {args.prompt!r} -> {len(ids)} tokens, TP={args.tp}\n")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, args.tp, args, queue))
        for rank in range(args.tp)
    ]
    for proc in procs:
        proc.start()
    result = queue.get(timeout=900)
    for proc in procs:
        proc.join(timeout=120)
    if any(proc.exitcode not in (0, None) for proc in procs):
        print(f"worker exit codes: {[p.exitcode for p in procs]}")

    hf_full, hf_text = build_hf(args.model)
    hf_states: list[torch.Tensor] = []
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

    real = result["real"]
    print("per-layer hidden states at TP=%d (first divergence is the bug):" % args.tp)
    ok = True
    from transformers import AutoConfig

    layer_types = AutoConfig.from_pretrained(args.model).text_config.layer_types
    for i, (mine, theirs) in enumerate(zip(result["layers"], hf_states)):
        kind = "delta" if layer_types[i] == "linear_attention" else "attn "
        passed = report(f"layer {i:2d} ({kind})", mine[:real], theirs[:real], args.tol)
        ok &= passed
        if not passed:
            print(f"\n  --> first divergence at layer {i} ({layer_types[i]})")
            break

    if ok:
        hf_logits = hf_full.lm_head(hf_final[:real].unsqueeze(0)).squeeze(0)
        ok &= report("logits (vocab-gathered)", result["logits"], hf_logits, args.tol)
        mine = result["logits"][-1].argmax().item()
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
