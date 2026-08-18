#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compile the whole Qwen3.5 text stack on the Neuron device, diffed against CPU.

``probe_device_ops.py`` covers one module at a time, which is not enough: the
engine's compile failure (``NCC_IBTN006``, a ``pftranspose`` whose copy fails
backend verification) does not reproduce for either mixer on its own. This builds
a real ``Qwen3_5TextModel`` — embedding, both mixer kinds, MLPs, norms — with
random weights, so it can be scaled from a few layers up to the full 24 and run
in the same compiled-graph shape the runner uses.

Random weights are the point: this asks whether the *compiler* can handle the
graph, and loading 4.3 GB per configuration would make the loop too slow to
bisect with. Use ``check_model_vs_hf.py`` for numerics against HF.

Usage:

    NEURON_SKIP_EFA_AFFINITY=1 VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm \
    NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp" \
    PYTHONPATH=/mnt/nvme/vllm-neuron PATH=$V/bin:$PATH $V/bin/python \
      probe_device_model.py --layers 4 [--tp 4] [--dtype bf16] [--phase prefill]
"""

from __future__ import annotations

import argparse
import os
import sys
from unittest import mock

import torch

DEVICE = "neuron:0"
BLOCK_SIZE = 288  # what the hybrid page alignment picks at runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="how many decoder layers (4 gives one [linear x3, full] cycle)",
    )
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--reqs", type=int, default=4, help="decode batch size")
    parser.add_argument("--tp", type=int, default=4, help="TP shapes to build for")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument(
        "--phase", default="prefill", choices=("prefill", "decode", "both")
    )
    parser.add_argument("--tol", type=float, default=5e-2)
    parser.add_argument(
        "--time",
        type=int,
        default=0,
        help="after checking correctness, run the compiled graph this many times "
        "and report mean wall time. Combine with "
        "VLLM_NEURON_QWEN35_ABLATE_MIXERS to attribute prefill cost to the two "
        "mixer kinds.",
    )
    parser.add_argument(
        "--state-views",
        action="store_true",
        help="allocate the DeltaNet state as offset views into one raw buffer per "
        "layer, exactly as the runner's initialize_kv_cache does, instead of as "
        "standalone tensors",
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="append the sampling index_select + vocab-sharded LM head, as the "
        "engine's graph does",
    )
    return parser.parse_args()


def init_single_rank_distributed() -> None:
    """The embedding and LM head need a process group even at TP=1."""
    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29597")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


class _Group:
    """Stand-in for vLLM's TP group.

    ``world_size`` is what the modules shard their weights by, so passing 4 here
    reproduces the real per-rank shapes. Collectives are never called because the
    caller sets ``world_size = 1`` on the built modules afterwards.
    """

    def __init__(self, world_size: int, rank: int, device_group=None):
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = device_group


def build(args, device: str, dtype: torch.dtype):
    """A text model with ``args.layers`` layers and random weights, on ``device``."""
    import torch.distributed as dist
    from transformers import AutoConfig

    from vllm_neuron.model.qwen3_5 import deltanet as dn
    from vllm_neuron.model.qwen3_5 import model as mod
    from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig

    text_config = AutoConfig.from_pretrained(args.model).text_config
    cfg = Qwen3_5TextConfig.from_hf(text_config)
    cfg.torch_dtype = dtype
    cfg.num_hidden_layers = args.layers
    cfg.layer_types = tuple(cfg.layer_types[: args.layers])

    # The embedding/head shard by the *real* group; the mixers shard by ``--tp``.
    single = _Group(1, 0, dist.group.WORLD)
    sharded = _Group(args.tp, 0, dist.group.WORLD)

    def pick():
        # ``Qwen3_5TextModel`` and the LM head want the real (single-rank) group so
        # no collective is needed; the mixers and MLP want the sharded shapes.
        frame = sys._getframe(1)
        owner = frame.f_locals.get("self")
        name = type(owner).__name__ if owner is not None else ""
        return single if name in ("Qwen3_5TextModel",) else sharded

    with (
        mock.patch.object(mod, "get_tp_group", pick),
        mock.patch.object(dn, "get_tp_group", pick),
    ):
        model = mod.Qwen3_5TextModel(cfg)

    torch.manual_seed(4)
    for param in model.parameters():
        param.data = (torch.randn(param.shape, dtype=torch.float32) * 0.02).to(
            param.dtype
        )
    for layer in model.layers:
        mixer = layer.linear_attn if layer.is_linear_attention else layer.self_attn
        mixer.world_size = 1
        layer.mlp.world_size = 1
    model.world_size = 1

    return model.to(device).eval(), cfg


def bind_caches(model, cfg, device, dtype, seq, blocks_per_req, state_views=False):
    num_blocks = blocks_per_req + 1
    for layer in model.layers:
        if layer.is_linear_attention:
            mixer = layer.linear_attn
            conv_shape = (8, cfg.linear_conv_kernel_dim - 1, mixer.conv_dim)
            rec_shape = (8, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim)
            if state_views:
                # Mirror initialize_kv_cache: one raw byte buffer per layer, with
                # each state tensor a view at an increasing offset. The second
                # view therefore has a non-zero storage offset.
                conv_elems = int(torch.tensor(conv_shape).prod())
                rec_elems = int(torch.tensor(rec_shape).prod())
                raw = torch.zeros(
                    conv_elems + rec_elems, dtype=torch.float32, device=device
                )
                mixer.conv_state = raw[:conv_elems].view(conv_shape)
                mixer.recurrent_state = raw[
                    conv_elems : conv_elems + rec_elems
                ].view(rec_shape)
            else:
                mixer.conv_state = torch.zeros(
                    *conv_shape, device=device, dtype=torch.float32
                )
                mixer.recurrent_state = torch.zeros(
                    *rec_shape, device=device, dtype=torch.float32
                )
        else:
            attn = layer.self_attn
            shape = (num_blocks, attn.num_kv_heads_per_rank, BLOCK_SIZE, attn.head_dim)
            attn.k_cache = torch.zeros(shape, device=device, dtype=dtype)
            attn.v_cache = torch.zeros(shape, device=device, dtype=dtype)


def make_metadata(model, num_tokens, num_reqs, is_decode, device, blocks_per_req):
    """Metadata keyed per layer, as the runner builds it.

    Block tables are built on CPU then moved: ``expand(...).contiguous()`` is not
    supported on device tensors.
    """
    block_table = (
        torch.arange(blocks_per_req, dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_reqs, blocks_per_req)
        .contiguous()
        .to(device)
    )
    entry = {
        "block_table_tensor": block_table,
        "slot_mapping": torch.arange(num_tokens, dtype=torch.int64).to(device),
        "max_query_len": 1 if is_decode else num_tokens,
        "decode_token_threshold": 1,
        "block_size": BLOCK_SIZE,
    }
    state_entry = dict(entry)
    state_entry["block_table_tensor"] = (
        torch.arange(num_reqs, dtype=torch.int32).reshape(num_reqs, 1).to(device)
    )
    return {
        layer.mixer_name: state_entry if layer.is_linear_attention else entry
        for layer in model.layers
    }


def run_phase(args, phase: str, dtype: torch.dtype) -> str:
    from vllm_neuron.compile.backend import compile as neuron_compile

    is_decode = phase == "decode"
    num_tokens = args.reqs if is_decode else args.seq
    num_reqs = args.reqs if is_decode else 1
    blocks_per_req = -(-args.seq // BLOCK_SIZE)

    torch.manual_seed(7)
    input_ids = torch.randint(0, 10000, (num_tokens,), dtype=torch.int32)
    if is_decode:
        positions = torch.arange(num_tokens, dtype=torch.int32)
    else:
        real = num_tokens - 37
        positions = torch.cat(
            [torch.arange(real), torch.full((num_tokens - real,), real - 1)]
        ).to(torch.int32)
    rotary = positions.unsqueeze(0).expand(3, -1).contiguous().to(torch.int64)

    outputs = {}
    for device in ("cpu", DEVICE):
        model, cfg = build(args, device, dtype)
        bind_caches(
            model, cfg, device, dtype, args.seq, blocks_per_req, args.state_views
        )
        metadata = make_metadata(
            model, num_tokens, num_reqs, is_decode, device, blocks_per_req
        )

        head = None
        if args.head:
            import vllm_neuron.nn as neuron_nn
            import torch.distributed as dist

            head = neuron_nn.ColumnParallelLinear(
                cfg.hidden_size,
                cfg.vocab_size,
                bias=False,
                dtype=dtype,
                gather_output=False,
                tp_group=dist.group.WORLD,
            )
            torch.manual_seed(9)
            head.weight.data = (
                torch.randn(head.weight.shape, dtype=torch.float32) * 0.01
            ).to(dtype)
            head = head.to(device)
        sampling_positions = torch.arange(num_reqs, dtype=torch.int64).to(device)

        def call(ids, pos, rot):
            hidden = model(ids, pos, rot, metadata, rank=None)
            if head is None:
                return hidden
            hidden = torch.index_select(hidden, 0, sampling_positions)
            return head(hidden)

        moved = (input_ids.to(device), positions.to(device), rotary.to(device))
        if device == "cpu":
            outputs["ref"] = call(*moved)
            continue
        try:
            compiled = torch.compile(call, backend=neuron_compile, dynamic=False)
            outputs["got"] = compiled(*moved)
            compiled_call, compiled_args = compiled, moved
        except Exception as exc:
            lines = str(exc).strip().splitlines()
            print(f"  FAIL   {phase}: {lines[0][:120] if lines else type(exc).__name__}")
            return "FAIL"

    if args.time:
        import time as _time

        # One untimed call first: the compiled graph is already built by the
        # correctness run above, but the first execution still pays warm-up.
        compiled_call(*compiled_args).cpu()
        started = _time.perf_counter()
        for _ in range(args.time):
            # Sync every iteration: the Neuron runtime queue is shallow and
            # unsynchronised submissions fail with "Execution Queue Full". The
            # device->host copy is therefore included, but it is a few MB against
            # a ~100 ms graph.
            compiled_call(*compiled_args).cpu()
        elapsed = (_time.perf_counter() - started) * 1000.0 / args.time
        ablate = os.environ.get("VLLM_NEURON_QWEN35_ABLATE_MIXERS", "") or "none"
        print(
            f"  TIME   {phase}: {elapsed:8.2f} ms/call "
            f"(layers={args.layers}, seq={args.seq}, ablate={ablate})"
        )

    ref = outputs["ref"].float()
    got = outputs["got"].cpu().float()
    scale = ref.abs().max().clamp(min=1e-9)
    rel = ((got - ref).abs().max() / scale).item()
    verdict = "OK" if rel <= args.tol else "WRONG"
    print(f"  {verdict:6s} {phase}: rel={rel:.3e}")
    return verdict


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    init_single_rank_distributed()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    phases = ("prefill", "decode") if args.phase == "both" else (args.phase,)
    print(
        f"layers={args.layers} tp={args.tp} dtype={args.dtype} "
        f"seq={args.seq} on {DEVICE}\n"
    )

    results = {p: run_phase(args, p, dtype) for p in phases}
    bad = {p: v for p, v in results.items() if v != "OK"}
    print("\n" + (f"PROBLEMS: {bad}" if bad else "ALL OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
