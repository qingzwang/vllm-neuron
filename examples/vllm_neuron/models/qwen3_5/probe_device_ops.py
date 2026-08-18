#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compile and run individual Qwen3.5 ops on the Neuron device, diffed against CPU.

The CPU checks prove the maths; this proves the *compiler* agrees. It exists
because bisecting through the engine costs ~9 minutes per attempt and only ever
says "the whole model is wrong", whereas this compiles one function at a time in
seconds and says which one.

It works by handing the plugin's own ``vllm_neuron.compile.backend.compile`` to
``torch.compile`` and putting the inputs on ``neuron:0`` — the same path the
model runner takes, so a failure here is a failure there.

Each probe reports one of:

    OK     compiled, and matches CPU
    WRONG  compiled, but disagrees with CPU  -> a codegen miscompile
    FAIL   did not compile                   -> the error is printed

What it has caught, so that the probes are not mistaken for busywork:

* ``Tensor.split(sizes, dim=-1)`` compiles and returns unrelated data (relative
  error 1.4) while ``t[..., a:b]``, ``torch.chunk`` and reshape-then-index are all
  exact. This was the model's actual bug — all 18 DeltaNet layers were wrong.
* A data-dependent ``index_select`` (index from a device-side ``sum``) also
  miscompiles, which is why ``_tail_rows`` selects by arithmetic instead.
* A rank-5 ``permute`` crashes the compiler outright (``NCC_IBTN006``, a
  ``pftranspose`` whose copy fails backend verification).

Two harness gotchas worth knowing: ``expand(...).contiguous()`` is unsupported on
device tensors (build on CPU, then move), and a module's rotary embedding cannot
run eagerly on device (``Expected self.dtype() == dst.dtype()``), so precompute
``cos``/``sin`` on CPU.

Usage (needs the device free — stop any engine first):

    NEURON_SKIP_EFA_AFFINITY=1 VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm \
    NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp" \
    PYTHONPATH=/mnt/nvme/vllm-neuron PATH=$V/bin:$PATH $V/bin/python \
      probe_device_ops.py [--only NAME]
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch

DEVICE = "neuron:0"
# Real per-rank DeltaNet geometry at TP=4, and the plugin's prefill bucket.
HEADS, SEQ, DK, DV, CHUNK = 4, 1024, 128, 128, 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="run just this probe")
    parser.add_argument("--seq", type=int, default=SEQ)
    parser.add_argument("--tol", type=float, default=2e-2)
    return parser.parse_args()


def run_probe(name: str, fn, cpu_args: tuple, tol: float) -> str:
    """Compile ``fn`` for the device, compare against eager CPU."""
    import vllm_neuron  # noqa: F401  (registers the platform + device)
    from vllm_neuron.compile.backend import compile as neuron_compile

    try:
        expected = fn(*cpu_args)
    except Exception:
        print(f"  ERROR  {name}: CPU reference raised")
        traceback.print_exc()
        return "ERROR"

    device_args = tuple(
        a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in cpu_args
    )
    try:
        compiled = torch.compile(fn, backend=neuron_compile, dynamic=False)
        got = compiled(*device_args)
    except Exception as exc:
        first = str(exc).strip().splitlines()
        print(f"  FAIL   {name}: {first[0] if first else type(exc).__name__}")
        return "FAIL"

    expected_list = expected if isinstance(expected, tuple) else (expected,)
    got_list = got if isinstance(got, tuple) else (got,)

    worst = 0.0
    for ref, out in zip(expected_list, got_list):
        out = out.cpu().float()
        ref = ref.float()
        scale = ref.abs().max().clamp(min=1e-9)
        worst = max(worst, ((out - ref).abs().max() / scale).item())

    verdict = "OK" if worst <= tol else "WRONG"
    print(f"  {verdict:6s} {name}: rel={worst:.3e}")
    return verdict


def run_module_probe(name: str, build, tol: float) -> str:
    """Probe a stateful module: ``build(device) -> (callable, args)``.

    A fresh module is built per device rather than one moved across. Both mixers
    write their state in place, so reusing an instance would make the second run
    start from the first run's state; and the plain attributes holding the state
    tensors are not registered buffers, so ``.to(device)`` would silently leave
    them behind.
    """
    import vllm_neuron  # noqa: F401
    from vllm_neuron.compile.backend import compile as neuron_compile

    try:
        fn_cpu, args_cpu = build("cpu")
        expected = fn_cpu(*args_cpu)
    except Exception:
        print(f"  ERROR  {name}: CPU reference raised")
        traceback.print_exc()
        return "ERROR"

    try:
        fn_dev, args_dev = build(DEVICE)
        compiled = torch.compile(fn_dev, backend=neuron_compile, dynamic=False)
        got = compiled(*args_dev)
    except Exception as exc:
        lines = str(exc).strip().splitlines()
        print(f"  FAIL   {name}: {lines[0] if lines else type(exc).__name__}")
        for line in lines[1:6]:
            print(f"         {line}")
        return "FAIL"

    expected_list = expected if isinstance(expected, tuple) else (expected,)
    got_list = got if isinstance(got, tuple) else (got,)
    worst = 0.0
    for ref, out in zip(expected_list, got_list):
        out, ref = out.cpu().float(), ref.float()
        scale = ref.abs().max().clamp(min=1e-9)
        worst = max(worst, ((out - ref).abs().max() / scale).item())
    verdict = "OK" if worst <= tol else "WRONG"
    print(f"  {verdict:6s} {name}: rel={worst:.3e}")
    return verdict


def realistic_delta_inputs(seq: int):
    """Inputs with the same ranges the real checkpoint produces.

    Magnitudes matter here: ``g`` reaching about -4.6 per token is what pushes
    the in-chunk cumulative decay to -297 and ``exp`` to underflow, and that
    dynamic range is the whole reason the naive Neumann-series inverse fails.
    """
    from vllm_neuron.model.qwen3_5.deltanet import l2norm

    torch.manual_seed(0)
    query = l2norm(torch.randn(1, HEADS, seq, DK))
    key = l2norm(torch.randn(1, HEADS, seq, DK))
    value = torch.randn(1, HEADS, seq, DV) * 0.1
    beta = torch.rand(1, HEADS, seq)
    g = -torch.rand(1, HEADS, seq) * 4.6
    state = torch.zeros(1, HEADS, DK, DV)
    return query, key, value, g, beta, state


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)

    from vllm_neuron.model.qwen3_5 import deltanet as dn

    probes: list[tuple[str, object, tuple]] = []

    # --- building blocks, cheapest first --------------------------------
    torch.manual_seed(1)
    probes.append(
        (
            "bmm_transpose_rank3",
            lambda x: x @ x.transpose(-1, -2),
            (torch.randn(64, CHUNK, DK),),
        )
    )
    probes.append(
        (
            "bmm_transpose_rank5",
            lambda x: x @ x.transpose(-1, -2),
            (torch.randn(1, HEADS, 16, CHUNK, DK),),
        )
    )
    probes.append(
        (
            "transpose_then_bmm_rank3",
            lambda x, y: x.transpose(-1, -2) @ y,
            (torch.randn(64, CHUNK, DK), torch.randn(64, CHUNK, DV)),
        )
    )
    probes.append(
        (
            "strictly_lower_inverse_rank3",
            lambda a: dn._strictly_lower_inverse(a, CHUNK),
            (torch.randn(64, CHUNK, CHUNK).tril(-1) * 0.1,),
        )
    )
    probes.append(
        (
            "strictly_lower_inverse_rank5",
            lambda a: dn._strictly_lower_inverse(a, CHUNK),
            (torch.randn(1, HEADS, 16, CHUNK, CHUNK).tril(-1) * 0.1,),
        )
    )

    # --- the kernel itself ----------------------------------------------
    q, k, v, g, beta, state = realistic_delta_inputs(args.seq)
    probes.append(
        (
            "chunk_gated_delta_rule",
            dn.chunk_gated_delta_rule,
            (q, k, v, g, beta, state),
        )
    )
    # A single chunk, to separate "the kernel's shapes" from "16 of them chained".
    probes.append(
        (
            "chunk_gated_delta_rule_1chunk",
            dn.chunk_gated_delta_rule,
            tuple(
                x[..., :CHUNK, :] if x.dim() == 4 else x[..., :CHUNK] if x.dim() == 3
                else x
                for x in (q, k, v, g, beta, state)
            ),
        )
    )

    # --- whole modules, with the real checkpoint's weights ---------------
    # These add everything the kernel probe leaves out: the projections, the
    # unrolled causal conv, the padding mask derived from ``positions``, the
    # gated output norm, the row-parallel projection, and the in-place state
    # writes.
    module_probes: list[tuple[str, object]] = []
    if args.only is None or args.only.startswith(("delta", "attention_")):
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from transformers import AutoConfig

        from check_attention_vs_hf import build_attention, make_config
        from check_attention_vs_hf import load_layer_weights as load_submodule
        from check_deltanet_vs_hf import build_module, copy_into_ours
        from check_deltanet_vs_hf import load_layer_weights as load_deltanet

        model_path = "/mnt/nvme/models/Qwen3.5-2B"
        text_config = AutoConfig.from_pretrained(model_path).text_config
        seq = args.seq
        block_size = 288  # what the hybrid page alignment picks at runtime

        # positions exactly as the runner supplies them for a padded prefill.
        real = seq - 37
        positions_cpu = torch.cat(
            [torch.arange(real), torch.full((seq - real,), real - 1)]
        ).to(torch.int32)
        torch.manual_seed(7)
        hidden_cpu = torch.randn(seq, text_config.hidden_size) * 0.05

        delta_weights = load_deltanet(model_path, 0)
        attn_weights = load_submodule(model_path, 3, "self_attn")
        acfg = make_config(text_config)

        def build_delta(device):
            module, dcfg = build_module(text_config, 0, num_blocks=4)
            copy_into_ours(module, delta_weights)
            module = module.to(device)
            module.conv_state = torch.zeros(
                4, dcfg.linear_conv_kernel_dim - 1, module.conv_dim, device=device
            )
            module.recurrent_state = torch.zeros(
                4,
                module.num_v_heads,
                module.head_k_dim,
                module.head_v_dim,
                device=device,
            )
            meta = {
                "block_table_tensor": torch.full(
                    (1, 1), 1, dtype=torch.int32, device=device
                ),
                "slot_mapping": torch.zeros(seq, dtype=torch.int64, device=device),
                "max_query_len": seq,
                "decode_token_threshold": 1,
                "block_size": block_size,
            }
            args_ = (hidden_cpu.to(device), positions_cpu.to(device))

            def call(h, pos):
                return module.forward_prefill(h, pos, meta)

            return call, args_

        module_probes.append(("deltanet_forward_prefill", build_delta))

        # The compare-shaped pieces of forward_prefill, each on its own. The
        # NCC_IBCG901 "Too many strides" failure names a _compare node, and the
        # prefill has exactly three: the padding mask, the conv-state row gather,
        # and the padded-slot check in state_indices.
        def build_padding_mask(device):
            def call(pos):
                offsets = torch.arange(seq, device=pos.device, dtype=pos.dtype)
                return ((pos - pos[0]) == offsets).to(torch.float32)

            return call, (positions_cpu.to(device),)

        module_probes.append(("delta_padding_mask", build_padding_mask))

        def build_conv_state_gather(device):
            from vllm_neuron.model.qwen3_5.deltanet import Qwen3_5GatedDeltaNet

            def call(qkv, pos):
                offsets = torch.arange(seq, device=pos.device, dtype=pos.dtype)
                is_real = (pos - pos[0]) == offsets
                return Qwen3_5GatedDeltaNet._tail_rows(qkv, is_real, 3)

            return call, (
                torch.randn(seq, 1536).to(device),
                positions_cpu.to(device),
            )

        module_probes.append(("delta_conv_state_gather", build_conv_state_gather))

        def build_state_indices(device):
            module, _ = build_module(text_config, 0, num_blocks=4)
            module.conv_state = torch.zeros(4, 3, module.conv_dim, device=device)
            meta = {
                "block_table_tensor": torch.full(
                    (1, 1), 1, dtype=torch.int32, device=device
                ),
                "slot_mapping": torch.zeros(seq, dtype=torch.int64, device=device),
            }

            def call(dummy):
                return module.state_indices(meta, 1).to(torch.float32) + dummy

            return call, (torch.zeros(1, device=device),)

        module_probes.append(("delta_state_indices", build_state_indices))

        from vllm_neuron.model.qwen3_5.model import Qwen3_5RotaryEmbedding

        cos_cpu, sin_cpu = Qwen3_5RotaryEmbedding(acfg)(
            torch.arange(seq).unsqueeze(0).expand(3, -1), dtype=torch.float32
        )

        def build_attn(device):
            module = build_attention(acfg, 3)
            with torch.no_grad():
                module.q_proj_weight.copy_(attn_weights["q_proj.weight"].T)
                module.k_proj_weight.copy_(attn_weights["k_proj.weight"].T)
                module.v_proj_weight.copy_(attn_weights["v_proj.weight"].T)
                module.o_proj_weight.copy_(attn_weights["o_proj.weight"].T)
                module.q_norm.weight.copy_(attn_weights["q_norm.weight"])
                module.k_norm.weight.copy_(attn_weights["k_norm.weight"])
            module = module.to(device)
            nblocks = -(-seq // block_size) + 1
            module.k_cache = torch.zeros(
                nblocks,
                module.num_kv_heads_per_rank,
                block_size,
                module.head_dim,
                device=device,
            )
            module.v_cache = torch.zeros_like(module.k_cache)
            meta = {
                "block_table_tensor": torch.arange(
                    nblocks, dtype=torch.int32, device=device
                ).view(1, nblocks),
                "slot_mapping": torch.arange(seq, dtype=torch.int64, device=device),
                "max_query_len": seq,
                "decode_token_threshold": 1,
                "block_size": block_size,
            }

            def call(h, c, s_):
                return module.forward_prefill(h, (c, s_), meta)

            return call, (hidden_cpu.to(device), cos_cpu.to(device), sin_cpu.to(device))

        module_probes.append(("attention_forward_prefill", build_attn))

    selected = [p for p in probes if args.only is None or p[0] == args.only]
    selected_modules = [
        p for p in module_probes if args.only is None or p[0] == args.only
    ]
    if not selected and not selected_modules:
        raise SystemExit(f"no probe named {args.only!r}")

    total = len(selected) + len(selected_modules)
    print(f"probing {total} op(s) on {DEVICE}, seq={args.seq}\n")
    results = {}
    for name, fn, cpu_args in selected:
        results[name] = run_probe(name, fn, cpu_args, args.tol)
    for name, build in selected_modules:
        results[name] = run_module_probe(name, build, args.tol)

    bad = {n: v for n, v in results.items() if v not in ("OK",)}
    print("\n" + (f"PROBLEMS: {bad}" if bad else "ALL PROBES OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
