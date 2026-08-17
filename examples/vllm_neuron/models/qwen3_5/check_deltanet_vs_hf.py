#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Numerically validate the Qwen3.5 DeltaNet mixer against HuggingFace, on CPU.

Runs three checks, cheapest first, so a failure points at one thing:

1. **Kernels** — our ``chunk_gated_delta_rule`` and
   ``recurrent_gated_delta_rule_step`` against ``transformers``'
   ``torch_chunk_gated_delta_rule`` / ``torch_recurrent_gated_delta_rule`` on
   random inputs. This is what proves the loop-free ``(I - A)^-1`` is the same
   matrix HF builds by forward substitution.
2. **Prefill** — our whole module against HF's ``Qwen3_5GatedDeltaNet.forward``,
   using layer 0's real weights from the checkpoint. Run twice: an exact-length
   prompt, and a short prompt padded out to a bucket, which is the case the
   plugin actually feeds it.
3. **Decode** — our decode path replayed one token at a time from a cold state,
   against our (now HF-verified) prefill over the same tokens. This is the only
   way to exercise the conv-window and recurrent-state carry, which HF's module
   reaches through a ``Cache`` object the Neuron runner does not have.

Everything runs in float32 on CPU: the point is to isolate implementation error
from bf16 rounding. Usage:

    python check_deltanet_vs_hf.py [--model /mnt/nvme/models/Qwen3.5-2B]
"""

from __future__ import annotations

import argparse
import sys
from unittest import mock

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/nvme/models/Qwen3.5-2B")
    parser.add_argument("--layer", type=int, default=0, help="DeltaNet layer to load")
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--tol", type=float, default=2e-4)
    return parser.parse_args()


class _SingleRankGroup:
    """Stand-in for vLLM's TP group. TP=1, so no collective is ever called."""

    world_size = 1
    rank_in_group = 0


def build_module(text_config, layer_idx: int, num_blocks: int = 4, tp_group=None):
    """Instantiate our mixer in float32 with vLLM's state tensors faked in."""
    from vllm_neuron.model.qwen3_5 import deltanet as dn
    from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig.from_hf(text_config)
    # Float32 throughout, so any mismatch is a bug and not bf16 noise.
    cfg.torch_dtype = torch.float32

    group = tp_group if tp_group is not None else _SingleRankGroup()
    with mock.patch.object(dn, "get_tp_group", lambda: group):
        module = dn.Qwen3_5GatedDeltaNet(cfg, layer_idx)

    module.conv_state = torch.zeros(
        num_blocks, cfg.linear_conv_kernel_dim - 1, module.conv_dim
    )
    module.recurrent_state = torch.zeros(
        num_blocks, module.num_v_heads, module.head_k_dim, module.head_v_dim
    )
    return module, cfg


def load_layer_weights(model_path: str, layer_idx: int) -> dict[str, torch.Tensor]:
    """Read one DeltaNet layer's tensors out of the checkpoint, in HF layout."""
    from safetensors import safe_open
    import glob
    import json
    import os

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as handle:
        weight_map = json.load(handle)["weight_map"]

    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
    wanted = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not wanted:
        raise SystemExit(
            f"no tensors under {prefix!r}; is layer {layer_idx} a DeltaNet layer? "
            f"(candidates: {sorted({k.rsplit('.', 1)[0] for k in weight_map})[:4]})"
        )

    out = {}
    for shard in sorted(set(wanted.values())):
        path = os.path.join(model_path, shard)
        if not os.path.exists(path):  # single-shard checkpoints
            (path,) = glob.glob(os.path.join(model_path, "*.safetensors"))
        with safe_open(path, framework="pt") as handle:
            for key in wanted:
                if wanted[key] == shard:
                    out[key[len(prefix) :]] = handle.get_tensor(key).float()
    return out


def copy_into_ours(module, hf_weights: dict[str, torch.Tensor]) -> None:
    """HF layout -> our transposed, TP=1 parameters."""
    with torch.no_grad():
        module.in_proj_qkv_weight.copy_(hf_weights["in_proj_qkv.weight"].T)
        module.in_proj_z_weight.copy_(hf_weights["in_proj_z.weight"].T)
        module.in_proj_ba_weight.copy_(
            torch.cat(
                [hf_weights["in_proj_b.weight"], hf_weights["in_proj_a.weight"]], dim=0
            ).T
        )
        module.conv1d_weight.copy_(hf_weights["conv1d.weight"].squeeze(1))
        module.dt_bias.copy_(hf_weights["dt_bias"])
        module.A_log.copy_(hf_weights["A_log"])
        module.norm_weight.copy_(hf_weights["norm.weight"])
        module.out_proj_weight.copy_(hf_weights["out_proj.weight"].T)


def build_hf_module(text_config, layer_idx: int, hf_weights: dict[str, torch.Tensor]):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    module = Qwen3_5GatedDeltaNet(text_config, layer_idx).float()
    missing, unexpected = module.load_state_dict(hf_weights, strict=False)
    # HF's Conv1d weight is [conv_dim, 1, kernel]; ours came from the checkpoint
    # in that shape, so nothing should be missing except a bias that does not
    # exist in this config.
    unexpected = [k for k in unexpected]
    if unexpected:
        raise SystemExit(f"unexpected keys for HF module: {unexpected}")
    if missing:
        print(f"  (HF module keys not in checkpoint, left at init: {missing})")
    return module.eval()


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


def fake_metadata(num_tokens: int, num_reqs: int, is_decode: bool, block_id: int = 1):
    """The subset of the runner's attention metadata the mixer reads."""
    return {
        "block_table_tensor": torch.full((num_reqs, 1), block_id, dtype=torch.int32),
        "slot_mapping": torch.zeros(num_tokens, dtype=torch.int64),
        "max_query_len": 1 if is_decode else num_tokens,
        "decode_token_threshold": 1,
    }


def check_kernels(args) -> bool:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )
    from vllm_neuron.model.qwen3_5.deltanet import (
        chunk_gated_delta_rule,
        l2norm,
        recurrent_gated_delta_rule_step,
    )

    print("1. kernels vs transformers reference")
    torch.manual_seed(0)
    b, h, t, d = 1, 4, args.seq_len, 128
    # HF's entry points take [B, T, H, D] and transpose internally.
    q = torch.randn(b, t, h, d)
    k = torch.randn(b, t, h, d)
    v = torch.randn(b, t, h, d)
    beta = torch.rand(b, t, h)
    # Same range the real model produces: -exp(A_log) * softplus(...) <= 0.
    g = -torch.rand(b, t, h) * 0.5
    state0 = torch.randn(b, h, d, d) * 0.1

    hf_out, hf_state = torch_chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=state0.clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ours_out, ours_state = chunk_gated_delta_rule(
        l2norm(q.transpose(1, 2)),
        l2norm(k.transpose(1, 2)),
        v.transpose(1, 2),
        g.transpose(1, 2),
        beta.transpose(1, 2),
        state0.clone(),
    )
    ok = report("chunk output", ours_out.transpose(1, 2), hf_out, args.tol)
    ok &= report("chunk final state", ours_state, hf_state, args.tol)

    hf_step_out, hf_step_state = torch_recurrent_gated_delta_rule(
        q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1],
        initial_state=state0.clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ours_step_out, ours_step_state = recurrent_gated_delta_rule_step(
        l2norm(q[:, 0]), l2norm(k[:, 0]), v[:, 0], g[:, 0], beta[:, 0], state0.clone()
    )
    ok &= report("step output", ours_step_out, hf_step_out[:, 0], args.tol)
    ok &= report("step state", ours_step_state, hf_step_state, args.tol)
    return ok


def check_prefill(args, hf_module, our_module, hidden_size: int) -> bool:
    print("2. module prefill vs HF module")
    torch.manual_seed(1)
    ok = True

    # (a) Exact length: no padding, so HF sees the identical token sequence.
    hidden = torch.randn(args.seq_len, hidden_size) * 0.05
    with torch.no_grad():
        hf_out = hf_module(hidden.unsqueeze(0)).squeeze(0)
    our_module.conv_state.zero_()
    our_module.recurrent_state.zero_()
    with torch.no_grad():
        ours = our_module.forward_prefill(
            hidden,
            torch.arange(args.seq_len),
            fake_metadata(args.seq_len, 1, is_decode=False),
        )
    ok &= report("prefill, exact length", ours[: args.seq_len], hf_out, args.tol)

    # (b) Padded to the bucket, which is what the runner actually does: real
    # tokens then token id 0 with `positions` frozen at the last real value.
    real = args.seq_len - 37
    padded_hidden = hidden.clone()
    padded_hidden[real:] = torch.randn(args.seq_len - real, hidden_size) * 0.05
    positions = torch.cat(
        [torch.arange(real), torch.full((args.seq_len - real,), real - 1)]
    )
    with torch.no_grad():
        hf_short = hf_module(hidden[:real].unsqueeze(0)).squeeze(0)
    our_module.conv_state.zero_()
    our_module.recurrent_state.zero_()
    with torch.no_grad():
        ours_padded = our_module.forward_prefill(
            padded_hidden, positions, fake_metadata(args.seq_len, 1, is_decode=False)
        )
    ok &= report("prefill, padded to bucket", ours_padded[:real], hf_short, args.tol)
    return ok


def check_decode(args, our_module, hidden_size: int) -> bool:
    """Replay decode token-by-token and compare against prefill of the same run.

    Prefill is verified against HF above, so agreement here means the decode
    step, the conv-window carry and the state read/write all match HF too.
    """
    print("3. decode replay vs prefill")
    torch.manual_seed(2)
    steps = 96
    hidden = torch.randn(steps, hidden_size) * 0.05

    our_module.conv_state.zero_()
    our_module.recurrent_state.zero_()
    decode_meta = fake_metadata(1, 1, is_decode=True)
    outs = []
    with torch.no_grad():
        for step in range(steps):
            outs.append(our_module.forward_decode(hidden[step : step + 1], decode_meta))
    decoded = torch.cat(outs, dim=0)

    # Prefill the same tokens, padded up to a chunk multiple the way the runner
    # would, so the comparison covers the padded path as well.
    bucket = -(-steps // 64) * 64
    padded = torch.cat([hidden, torch.randn(bucket - steps, hidden_size) * 0.05])
    positions = torch.cat(
        [torch.arange(steps), torch.full((bucket - steps,), steps - 1)]
    )
    conv_after_decode = our_module.conv_state.clone()
    rec_after_decode = our_module.recurrent_state.clone()
    our_module.conv_state.zero_()
    our_module.recurrent_state.zero_()
    with torch.no_grad():
        prefilled = our_module.forward_prefill(
            padded, positions, fake_metadata(bucket, 1, is_decode=False)
        )

    ok = report("decode outputs", decoded, prefilled[:steps], args.tol)
    ok &= report("conv state carry", conv_after_decode, our_module.conv_state, args.tol)
    ok &= report(
        "recurrent state carry", rec_after_decode, our_module.recurrent_state, args.tol
    )
    return ok


class _TensorSlice:
    """Minimal stand-in for safetensors' ``PySafeSlice``, for the loaders."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def get_shape(self) -> tuple[int, ...]:
        return tuple(self._tensor.shape)

    def __getitem__(self, key):
        return self._tensor[key]

    @property
    def T(self):  # noqa: N802 - mirrors torch.Tensor
        return self._tensor.T


def check_tp_sharding(args, text_config, hf_weights, tp_size: int = 4) -> bool:
    """TP=``tp_size`` must reproduce TP=1, using the real weight loaders.

    Each rank's parameters are produced by calling the module's own
    ``SafetensorsWeightLoader``s, so this covers the part most likely to be
    silently wrong: ``in_proj_qkv`` and ``conv1d`` hold ``[q | k | v]``
    concatenated on their output dimension, and slicing that contiguously would
    hand rank 1 a mix of q and k channels rather than its own heads.

    ``out_proj`` is row-parallel, so the ranks' outputs are summed the way the
    TP all-reduce would.
    """
    from vllm_neuron.utils.weight_loader import get_weight_loader

    print(f"4. TP={tp_size} sharding vs TP=1")
    hidden_size = text_config.hidden_size
    torch.manual_seed(3)
    seq_len = 128
    hidden = torch.randn(seq_len, hidden_size) * 0.05
    positions = torch.arange(seq_len)

    reference, _ = build_module(text_config, args.layer)
    copy_into_ours(reference, hf_weights)
    with torch.no_grad():
        expected = reference.forward_prefill(
            hidden, positions, fake_metadata(seq_len, 1, is_decode=False)
        )

    # Which checkpoint tensor(s) each parameter is loaded from, in the order
    # ``load_weights`` will pass them.
    sources = {
        "in_proj_qkv_weight": ["in_proj_qkv.weight"],
        "in_proj_z_weight": ["in_proj_z.weight"],
        "in_proj_ba_weight": ["in_proj_b.weight", "in_proj_a.weight"],
        "conv1d_weight": ["conv1d.weight"],
        "dt_bias": ["dt_bias"],
        "A_log": ["A_log"],
        "norm_weight": ["norm.weight"],
        "out_proj_weight": ["out_proj.weight"],
    }

    total = torch.zeros_like(expected)
    shard_states = []
    for rank in range(tp_size):
        group = type("_G", (), {"world_size": tp_size, "rank_in_group": rank})()
        shard, _ = build_module(text_config, args.layer, tp_group=group)
        # world_size > 1 would trigger collectives; the shard math is what is
        # under test, so run each rank standalone and combine here.
        shard.world_size = 1
        with torch.no_grad():
            for name, keys in sources.items():
                param = getattr(shard, name)
                loader = get_weight_loader(param)
                slices = [_TensorSlice(hf_weights[k]) for k in keys]
                param.copy_(loader.load(slices, rank))
            out = shard.forward_prefill(
                hidden, positions, fake_metadata(seq_len, 1, is_decode=False)
            )
        total = total + out
        shard_states.append(shard.recurrent_state[1].clone())

    ok = report("summed rank outputs", total, expected, args.tol)
    # Heads are partitioned, so concatenating the ranks' states must rebuild the
    # unsharded one; a wrong shard order would show up here even if the summed
    # output happened to look right.
    ok &= report(
        "concatenated rank states",
        torch.cat(shard_states, dim=0),
        reference.recurrent_state[1],
        args.tol,
    )
    return ok


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    text_config = hf_config.text_config
    layer_type = text_config.layer_types[args.layer]
    print(f"checkpoint: {args.model}")
    print(f"layer {args.layer} is {layer_type!r}\n")
    if layer_type != "linear_attention":
        raise SystemExit(f"layer {args.layer} is {layer_type}, not a DeltaNet layer")

    ok = check_kernels(args)

    hf_weights = load_layer_weights(args.model, args.layer)
    our_module, cfg = build_module(text_config, args.layer)
    copy_into_ours(our_module, hf_weights)
    hf_module = build_hf_module(text_config, args.layer, hf_weights)

    ok &= check_prefill(args, hf_module, our_module, cfg.hidden_size)
    ok &= check_decode(args, our_module, cfg.hidden_size)
    ok &= check_tp_sharding(args, text_config, hf_weights)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
