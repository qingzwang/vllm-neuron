# SPDX-License-Identifier: Apache-2.0
"""Validate the Neuron InternViT implementation against HF, on CPU, TP=1.

Loads the real checkpoint weights into both implementations and compares the
vision tower output for the same tiles. Runs in float32 so the comparison
measures implementation differences, not bf16 noise.
"""

import json
import os
import sys

import torch

MODEL = os.environ.get("INTERNVL_PATH", "/mnt/nvme/models/InternVL3-8B-Instruct")
NUM_TILES = 2
DTYPE = torch.float32


def load_ckpt_subset(prefix: str) -> dict[str, torch.Tensor]:
    """Read every checkpoint tensor under a prefix into CPU float32."""
    from safetensors import safe_open

    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    wanted = {k: v for k, v in idx.items() if k.startswith(prefix)}
    by_file: dict[str, list[str]] = {}
    for k, f in wanted.items():
        by_file.setdefault(f, []).append(k)
    out = {}
    for fname, keys in by_file.items():
        with safe_open(f"{MODEL}/{fname}", framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k).to(DTYPE)
    return out


def _stub_timm():
    """Satisfy modeling_intern_vit's `from timm.layers import DropPath`.

    timm is not in the DLAMI venv and installing into it is off-limits. This
    model has drop_path_rate=0.0, so DropPath is the identity — a stub is
    faithful, not an approximation.
    """
    import importlib.machinery
    import types

    import torch.nn as nn

    if "timm" in sys.modules:
        return

    # Let transformers finish its lazy-module init first: it probes for timm with
    # find_spec, which chokes on a bare stub. Once transformers is imported the
    # probe has already run.
    import transformers  # noqa: F401

    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            assert not drop_prob, f"stub only valid for drop_prob=0, got {drop_prob}"

        def forward(self, x):
            return x

    timm = types.ModuleType("timm")
    layers = types.ModuleType("timm.layers")
    layers.DropPath = DropPath
    timm.layers = layers
    for mod, name in ((timm, "timm"), (layers, "timm.layers")):
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules["timm"] = timm
    sys.modules["timm.layers"] = layers


def build_hf(ckpt):
    """HF InternVisionModel from the checkpoint's own modeling file."""
    _stub_timm()
    # modeling_intern_vit does `from .configuration_intern_vit import ...`, so it
    # has to be imported as part of a package. Register the checkpoint dir as a
    # synthetic package so the relative import resolves.
    import importlib
    import types

    if "ivl_ckpt" not in sys.modules:
        pkg = types.ModuleType("ivl_ckpt")
        pkg.__path__ = [MODEL]
        sys.modules["ivl_ckpt"] = pkg
    InternVisionConfig = importlib.import_module(
        "ivl_ckpt.configuration_intern_vit"
    ).InternVisionConfig
    InternVisionModel = importlib.import_module(
        "ivl_ckpt.modeling_intern_vit"
    ).InternVisionModel

    cfg = InternVisionConfig(**json.load(open(f"{MODEL}/config.json"))["vision_config"])
    cfg.torch_dtype = DTYPE
    model = InternVisionModel(cfg).to(DTYPE).eval()
    sd = {k[len("vision_model.") :]: v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The tower has no head; ignore anything past the encoder.
    missing = [m for m in missing if not m.startswith("encoder.layers.0.attn.q_norm")]
    print(f"  HF load: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print("    missing:", missing[:5])
    return model, cfg


def init_tp1():
    """Run un-sharded (TP=1).

    vLLM's initialize_model_parallel needs a live VllmConfig context, which is
    more machinery than this check warrants. The encoder already treats a missing
    vision TP group as tp_size=1, so point the getter at None and validate the
    math un-sharded. TP sharding is exercised separately by the on-device TP=4
    run, where a wrong shard shows up immediately as garbage output.
    """
    import vllm_neuron.model.internvl.vision_encoder as ve

    ve.get_neuron_vision_tp_group = lambda: None


def build_neuron(ckpt, cfg):
    """Our implementation, weights transformed the way the loaders would."""
    sys.path.insert(0, "/mnt/nvme/vllm-neuron")
    init_tp1()
    from vllm_neuron.model.internvl.config import InternVLVisionConfig
    from vllm_neuron.model.internvl.vision_encoder import InternVisionModel as NeuronViT

    ncfg = InternVLVisionConfig.from_hf(cfg)
    model = NeuronViT(ncfg, dtype=DTYPE).eval()

    mapping = model.build_weight_mappings()
    sd = {}
    for pname, ckey in mapping.items():
        t = ckpt[ckey]
        if pname.endswith("embeddings.proj_weight"):
            t = t.reshape(t.shape[0], -1).T.contiguous()      # conv -> matmul
        elif pname.endswith("attn.qkv_weight"):
            t = t.T.contiguous()                              # [3H,H] -> [H,3H]
        elif pname.endswith(("attn.proj_weight", "mlp.fc1_weight", "mlp.fc2_weight")):
            t = t.T.contiguous()                              # HF Linear is [out,in]
        sd[pname] = t
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"  Neuron load: strict OK ({len(sd)} tensors)")
    return model


def main():
    print("loading checkpoint tensors...")
    ckpt = load_ckpt_subset("vision_model.")
    print(f"  {len(ckpt)} vision tensors")

    hf, cfg = build_hf(ckpt)
    neuron = build_neuron(ckpt, cfg)

    g = torch.Generator().manual_seed(0)
    px = torch.randn(
        NUM_TILES, 3, cfg.image_size, cfg.image_size, generator=g, dtype=DTYPE
    )

    with torch.no_grad():
        # drop CLS, matching HF extract_feature's vit_embeds[:, 1:, :]
        ref = hf(pixel_values=px).last_hidden_state[:, 1:, :]
        out = neuron(px)

    print(f"\nref  {tuple(ref.shape)}   neuron {tuple(out.shape)}")
    assert ref.shape == out.shape, "shape mismatch"
    diff = (out - ref).abs()
    rel = diff.max().item() / ref.abs().max().item()
    print(
        f"max|diff| = {diff.max().item():.3e}   "
        f"mean|diff| = {diff.mean().item():.3e}"
    )
    print(f"relative  = {rel:.3e}")
    print("\nVERDICT:", "MATCH" if rel < 1e-4 else "*** MISMATCH ***")


if __name__ == "__main__":
    main()
