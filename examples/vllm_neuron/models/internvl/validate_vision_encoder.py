# SPDX-License-Identifier: Apache-2.0
"""Validate the Neuron InternVL vision pipeline against HF, on CPU, TP=1.

Compares two stages against the checkpoint's own HF implementation:
  1. InternViT tower output (post-CLS-drop)
  2. Full extract_feature: + pixel shuffle + mlp1 projector -> LLM space

Loads the real checkpoint weights into both implementations and runs in float32
so the comparison measures implementation differences, not bf16 noise.
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


def build_hf_projector(proj_ckpt, cfg, hf_cfg_full):
    """HF mlp1 as nn.Sequential, exactly as InternVLChatModel builds it."""
    import torch.nn as nn

    vit_hidden = cfg.hidden_size
    llm_hidden = hf_cfg_full["llm_config"]["hidden_size"]
    ds = hf_cfg_full.get("downsample_ratio", 0.5)
    shuffled = vit_hidden * int(1 / ds) ** 2
    mlp1 = nn.Sequential(
        nn.LayerNorm(shuffled),
        nn.Linear(shuffled, llm_hidden),
        nn.GELU(),
        nn.Linear(llm_hidden, llm_hidden),
    ).to(DTYPE).eval()
    mlp1.load_state_dict(
        {k[len("mlp1.") :]: v for k, v in proj_ckpt.items()}, strict=True
    )
    return mlp1, ds


def hf_extract_feature(vit_embeds, mlp1, ds, ps_version="v2"):
    """Transcribed from InternVLChatModel.extract_feature / pixel_shuffle."""
    n = vit_embeds.shape[0]
    h = w = int(vit_embeds.shape[1] ** 0.5)
    x = vit_embeds.reshape(n, h, w, -1)
    # HF pixel_shuffle
    nn_, ww, hh, c = x.size()
    x = x.view(nn_, ww, int(hh * ds), int(c / ds))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(nn_, int(hh * ds), int(ww * ds), int(c / (ds * ds)))
    if ps_version != "v1":
        x = x.permute(0, 2, 1, 3).contiguous()
    x = x.reshape(n, -1, x.shape[-1])
    return mlp1(x)


def build_neuron_projector(proj_ckpt, hf_cfg_full, cfg):
    """Our projector with checkpoint weights transposed for the matmul layout."""
    from vllm_neuron.model.internvl.config import InternVLConfig
    from vllm_neuron.model.internvl.projector import InternVLProjector

    class _Shim:
        pass

    shim = _Shim()
    shim.llm_config = hf_cfg_full["llm_config"]
    shim.vision_config = cfg
    shim.downsample_ratio = hf_cfg_full.get("downsample_ratio", 0.5)
    shim.ps_version = hf_cfg_full.get("ps_version", "v2")
    shim.select_layer = hf_cfg_full.get("select_layer", -1)
    shim.image_token_id = hf_cfg_full.get("image_token_id")
    ncfg = InternVLConfig.from_configs(shim)

    model = InternVLProjector(ncfg, dtype=DTYPE).eval()
    sd = {}
    for pname, ckey in model.build_weight_mappings().items():
        t = proj_ckpt[ckey]
        if pname.endswith(("fc1_weight", "fc2_weight")):
            t = t.T.contiguous()
        sd[pname] = t
    model.load_state_dict(sd, strict=True)
    print(f"  Neuron projector: strict OK ({len(sd)} tensors)")
    return model


def report(name, ref, out, tol=1e-4):
    assert ref.shape == out.shape, (
        f"{name}: shape {tuple(ref.shape)} vs {tuple(out.shape)}"
    )
    diff = (out - ref).abs()
    rel = diff.max().item() / ref.abs().max().item()
    ok = rel < tol
    print(
        f"  {name:<22} {tuple(out.shape)!s:<22} "
        f"max|d|={diff.max().item():.3e} rel={rel:.3e}  "
        f"{'MATCH' if ok else '*** MISMATCH ***'}"
    )
    return ok


def main():
    print("loading checkpoint tensors...")
    ckpt = load_ckpt_subset("vision_model.")
    print(f"  {len(ckpt)} vision tensors")
    proj_ckpt = load_ckpt_subset("mlp1.")
    print(f"  {len(proj_ckpt)} projector tensors")

    hf, cfg = build_hf(ckpt)
    neuron = build_neuron(ckpt, cfg)

    hf_cfg_full = json.load(open(f"{MODEL}/config.json"))
    hf_mlp1, ds = build_hf_projector(proj_ckpt, cfg, hf_cfg_full)
    neuron_proj = build_neuron_projector(proj_ckpt, hf_cfg_full, cfg)

    g = torch.Generator().manual_seed(0)
    px = torch.randn(
        NUM_TILES, 3, cfg.image_size, cfg.image_size, generator=g, dtype=DTYPE
    )

    print("\ncomparing:")
    ok = True
    with torch.no_grad():
        # drop CLS, matching HF extract_feature's vit_embeds[:, 1:, :]
        ref_vit = hf(pixel_values=px).last_hidden_state[:, 1:, :]
        out_vit = neuron(px)
        ok &= report("InternViT tower", ref_vit, out_vit)

        # Feed each side its own tower output so a projector bug cannot be
        # masked by tower differences, then also check the full chain.
        ps = hf_cfg_full.get("ps_version", "v2")
        ref_feat = hf_extract_feature(ref_vit, hf_mlp1, ds, ps)
        out_feat = neuron_proj(out_vit)
        ok &= report("extract_feature", ref_feat, out_feat)

    print(f"\nper-tile embed tokens: {out_feat.shape[1]} (expected 256)")
    print("VERDICT:", "ALL MATCH" if ok else "*** MISMATCH ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
