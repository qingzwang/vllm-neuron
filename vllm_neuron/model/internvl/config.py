# SPDX-License-Identifier: Apache-2.0
"""InternVL3 model configs (text = Qwen2, vision = InternViT).

The HF config nests two sub-configs under a top-level ``internvl_chat`` config:
``llm_config`` (a Qwen2 config) and ``vision_config`` (an InternViT config), plus
the multimodal glue parameters (``downsample_ratio``, ``ps_version``,
``select_layer``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


@dataclass
class InternVLTextConfig:
    """Qwen2 decoder parameters.

    Differs from Qwen3 (used by qwen3_vl) in two ways that matter for the
    attention implementation: Qwen2 has **bias on q/k/v** and has **no per-head
    QK normalization**.
    """

    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    num_hidden_layers: int = 28
    intermediate_size: int = 18944
    head_dim: int = 128
    vocab_size: int = 151674
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    # Qwen2 always has qkv bias; kept explicit so the weight loaders and the
    # NF.qkv_proj call stay readable.
    attention_qkv_bias: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_hf(
        cls, llm_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ) -> "InternVLTextConfig":
        d = (
            llm_config.to_dict()
            if isinstance(llm_config, PretrainedConfig)
            else dict(llm_config)
        )
        head_dim = d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"]
        return cls(
            hidden_size=d["hidden_size"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            num_hidden_layers=d["num_hidden_layers"],
            intermediate_size=d["intermediate_size"],
            head_dim=head_dim,
            vocab_size=d["vocab_size"],
            max_position_embeddings=d.get("max_position_embeddings", 32768),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=d.get("rope_theta", 1000000.0),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            neuron_config=neuron_config,
        )


@dataclass
class InternVLVisionConfig:
    """InternViT parameters.

    Note ``head_dim`` is 64 here (1024 / 16), not the 128 the text side uses.
    """

    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    image_size: int = 448
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    qk_normalization: bool = False
    norm_type: str = "layer_norm"
    torch_dtype: torch.dtype = torch.bfloat16

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def grid_size(self) -> int:
        """Patches per side for one tile (448 / 14 = 32)."""
        return self.image_size // self.patch_size

    @property
    def num_patches_per_tile(self) -> int:
        """Vision tokens per tile before pixel shuffle (32 * 32 = 1024)."""
        return self.grid_size**2

    @classmethod
    def from_hf(cls, vision_config: PretrainedConfig) -> "InternVLVisionConfig":
        d = (
            vision_config.to_dict()
            if isinstance(vision_config, PretrainedConfig)
            else dict(vision_config)
        )
        cfg = cls(
            hidden_size=d["hidden_size"],
            num_attention_heads=d["num_attention_heads"],
            num_hidden_layers=d["num_hidden_layers"],
            intermediate_size=d["intermediate_size"],
            image_size=d.get("image_size", 448),
            patch_size=d.get("patch_size", 14),
            num_channels=d.get("num_channels", 3),
            layer_norm_eps=d.get("layer_norm_eps", 1e-6),
            qkv_bias=d.get("qkv_bias", True),
            qk_normalization=d.get("qk_normalization", False),
            norm_type=d.get("norm_type", "layer_norm"),
        )
        # The HF implementation bicubic-interpolates the position embedding from
        # its stored grid to the actual patch grid. Both are image_size/patch_size
        # for every tile InternVL produces, so the interpolation is the identity
        # and this implementation skips it. Guard the assumption rather than
        # silently producing wrong embeddings if a config ever diverges.
        if cfg.norm_type != "layer_norm":
            raise NotImplementedError(
                f"InternViT norm_type={cfg.norm_type!r} is not implemented; "
                f"only 'layer_norm' is supported."
            )
        if cfg.qk_normalization:
            raise NotImplementedError(
                "InternViT qk_normalization=True is not implemented "
                "(InternVL3-8B has it disabled)."
            )
        return cfg


@dataclass
class InternVLConfig:
    """Top-level InternVL3 config: text + vision + multimodal glue."""

    text_config: InternVLTextConfig = field(default_factory=InternVLTextConfig)
    vision_config: InternVLVisionConfig = field(default_factory=InternVLVisionConfig)

    # Multimodal glue
    downsample_ratio: float = 0.5
    ps_version: str = "v2"
    select_layer: int = -1
    image_token_id: int | None = None

    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None
    vision_neuron_config: VisionNeuronConfig | None = None

    @property
    def vision_token_merge_factor(self) -> int:
        """Raw vision tokens collapsed into one embed token by pixel shuffle.

        downsample_ratio 0.5 shuffles a 2x2 spatial neighbourhood into the
        channel dim, so 4 patches become 1 token.
        """
        return int(round(1.0 / (self.downsample_ratio**2)))

    @property
    def embed_tokens_per_tile(self) -> int:
        """Vision tokens the LLM sees per 448x448 tile (1024 / 4 = 256)."""
        return self.vision_config.num_patches_per_tile // self.vision_token_merge_factor

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> "InternVLConfig":
        llm_config = getattr(hf_config, "llm_config", None) or getattr(
            hf_config, "text_config"
        )
        vision_config = hf_config.vision_config

        cfg = cls(
            text_config=InternVLTextConfig.from_hf(llm_config, text_neuron_config),
            vision_config=InternVLVisionConfig.from_hf(vision_config),
            downsample_ratio=getattr(hf_config, "downsample_ratio", 0.5),
            ps_version=getattr(hf_config, "ps_version", "v2"),
            select_layer=getattr(hf_config, "select_layer", -1),
            image_token_id=getattr(hf_config, "image_token_id", None),
            neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        if cfg.ps_version != "v2":
            raise NotImplementedError(
                f"ps_version={cfg.ps_version!r} is not implemented; the v1 layout "
                f"omits the final transpose in pixel_shuffle."
            )
        if cfg.select_layer != -1:
            raise NotImplementedError(
                f"select_layer={cfg.select_layer} is not implemented; this "
                f"implementation always takes the last encoder layer."
            )
        return cfg
