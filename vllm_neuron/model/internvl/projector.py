# SPDX-License-Identifier: Apache-2.0
"""Pixel shuffle + MLP projector mapping InternViT output into LLM space.

HF reference (``InternVLChatModel.extract_feature``):

    vit_embeds = vision_model(pixel_values)[:, 1:, :]   # drop CLS
    h = w = sqrt(num_patches)
    vit_embeds = vit_embeds.reshape(n, h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=downsample_ratio)
    vit_embeds = vit_embeds.reshape(n, -1, vit_embeds.shape[-1])
    vit_embeds = mlp1(vit_embeds)

``mlp1`` is ``nn.Sequential(LayerNorm(4096), Linear(4096, 3584), GELU,
Linear(3584, 3584))`` — checkpoint indices 0, 1, 3 carry parameters (2 is the
activation).

Sharding note: the projector is small (about 30M params) and runs once per tile,
so it is replicated on every vision TP rank rather than sharded. That keeps the
output identical across ranks, which is what the encoder-cache write path wants
— no reduction needed before the scatter.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm_neuron.utils.weight_loader import set_weight_loader

from .config import InternVLConfig
from .vision_encoder import _gelu
from .weight_loaders import replicated_transposed_loader


def pixel_shuffle(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Fold a spatial neighbourhood into the channel dim (``ps_version='v2'``).

    ``[n, w, h, c]`` -> ``[n, w*s, h*s, c/s**2]`` with ``s = scale_factor``.

    Transcribed from the HF implementation, including its final transpose, which
    is the only thing distinguishing v2 from v1. The two permutes do not cancel:
    the first moves the shuffled height axis ahead of width, the second restores
    width-major order after the channel regrouping.
    """
    n, w, h, c = x.size()
    # [n, w, h, c] -> [n, w, h*s, c/s]
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # -> [n, h*s, w, c/s]
    x = x.permute(0, 2, 1, 3).contiguous()
    # -> [n, h*s, w*s, c/s**2]
    x = x.view(
        n,
        int(h * scale_factor),
        int(w * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )
    # v2 restores width-major order.
    return x.permute(0, 2, 1, 3).contiguous()


class InternVLProjector(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear, over pixel-shuffled ViT output.

    forward: ``[num_tiles, num_patches, vit_hidden]``
          -> ``[num_tiles, num_patches / merge, llm_hidden]``
    """

    def __init__(
        self, config: InternVLConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.downsample_ratio = config.downsample_ratio
        self.grid_size = config.vision_config.grid_size

        vit_hidden = config.vision_config.hidden_size
        llm_hidden = config.text_config.hidden_size
        # After pixel shuffle each token carries merge_factor patches' channels.
        shuffled_dim = vit_hidden * config.vision_token_merge_factor

        self.norm = nn.LayerNorm(shuffled_dim, dtype=dtype)
        # Stored transposed ([in, out]) so the forward is a plain matmul, matching
        # the convention used across this plugin's model code.
        self.fc1_weight = nn.Parameter(
            torch.empty(shuffled_dim, llm_hidden, dtype=dtype)
        )
        self.fc1_bias = nn.Parameter(torch.empty(llm_hidden, dtype=dtype))
        self.fc2_weight = nn.Parameter(torch.empty(llm_hidden, llm_hidden, dtype=dtype))
        self.fc2_bias = nn.Parameter(torch.empty(llm_hidden, dtype=dtype))

        # Checkpoint holds HF Linear layout [out, in]; these params are [in, out].
        for w in (self.fc1_weight, self.fc2_weight):
            set_weight_loader(w, replicated_transposed_loader())

    def forward(self, vit_embeds: torch.Tensor) -> torch.Tensor:
        n, num_patches, _ = vit_embeds.shape
        g = self.grid_size
        if num_patches != g * g:
            raise ValueError(
                f"projector expects {g * g} patches per tile, got {num_patches}"
            )
        x = vit_embeds.reshape(n, g, g, -1)
        x = pixel_shuffle(x, self.downsample_ratio)
        x = x.reshape(n, -1, x.shape[-1])

        x = self.norm(x)
        x = torch.matmul(x, self.fc1_weight) + self.fc1_bias
        x = _gelu(x)
        return torch.matmul(x, self.fc2_weight) + self.fc2_bias

    def build_weight_mappings(self, prefix: str = "mlp1") -> dict[str, str]:
        """Parameter name -> checkpoint key.

        Sequential indices: 0 = LayerNorm, 1 = Linear, 2 = GELU (no params),
        3 = Linear.
        """
        return {
            "norm.weight": f"{prefix}.0.weight",
            "norm.bias": f"{prefix}.0.bias",
            "fc1_weight": f"{prefix}.1.weight",
            "fc1_bias": f"{prefix}.1.bias",
            "fc2_weight": f"{prefix}.3.weight",
            "fc2_bias": f"{prefix}.3.bias",
        }
