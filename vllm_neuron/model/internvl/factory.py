# SPDX-License-Identifier: Apache-2.0
"""Factory for InternVL3 model selection.

Registered under the architecture name ``InternVLChatModel``, which is what the
HF ``config.json`` declares. Registration happens in the **worker** process only
(``neuron_worker.py``), so vLLM's own InternVL class stays in place in the
frontend and its multimodal processor — dynamic tiling, prompt replacement,
``get_num_image_tokens`` — is reused unchanged. Only execution is replaced.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.interfaces import SupportsSpatialMerge
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


class InternVLChatModel(nn.Module, SupportsSpatialMerge):
    """Validates config and selects the InternVL3 implementation.

    The model runner passes ``text_neuron_config`` and ``vision_neuron_config``
    separately for multimodal models.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        vision_neuron_config: VisionNeuronConfig | None,
    ) -> nn.Module:
        cls._validate_config(hf_config)

        from .model import InternVLChatModel as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        """Raw vision tokens collapsed into one embed token by pixel shuffle.

        downsample_ratio 0.5 folds a 2x2 neighbourhood into the channel dim, so
        4 patches become 1 token: 1024 patches per tile -> 256 embed tokens.
        """
        ratio = getattr(hf_config, "downsample_ratio", 0.5)
        return int(round(1.0 / (ratio**2)))

    @classmethod
    def _validate_config(cls, hf_config: PretrainedConfig) -> None:
        """Reject configurations this implementation does not cover.

        InternVLConfig.from_configs re-checks the same things, but failing here
        gives a clear error before any weights are touched.
        """
        llm = getattr(hf_config, "llm_config", None) or getattr(
            hf_config, "text_config", None
        )
        if llm is None:
            raise ValueError("InternVL config is missing llm_config/text_config")
        llm_type = getattr(llm, "model_type", None) or (
            llm.get("model_type") if isinstance(llm, dict) else None
        )
        # <-- MODEL-SPECIFIC: only the Qwen2 backbone is implemented. InternVL3
        # also ships InternLM2 variants, whose attention layout differs.
        if llm_type not in ("qwen2",):
            raise NotImplementedError(
                f"InternVL3 LLM backbone {llm_type!r} is not implemented; "
                f"only 'qwen2' (e.g. InternVL3-8B) is supported."
            )
