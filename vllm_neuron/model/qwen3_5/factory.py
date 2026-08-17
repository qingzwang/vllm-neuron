# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.5, registered under the checkpoint's HF architecture name.

The checkpoint declares ``Qwen3_5ForConditionalGeneration``, so vLLM's frontend
supplies the config and the multimodal processor for free and only *execution*
is replaced here. This port implements the text decoder; the ViT tower is not
built yet, so the validation below rejects a vision request outright rather than
dropping the image and answering from the text alone.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Validates the config, then builds the text-only implementation."""

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
        cls._validate_config(vision_neuron_config)

        from .model import Qwen3_5ForCausalLM

        return Qwen3_5ForCausalLM.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

    @classmethod
    def _validate_config(cls, vision_neuron_config: VisionNeuronConfig | None) -> None:
        # The runner only builds a VisionNeuronConfig when the engine was set up
        # to serve images or video. Failing here beats accepting the request and
        # answering from the text alone, which reads as a bad model rather than a
        # missing feature.
        if vision_neuron_config is not None:
            raise NotImplementedError(
                "Qwen3.5 on Neuron is text-only so far: the ViT tower is not "
                "implemented. Start the engine without vision "
                "(no limit_mm_per_prompt / vision bucket configuration)."
            )
