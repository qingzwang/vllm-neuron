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
        from .model import Qwen3_5ForCausalLM

        return Qwen3_5ForCausalLM.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

    # ── vLLM's hybrid-model contract ─────────────────────────────────────
    # vLLM asks the *registered* class — which is this one, not its own
    # implementation — for the recurrent state geometry, and uses it to size the
    # state pages the block planner allocates. Delegating to
    # ``Mamba*Calculator`` is what keeps that sizing identical to what
    # ``Qwen3_5TextConfig.state_shapes`` reports to the model runner; a second
    # copy of the arithmetic would alias memory rather than raise.

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> tuple[tuple[int, ...], ...]:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        hf_text_config = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf_text_config.linear_num_key_heads,
            hf_text_config.linear_num_value_heads,
            hf_text_config.linear_key_head_dim,
            hf_text_config.linear_value_head_dim,
            hf_text_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )

        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    # No vision guard here on purpose. There is no signal at construction time
    # that distinguishes "the user configured vision" from "the checkpoint merely
    # has a vision_config": the runner auto-derives a VisionNeuronConfig with
    # num_vision_tokens_buckets for every such checkpoint, so both an
    # `is not None` check and a bucket check reject a plain text-only launch.
    # The text-only model simply does not implement ``embed_multimodal`` or
    # ``SupportsVisionWarmup``, so a request carrying an image fails loudly
    # rather than being answered from the text alone.
