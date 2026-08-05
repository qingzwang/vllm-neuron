# SPDX-License-Identifier: Apache-2.0
"""Factory for DeepSeek-V4 model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class DeepseekV4ForCausalLM(nn.Module):
    """Factory that validates config and selects the DeepSeek-V4 implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls
    to it.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        # Only a bf16-compute implementation exists today: FP8/FP4 checkpoint
        # weights are dequantized at load time (see weight_loaders.py).
        from .model_bf16 import DeepseekV4ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject configurations this implementation cannot serve correctly."""
        from .config import DeepseekV4Config

        config = DeepseekV4Config.from_configs(hf_config, neuron_config)

        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for DeepSeek-V4. "
                "Checkpoint FP8/FP4 weights are dequantized to bf16 at load time; "
                "pass quantization='bf16' or leave it unset."
            )

        if config.n_shared_experts != 1:
            raise ValueError(
                "DeepSeek-V4 support assumes exactly one shared expert, got "
                f"n_shared_experts={config.n_shared_experts}."
            )

        if config.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"scoring_func={config.scoring_func!r} is not supported for "
                "DeepSeek-V4; only 'sqrtsoftplus' is implemented."
            )

        rope_type = (config.rope_scaling or {}).get("type")
        if rope_type not in (None, "yarn"):
            raise ValueError(
                f"rope_scaling type={rope_type!r} is not supported for "
                "DeepSeek-V4; only YaRN is implemented."
            )

        # Ratio 4 selects slots via the Indexer (CSA); any other non-zero ratio
        # attends over every compressed slot (HCA). Both need the ratio to
        # divide the compressor's pooling window evenly.
        invalid_ratios = sorted(r for r in config.compress_ratios if r < 0)
        if invalid_ratios:
            raise ValueError(
                f"compress_ratios must be non-negative, got {invalid_ratios}"
            )

        # DSpark speculative decoding lives in the mtp.* checkpoint namespace.
        # The backbone is served without it; refuse only if the runner was
        # explicitly configured for speculative decoding.
        from vllm.config import get_current_vllm_config

        try:
            vllm_config = get_current_vllm_config()
        except Exception:
            vllm_config = None
        if vllm_config is not None and vllm_config.speculative_config is not None:
            raise ValueError(
                "Speculative decoding is not supported for DeepSeek-V4. The "
                "checkpoint's DSpark draft stages (mtp.*) are not implemented; "
                "run without --speculative-config."
            )
