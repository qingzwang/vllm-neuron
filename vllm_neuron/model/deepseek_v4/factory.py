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

        # Checkpoint FP8/FP4 weights are dequantized to bf16 during load, so the
        # only meaningful compute path is bf16. "deepseek_v4_fp8" arrives here
        # when vLLM resolves it from the checkpoint's quantization_config; it
        # describes the checkpoint, not a compute mode, so it is accepted.
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16", "deepseek_v4_fp8"):
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
        # attends over every compressed slot (HCA).
        invalid_ratios = sorted(r for r in config.compress_ratios if r < 0)
        if invalid_ratios:
            raise ValueError(
                f"compress_ratios must be non-negative, got {invalid_ratios}"
            )

        # The indexer's top-k only runs when the compressed stream is longer
        # than index_topk. XLA lowers top-k to `sort`, which neuronx-cc rejects
        # on trn2/trn3, so that path cannot compile today. Below the threshold
        # the indexer degenerates to "attend over every slot" and is skipped
        # entirely, which is what makes shorter contexts work.
        from vllm.config import get_current_vllm_config

        try:
            vllm_config = get_current_vllm_config()
        except Exception:
            vllm_config = None

        csa_ratios = [r for r in config.compress_ratios if r == 4]
        if csa_ratios and vllm_config is not None:
            max_model_len = vllm_config.model_config.max_model_len
            max_slots = max_model_len // 4
            if max_slots > config.index_topk:
                raise ValueError(
                    f"max_model_len={max_model_len} yields {max_slots} compressed "
                    f"slots on the CSA layers, above index_topk="
                    f"{config.index_topk}. That engages the indexer's top-k, "
                    "which XLA lowers to `sort` — an operation neuronx-cc does "
                    "not support. Serve with "
                    f"--max-model-len <= {config.index_topk * 4} until the "
                    "top-k is reimplemented with a NKI kernel."
                )

        # The compressed streams are model-owned buffers holding one request's
        # state at a time, so concurrent requests would read each other's
        # compressed KV. See compressed_state.py.
        has_compression = any(r for r in config.compress_ratios)
        if has_compression and vllm_config is not None:
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            if max_num_seqs != 1:
                raise ValueError(
                    f"max_num_seqs={max_num_seqs} is not supported for "
                    "DeepSeek-V4: the compressed KV streams (CSA/HCA layers) "
                    "are model-owned buffers scoped to a single request, so "
                    "concurrent requests would corrupt each other's compressed "
                    "state. Serve with --max-num-seqs 1."
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
