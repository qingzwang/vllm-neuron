# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4 Configuration
=========================
<-- MODEL-SPECIFIC: All fields in this config are model-specific.

DeepSeek-V4-Flash architecture overview:
- 43 transformer blocks, each wrapped in Manifold-Constrained Hyper-Connections
  (mHC): the residual stream carries ``hc_mult`` copies of the hidden state
  instead of one, mixed by Sinkhorn-normalized learned weights.
- Attention is Multi-head Latent Attention (MLA): low-rank Q (wq_a -> q_norm ->
  wq_b), a single ``head_dim``-wide latent KV shared by all Q heads, and a
  grouped low-rank output projection (wo_a -> wo_b).
- Hybrid attention: every layer attends over a ``sliding_window`` of recent
  latent KV. Layers with ``compress_ratios[i] != 0`` additionally attend over a
  compressed KV stream: ratio 4 layers use Compressed Sparse Attention (CSA,
  top-k selected by a learned Indexer), ratio 128 layers use Heavily Compressed
  Attention (HCA, all compressed slots).
- MoE feed-forward with ``n_routed_experts`` FP4 experts + 1 shared expert.
  The first ``num_hash_layers`` layers route by a fixed token-id -> expert table
  (``tid2eid``); the rest route by a sqrt-softplus score with a learned bias.
- The ``mtp.*`` checkpoint namespace holds DSpark block-wise speculative
  decoding (3 stages, Markov + confidence heads). Not built by this
  implementation — see the model README.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class DeepseekV4Config:
    """Configuration for the DeepSeek-V4 family (Flash / Pro).

    Field names follow the HuggingFace ``config.json`` keys so
    :meth:`from_configs` can filter the dict directly. The reference
    implementation shipped in the checkpoint's ``inference/`` directory uses
    shorter names (``dim``, ``n_layers``, ...); the mapping is noted per field
    where it differs.
    """

    # ── Model architecture ───────────────────────────────────────────────
    vocab_size: int = 129280
    hidden_size: int = 4096  # reference: dim
    num_hidden_layers: int = 43  # reference: n_layers
    num_attention_heads: int = 64  # reference: n_heads
    rms_norm_eps: float = 1e-6  # reference: norm_eps
    torch_dtype: torch.dtype = torch.bfloat16
    tie_word_embeddings: bool = False

    # ── MLA attention ────────────────────────────────────────────────────
    head_dim: int = 512  # latent KV width, shared across Q heads
    qk_rope_head_dim: int = 64  # reference: rope_head_dim
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    num_key_value_heads: int = 1  # MLA: one latent KV head

    # ── Hybrid attention (sliding window + CSA/HCA) ──────────────────────
    sliding_window: int = 128  # reference: window_size
    # Per-layer KV compression ratio. 0 = sliding-window only, 4 = CSA
    # (indexer-selected top-k), 128 = HCA (all compressed slots).
    compress_ratios: tuple[int, ...] = ()

    # ── Indexer (CSA top-k selection) ────────────────────────────────────
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512

    # ── MoE ──────────────────────────────────────────────────────────────
    moe_intermediate_size: int = 2048  # reference: moe_inter_dim
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6  # reference: n_activated_experts
    num_hash_layers: int = 3  # reference: n_hash_layers
    scoring_func: str = "sqrtsoftplus"  # reference: score_func
    routed_scaling_factor: float = 1.5  # reference: route_scale
    norm_topk_prob: bool = True
    topk_method: str = "noaux_tc"
    swiglu_limit: float = 10.0
    expert_dtype: str | None = "fp4"

    # ── RoPE (YaRN) ──────────────────────────────────────────────────────
    rope_theta: float = 10000.0
    # Compressed streams use a longer base than the sliding-window stream.
    compress_rope_theta: float = 160000.0
    rope_scaling: dict | None = field(
        default_factory=lambda: {
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        }
    )
    max_position_embeddings: int = 1048576

    # ── Hyper-connections (mHC) ──────────────────────────────────────────
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ── Quantization ─────────────────────────────────────────────────────
    # HF: {"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0",
    #      "weight_block_size": [128, 128], "activation_scheme": "dynamic"}
    quantization_config: dict | None = None

    # ── DSpark speculative decoding (not implemented) ─────────────────────
    # Present in the -0731 revision only. Retained so validation can reject
    # configurations that ask for it rather than silently ignoring it.
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256
    num_nextn_predict_layers: int = 0

    # ── Special tokens ───────────────────────────────────────────────────
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int | None = None

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if not self.compress_ratios:
            # Default to sliding-window-only on every layer.
            self.compress_ratios = (0,) * self.num_hidden_layers
        else:
            self.compress_ratios = tuple(self.compress_ratios)

        # The -0731 checkpoint lists 46 ratios for 43 backbone layers: the
        # trailing entries belong to the DSpark stages, which reuse the
        # ``mtp.*`` namespace and are always sliding-window-only. Truncate so
        # per-layer indexing stays aligned with the backbone.
        if len(self.compress_ratios) > self.num_hidden_layers:
            self.compress_ratios = self.compress_ratios[: self.num_hidden_layers]
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries but "
                f"num_hidden_layers is {self.num_hidden_layers}"
            )

        self.dspark_target_layer_ids = tuple(self.dspark_target_layer_ids)

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def qk_nope_head_dim(self) -> int:
        """Width of the non-RoPE (content) part of the latent KV."""
        return self.head_dim - self.qk_rope_head_dim

    @property
    def is_hash_layer(self) -> tuple[bool, ...]:
        """Per-layer flag: True where MoE routing uses the tid2eid table."""
        return tuple(i < self.num_hash_layers for i in range(self.num_hidden_layers))

    @property
    def quantization_block_size(self) -> tuple[int, int]:
        """Block size of the FP8 weight scales, as (out, in)."""
        if self.quantization_config:
            bs = self.quantization_config.get("weight_block_size", [128, 128])
            return (int(bs[0]), int(bs[1]))
        return (128, 128)

    @property
    def scale_fmt(self) -> str | None:
        """Scale storage format, e.g. ``"ue8m0"`` (power-of-two scales)."""
        if self.quantization_config:
            return self.quantization_config.get("scale_fmt")
        return None

    @property
    def has_dspark(self) -> bool:
        return self.dspark_block_size > 0

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ) -> "DeepseekV4Config":
        """Create config from a HuggingFace config + NeuronConfig.

        Accepts a ``PretrainedConfig``, a path to a ``config.json``, or a plain
        dict, mirroring the other model implementations in this plugin.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
        else:
            config_dict = dict(hf_config)

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}

        if isinstance(filtered.get("torch_dtype"), str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

        # HF stores eos_token_id as either an int or a list of ints.
        eos = filtered.get("eos_token_id")
        if isinstance(eos, (list, tuple)):
            filtered["eos_token_id"] = int(eos[0])

        filtered["neuron_config"] = neuron_config

        return cls(**filtered)
