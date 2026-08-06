# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 model for vLLM Neuron (bf16 compute).

Checkpoint weights are FP8 (most linears) and FP4 (routed experts); both are
dequantized to bf16 at load time because Trainium has no FP4 datapath. The
quantization-aware fake-quant the model was trained with is preserved in the
activation path (see :func:`~.layers.fake_quant_fp8`), so accuracy tracks the
reference implementation rather than a naive bf16 upcast.
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from .attention import DeepseekV4Attention
from .config import DeepseekV4Config
from .layers import DeepseekV4RMSNorm, HyperConnection, HyperConnectionHead
from .moe import DeepseekV4MoE
from .weight_loaders import cast_weight_loader


class DeepseekV4DecoderLayer(nn.Module):
    """One block: hyper-connected attention, then hyper-connected MoE.

    Unlike a standard pre-norm block, the residual stream is ``hc_mult`` copies
    wide. Each sublayer collapses the copies (``hc.pre``), runs on the single
    collapsed tensor, then mixes its output back into all copies (``hc.post``).
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.compress_ratio = config.compress_ratios[layer_idx]

        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.hc_attn = HyperConnection(
            config.hidden_size,
            config.hc_mult,
            config.hc_sinkhorn_iters,
            config.rms_norm_eps,
            config.hc_eps,
        )
        self.hc_ffn = HyperConnection(
            config.hidden_size,
            config.hc_mult,
            config.hc_sinkhorn_iters,
            config.rms_norm_eps,
            config.hc_eps,
        )

        for hc in (self.hc_attn, self.hc_ffn):
            for param in (hc.fn, hc.base, hc.scale):
                set_weight_loader(param, cast_weight_loader(out_dtype=torch.float32))
        for norm in (self.attn_norm, self.ffn_norm):
            set_weight_loader(
                norm.weight, cast_weight_loader(out_dtype=torch.float32)
            )

    def forward(
        self,
        streams: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Advance the ``[T, hc_mult, H]`` residual streams through this block."""
        residual = streams
        collapsed, post, comb = self.hc_attn.pre(streams)
        collapsed = self.attn_norm(collapsed)
        attn_out = self.self_attn(collapsed, positions, attn_metadata)
        streams = self.hc_attn.post(attn_out, residual, post, comb)

        residual = streams
        collapsed, post, comb = self.hc_ffn.pre(streams)
        collapsed = self.ffn_norm(collapsed)
        ffn_out = self.mlp(collapsed, input_ids)
        return self.hc_ffn.post(ffn_out, residual, post, comb)


class DeepseekV4Model(nn.Module):
    """The DeepSeek-V4 backbone: embedding, hyper-connected blocks, final norm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                is_storage_transposed=False,
            ),
        )

        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        set_weight_loader(
            self.norm.weight, cast_weight_loader(out_dtype=torch.float32)
        )

        self.hc_head = HyperConnectionHead(
            config.hidden_size, config.hc_mult, config.rms_norm_eps, config.hc_eps
        )
        for param in (self.hc_head.fn, self.hc_head.base, self.hc_head.scale):
            set_weight_loader(param, cast_weight_loader(out_dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns ``[T, H]`` final hidden states (streams already collapsed)."""
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=False, rank=rank
        )
        # Expand into the hyper-connection residual streams.
        streams = hidden_states.unsqueeze(-2).expand(-1, self.hc_mult, -1)

        for layer in self.layers:
            streams = layer(streams, positions, attn_metadata, input_ids)

        collapsed = self.hc_head(streams)
        return self.norm(collapsed)


class DeepseekV4ForCausalLM(nn.Module):
    """DeepSeek-V4 with the language modeling head and on-device sampling."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.model = DeepseekV4Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        neuron_config = config.neuron_config
        self.on_device_sampling_config = (
            neuron_config.on_device_sampling_config if neuron_config else None
        )
        debug_logits_enabled = (
            neuron_config is not None and neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            neuron_config is not None and neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.lm_head.out_features_per_rank,
                num_shards=self.lm_head.tp_size,
                is_storage_transposed=False,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        # Clamp into range: an out-of-range index faults the hardware gather
        # rather than returning garbage, and the runner's padding contract
        # (padding entries repeat the last real index) is not worth trusting
        # blind at this cost.
        hidden_states_for_logits = torch.index_select(
            hidden_states,
            dim=0,
            index=sampling_positions.clamp(0, hidden_states.shape[0] - 1),
        )
        logits = self.lm_head(hidden_states_for_logits)

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = (
                logits
                if self.lm_head.gather_output
                else self.tp_group.all_gather(logits, dim=1)
            )

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> "DeepseekV4ForCausalLM":
        config = DeepseekV4Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV cache ─────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        """Declare the sliding-window latent KV cache.

        Only the sliding-window stream is declared to the runner: it is the one
        addressed by ``slot_mapping``. Every layer gets a ``SlidingWindowSpec``
        (non-None ``sliding_window_size``) so the block manager only retains
        ``sliding_window`` positions per request rather than the full context —
        which is what makes a 1M-token context affordable.

        The compressed streams are model-owned; see
        :mod:`~.compressed_state` for why.
        """
        layers = []
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            layers.append(
                LayerSpec(
                    name=f"layers.{layer_idx}.self_attn",
                    num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(
        self, kv_caches: dict[str, list[torch.Tensor]]
    ) -> None:
        """Bind paged caches and allocate the compressed KV buffers."""
        for layer_idx, layer in enumerate(self.model.layers):
            layer_name = f"layers.{layer_idx}.self_attn"
            if layer_name not in kv_caches:
                raise RuntimeError(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

        self._allocate_compressed_states()

    def _allocate_compressed_states(self) -> None:
        """Allocate the per-layer compressed KV buffers.

        Sized from the vLLM config rather than the paged allocator, since these
        streams are model-owned — see :mod:`~.compressed_state`.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        max_model_len = vllm_config.model_config.max_model_len
        dtype = self.config.torch_dtype

        for layer in self.model.layers:
            attn = layer.self_attn
            if not attn.compress_ratio:
                continue
            device = attn.k_cache.device
            # One slot beyond the addressable capacity: decode steps that do not
            # close a compression window park their write there instead of
            # branching. See DeepseekV4Attention._append_compressed.
            max_slots = max(1, max_model_len // attn.compress_ratio) + 1

            attn.compressed_kv = torch.zeros(
                max_slots, attn.head_dim, dtype=dtype, device=device
            )
            if attn.indexer is not None:
                attn.compressed_index_kv = torch.zeros(
                    max_slots, attn.indexer.head_dim, dtype=dtype, device=device
                )
            attn.compress_window_hidden = torch.zeros(
                attn.compress_ratio,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            )
            # Shape [1] rather than a 0-d scalar: the FX inplace-to-outofplace
            # pass rewrites copy_ into slice_scatter, which cannot address a
            # 0-dimensional tensor.
            attn.compressed_length = torch.zeros(
                1, dtype=torch.int32, device=device
            )

    # ── Weight loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load a DeepSeek-V4 checkpoint, dequantizing FP8/FP4 to bf16.

        Checkpoint keys use the reference implementation's short names
        (``layers.N.attn.wq_a.weight``), not HuggingFace's. Quantized weights
        map to a ``[weight, scale]`` pair so the dequant loaders receive both.
        """
        mappings: dict[str, str | list[str]] = {
            "model.embed_tokens.weight": "embed.weight",
            "model.norm.weight": "norm.weight",
            "model.hc_head.fn": "hc_head_fn",
            "model.hc_head.base": "hc_head_base",
            "model.hc_head.scale": "hc_head_scale",
            "lm_head.weight": "head.weight",
        }

        for layer_idx in range(len(self.model.layers)):
            layer = self.model.layers[layer_idx]
            attn = layer.self_attn
            moe = layer.mlp
            param_prefix = f"model.layers.{layer_idx}"
            ckpt_prefix = f"layers.{layer_idx}"

            # Hyper-connections and block norms.
            for part, ckpt_part in (("hc_attn", "hc_attn"), ("hc_ffn", "hc_ffn")):
                mappings[f"{param_prefix}.{part}.fn"] = f"{ckpt_prefix}.{ckpt_part}_fn"
                mappings[f"{param_prefix}.{part}.base"] = (
                    f"{ckpt_prefix}.{ckpt_part}_base"
                )
                mappings[f"{param_prefix}.{part}.scale"] = (
                    f"{ckpt_prefix}.{ckpt_part}_scale"
                )
            mappings[f"{param_prefix}.attn_norm.weight"] = (
                f"{ckpt_prefix}.attn_norm.weight"
            )
            mappings[f"{param_prefix}.ffn_norm.weight"] = (
                f"{ckpt_prefix}.ffn_norm.weight"
            )

            # Attention: quantized projections take [weight, scale].
            attn_ckpt = f"{ckpt_prefix}.attn"
            attn_param = f"{param_prefix}.self_attn"
            for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
                mappings[f"{attn_param}.{name}"] = [
                    f"{attn_ckpt}.{name}.weight",
                    f"{attn_ckpt}.{name}.scale",
                ]
            mappings[f"{attn_param}.attn_sink"] = f"{attn_ckpt}.attn_sink"
            mappings[f"{attn_param}.q_norm.weight"] = f"{attn_ckpt}.q_norm.weight"
            mappings[f"{attn_param}.kv_norm.weight"] = f"{attn_ckpt}.kv_norm.weight"

            # Compressed streams.
            if attn.compress_ratio:
                comp_ckpt = f"{attn_ckpt}.compressor"
                comp_param = f"{attn_param}.compressor"
                mappings[f"{comp_param}.wkv"] = f"{comp_ckpt}.wkv.weight"
                mappings[f"{comp_param}.wgate"] = f"{comp_ckpt}.wgate.weight"
                mappings[f"{comp_param}.ape"] = f"{comp_ckpt}.ape"
                mappings[f"{comp_param}.norm.weight"] = f"{comp_ckpt}.norm.weight"

                if attn.indexer is not None:
                    idx_ckpt = f"{attn_ckpt}.indexer"
                    idx_param = f"{attn_param}.indexer"
                    mappings[f"{idx_param}.wq_b"] = [
                        f"{idx_ckpt}.wq_b.weight",
                        f"{idx_ckpt}.wq_b.scale",
                    ]
                    mappings[f"{idx_param}.weights_proj"] = (
                        f"{idx_ckpt}.weights_proj.weight"
                    )
                    icomp_ckpt = f"{idx_ckpt}.compressor"
                    icomp_param = f"{idx_param}.compressor"
                    mappings[f"{icomp_param}.wkv"] = f"{icomp_ckpt}.wkv.weight"
                    mappings[f"{icomp_param}.wgate"] = f"{icomp_ckpt}.wgate.weight"
                    mappings[f"{icomp_param}.ape"] = f"{icomp_ckpt}.ape"
                    mappings[f"{icomp_param}.norm.weight"] = (
                        f"{icomp_ckpt}.norm.weight"
                    )

            # MoE router. tid2eid is a buffer, loaded separately below.
            ffn_ckpt = f"{ckpt_prefix}.ffn"
            ffn_param = f"{param_prefix}.mlp"
            mappings[f"{ffn_param}.router_weight"] = f"{ffn_ckpt}.gate.weight"
            if not moe.is_hash_layer:
                mappings[f"{ffn_param}.router_bias"] = f"{ffn_ckpt}.gate.bias"

            # Routed experts: one [weight, scale] pair per local expert, in
            # expert order, matching the expert dequant loaders' contract.
            def expert_keys(ckpt_name: str) -> list[str]:
                keys: list[str] = []
                for local in range(moe.num_local_experts):
                    expert_id = moe.expert_start + local
                    base = f"{ffn_ckpt}.experts.{expert_id}.{ckpt_name}"
                    keys.extend([f"{base}.weight", f"{base}.scale"])
                return keys

            # The MoE kernel wants gate and up fused into one [E, H, 2, I]
            # tensor, so w1's pairs are followed by w3's in a single mapping.
            mappings[f"{ffn_param}.gate_up_proj_weight"] = expert_keys(
                "w1"
            ) + expert_keys("w3")
            mappings[f"{ffn_param}.down_proj_weight"] = expert_keys("w2")

            # Shared expert.
            for name in ("w1", "w2", "w3"):
                mappings[f"{ffn_param}.shared_expert.{name}"] = [
                    f"{ffn_ckpt}.shared_experts.{name}.weight",
                    f"{ffn_ckpt}.shared_experts.{name}.scale",
                ]

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        # Hash-routing tables are buffers, which the pipelined loader (which
        # walks named_parameters()) does not cover. Read them directly.
        self._load_hash_tables(checkpoint, device)

    def _load_hash_tables(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ) -> None:
        """Load the ``tid2eid`` token-id to expert tables for hash layers.

        Replicated across ranks (every rank must resolve any token's experts)
        and cast to int32 — the checkpoint stores int64, which would otherwise
        double a 129280 x 6 table per hash layer.
        """
        for layer_idx, layer in enumerate(self.model.layers):
            moe = layer.mlp
            if not moe.is_hash_layer:
                continue
            key = f"layers.{layer_idx}.ffn.gate.tid2eid"
            table = checkpoint._get_slice(key)[:]
            moe.tid2eid = table.to(torch.int32).to(device)

