# SPDX-License-Identifier: Apache-2.0
"""InternVL3 top-level model (BF16): InternViT + projector + Qwen2 decoder.

Structure follows ``vllm_neuron/model/qwen3_vl/model_bf16.py``. Differences that
shape this file:

  * **No M-RoPE.** InternVL image tokens sit at ordinary sequential positions, so
    ``forward`` takes no ``rotary_position_ids`` and the model does not implement
    ``SupportsMRoPE``.
  * **No DeepStack.** Only the final ViT layer feeds the LLM, so the encoder cache
    ``fat_dim`` is just ``llm_hidden`` and the shared merge helper returns
    ``deepstack=None`` — the decoder loop injects nothing.
  * **Fixed-size vision items.** Every tile is 448x448 -> 1024 patches -> 256
    embed tokens, so encoder-cache sizing is plain arithmetic on the tile count
    rather than Qwen3-VL's bin-packing.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    InternVL-specific. Change when porting.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig
from vllm_neuron.model.qwen3_vl.utils.merge_vision_embeds import merge_vision_embeddings
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from .config import InternVLConfig, InternVLTextConfig
from .projector import InternVLProjector
from .text_model import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
from .vision_encoder import InternVisionModel

# HF checkpoint prefix for the text weights. Note there is no leading "model."
# in front of language_model, unlike Qwen3-VL's "model.language_model".
HF_TEXT_PREFIX = "language_model.model"


class InternVLTextModel(nn.Module):
    """Qwen2 backbone: embed_tokens -> N decoder layers -> norm."""

    def __init__(self, config: InternVLTextConfig):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group, also used for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: vocab-sharded embedding, handles SP scatter itself <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                is_storage_transposed=False,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_layer = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer]["max_query_len"]
        threshold = attn_metadata[first_layer]["decode_token_threshold"]
        is_prefill = max_query_len > threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        # Vision merge is prefill-only; decode carries no vision inputs.
        if (
            is_prefill
            and vision_embedding_blocks is not None
            and vision_positions is not None
        ):
            # <-- MODEL-SPECIFIC: no DeepStack, so the second return is None.
            hidden_states, _ = merge_vision_embeddings(
                hidden_states,
                vision_embedding_blocks,
                vision_positions,
                rank=self.rank,
            )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                is_decode=not is_prefill,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — reconstruct the full sequence before the LM head <<<
        if is_prefill:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


class InternVLChatModel(nn.Module):
    """InternVL3: vision tower + projector + Qwen2 decoder + LM head."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.config = config
        self.text_config = config.text_config

        dtype = config.text_config.torch_dtype
        self.visual = InternVisionModel(config.vision_config, dtype=dtype)
        self.projector = InternVLProjector(config, dtype=dtype)
        self.language_model = InternVLTextModel(config.text_config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.text_config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (nc is not None and nc.max_logprobs != 0) or debug_logits

        # >>> PARALLELISM: column-parallel LM head <<<
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            dtype=dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.text_config.vocab_size // self.world_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=self.text_config.hidden_size,
                unpadded_size=self.text_config.hidden_size,
            ),
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    # ── Forward ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        first_layer = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer]["max_query_len"]
        threshold = attn_metadata[first_layer]["decode_token_threshold"]
        is_prefill = max_query_len > threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt length ({T}) must exceed and divide world_size "
                f"({self.world_size}) for sequence parallelism."
            )

        hidden_states = self.language_model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            vision_embedding_blocks=vision_embedding_blocks,
            vision_positions=vision_positions,
        )

        hidden_states = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states)

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        if self.on_device_sampling_config is None:
            return logits

        sampled = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled)
        return sampled, gathered_logits

    # ── Vision ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_multimodal(
        self,
        pixel_values_flat: torch.Tensor | None = None,
        image_num_patches: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Encode tiles and scatter the result into the on-device encoder cache.

        Args:
            pixel_values_flat: ``[total_tiles, 3, 448, 448]``, flat across images.
            image_num_patches: ``[num_images]`` — tiles belonging to each image.
            encoder_cache: ``EncoderCacheBlocks`` (buffer + allocate).
            mm_hashes: One identifier per image, same order as image_num_patches.

        Every tile yields exactly ``embed_tokens_per_tile`` (256) tokens, so an
        image with n tiles occupies ``n * 256`` rows of the cache — no packing
        decisions to make, unlike Qwen3-VL's variable-size items.
        """
        if pixel_values_flat is None:
            raise ValueError("embed_multimodal requires pixel_values_flat")
        if image_num_patches is None or mm_hashes is None:
            raise ValueError(
                "embed_multimodal requires image_num_patches and mm_hashes"
            )

        dtype = self.text_config.torch_dtype
        vit_embeds = self.visual(pixel_values_flat.to(dtype))
        embeds = self.projector(vit_embeds)  # [total_tiles, 256, llm_hidden]

        per_tile = self.config.embed_tokens_per_tile
        llm_hidden = self.text_config.hidden_size
        block_size = encoder_cache.block_size

        tiles = [int(n) for n in image_num_patches.tolist()]
        if sum(tiles) != embeds.shape[0]:
            raise ValueError(
                f"image_num_patches sums to {sum(tiles)} tiles but the encoder "
                f"produced {embeds.shape[0]}"
            )

        tile_cursor = 0
        for mm_hash, n_tiles in zip(mm_hashes, tiles):
            item = embeds[tile_cursor : tile_cursor + n_tiles]
            tile_cursor += n_tiles
            # [n_tiles, 256, hidden] -> [n_tiles * 256, hidden]
            item = item.reshape(n_tiles * per_tile, llm_hidden)

            num_tokens = item.shape[0]
            num_blocks = math.ceil(num_tokens / block_size)
            # allocate() wants the token count carried by each block, so the
            # final block reports the remainder rather than a full block_size.
            tokens_per_block = [
                min(block_size, num_tokens - b * block_size) for b in range(num_blocks)
            ]
            block_ids = encoder_cache.allocate(mm_hash, tokens_per_block)

            for b, (block_id, n_rows) in enumerate(zip(block_ids, tokens_per_block)):
                start = b * block_size
                rows = item[start : start + n_rows]
                encoder_cache.buffer[block_id, :n_rows] = rows.to(
                    encoder_cache.buffer.dtype
                )

    # ── Config / KV cache / weights ──────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> "InternVLChatModel":
        config = InternVLConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        return cls(config)

    def get_kv_spec(self) -> KVSpec:
        layers = []
        for i, layer in enumerate(self.language_model.layers):
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for i, layer in enumerate(self.language_model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise Exception(f"KV cache for layer {name} not initialized")
            layer.self_attn.k_cache = kv_caches[name][0]
            layer.self_attn.v_cache = kv_caches[name][1]

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load text, vision and projector weights.

        >>> PARALLELISM: the weight loaders attached in each module do the
        TP sharding; this only supplies the checkpoint key mapping. <<<
        """
        mappings: dict[str, str | list[str]] = {}

        for layer_id, layer in enumerate(self.language_model.layers):
            hf_prefix = f"{HF_TEXT_PREFIX}.layers.{layer_id}"
            our_prefix = f"language_model.layers.{layer_id}"
            for name, key in layer.build_weight_mappings(hf_prefix).items():
                mappings[f"{our_prefix}.{name}"] = key

        mappings["language_model.embed_tokens.weight"] = (
            f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        )
        mappings["language_model.norm.weight"] = f"{HF_TEXT_PREFIX}.norm.weight"
        # tie_word_embeddings is False for InternVL3-8B, so lm_head is a real tensor.
        if not self.text_config.tie_word_embeddings:
            mappings["lm_head.weight"] = "language_model.lm_head.weight"

        for name, key in self.visual.build_weight_mappings().items():
            mappings[f"visual.{name}"] = key
        for name, key in self.projector.build_weight_mappings().items():
            mappings[f"projector.{name}"] = key

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank,
            self.world_size,
            self,
            mappings,
            device,
            strict=False,
        ).state_dict

        target_dtype = self.text_config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)
        self.load_state_dict(rank_sharded, strict=False, assign=True)
