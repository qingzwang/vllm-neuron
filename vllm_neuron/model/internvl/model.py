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
from vllm_neuron.model.interfaces import SupportsVisionWarmup
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig
from vllm_neuron.model.qwen3_vl.utils.merge_vision_embeds import merge_vision_embeddings
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
)

from .config import InternVLConfig, InternVLTextConfig
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

        # No weight loader is attached here on purpose: VocabDimShardedEmbedding
        # installs its own with pad_shard=True, which InternVL3 needs because its
        # vocab (151674) is not divisible by the TP size. Overriding it with a
        # non-padding loader makes the last rank's slice run past the end of the
        # tensor, and strict=False then silently leaves the parameter on meta.

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

        # cos/sin cover the FULL sequence, not this rank's SP slice: attention and
        # MLP each all-gather at entry during prefill, so they see all T tokens.
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


class InternVLChatModel(nn.Module, SupportsVisionWarmup):
    """InternVL3: vision tower + projector + Qwen2 decoder + LM head."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.config = config
        self.text_config = config.text_config

        dtype = config.text_config.torch_dtype
        # The projector lives inside the tower so one compiled module spans
        # pixels -> LLM space, matching Qwen3VLVisionModel.
        self.visual = InternVisionModel(config, dtype=dtype)
        self.language_model = InternVLTextModel(config.text_config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.text_config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (nc is not None and nc.max_logprobs != 0) or debug_logits

        # <-- MODEL-SPECIFIC: InternVL3's vocab is 151674, which is NOT divisible
        # by 4, and ColumnParallelLinear asserts divisibility (nn/cpl.py has a
        # "TODO: Add flag to enable padding"). Every other model in this plugin
        # happens to have a divisible vocab, so this is the first one to need it.
        # Round the LM head up to a multiple of world_size and mask the padding
        # out of the logits — zero-padded weights would otherwise yield logit 0,
        # which can beat genuinely negative logits and silently emit a token
        # outside the vocabulary.
        vocab = self.text_config.vocab_size
        self.vocab_size_per_rank = math.ceil(vocab / self.world_size)
        self.padded_vocab_size = self.vocab_size_per_rank * self.world_size

        # >>> PARALLELISM: column-parallel LM head <<<
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.text_config.hidden_size,
            self.padded_vocab_size,
            bias=False,
            dtype=dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

        # How many of this rank's logit columns fall past the real vocabulary.
        # Computed from plain ints: the model is constructed under a meta device,
        # where building a mask tensor and calling .any() would raise
        # "Tensor.item() cannot be called on meta tensors".
        start = self.rank * self.vocab_size_per_rank
        end = start + self.vocab_size_per_rank
        self._num_pad_cols = min(self.vocab_size_per_rank, max(0, end - vocab))
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
        if self._num_pad_cols:
            # Padding always occupies the tail of the last rank's shard. Rebuild
            # the row by concatenation rather than an in-place write so the shape
            # stays static and the graph stays functional. Zero-weight columns
            # would otherwise score 0 and can beat genuinely negative logits,
            # emitting a token id outside the vocabulary.
            real = logits[..., : -self._num_pad_cols]
            pad = torch.full_like(logits[..., -self._num_pad_cols :], float("-inf"))
            logits = torch.cat([real, pad], dim=-1)

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
        pixel_values_flat_video: torch.Tensor | None = None,
        video_num_patches: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Encode tiles and scatter the result into the on-device encoder cache.

        Args:
            pixel_values_flat: ``[total_tiles, 3, 448, 448]``, flat across images.
            image_num_patches: ``[num_images]`` — tiles belonging to each image.
            encoder_cache: ``EncoderCacheBlocks`` (buffer + allocate).
            mm_hashes: One identifier per item, same order as the count tensor.
            pixel_values_flat_video: ``[total_frames, 3, 448, 448]``, video
                equivalent of ``pixel_values_flat``.
            video_num_patches: ``[num_videos]`` — frames belonging to each video.

        Every tile yields exactly ``embed_tokens_per_tile`` (256) tokens, so an
        image with n tiles occupies ``n * 256`` rows of the cache — no packing
        decisions to make, unlike Qwen3-VL's variable-size items.

        Video needs no separate path. vLLM's video processor asserts exactly one
        tile per frame (``video_to_pixel_values_internvl``), so a video arrives as
        one tile per frame in the same layout an image uses, and a frame costs the
        same 256 tokens as a tile. There is no temporal merging and no timestamp
        text, so unlike Qwen3-VL a frame is just an independent single-tile image;
        only the kwarg names differ.
        """
        # The runner groups multimodal kwargs by modality, so exactly one pair
        # arrives per call. Enforce it: silently dropping one modality would only
        # surface as wrong output far downstream.
        if pixel_values_flat is not None and pixel_values_flat_video is not None:
            raise ValueError(
                "embed_multimodal: cannot supply both pixel_values_flat and "
                "pixel_values_flat_video in a single call; caller must group by "
                "modality."
            )
        if pixel_values_flat is None and pixel_values_flat_video is not None:
            pixel_values_flat = pixel_values_flat_video
            image_num_patches = video_num_patches

        if pixel_values_flat is None:
            raise ValueError(
                "embed_multimodal requires pixel_values_flat (images) or "
                "pixel_values_flat_video (video)"
            )
        if image_num_patches is None or mm_hashes is None:
            raise ValueError(
                "embed_multimodal requires a per-item tile/frame count "
                "(image_num_patches or video_num_patches) and mm_hashes"
            )

        per_tile = self.config.embed_tokens_per_tile
        block_size = encoder_cache.block_size
        if per_tile % block_size and block_size % per_tile:
            raise ValueError(
                f"embed tokens per tile ({per_tile}) and cache block size "
                f"({block_size}) must be multiples of one another"
            )

        tiles = [int(n) for n in image_num_patches.tolist()]
        if sum(tiles) != pixel_values_flat.shape[0]:
            raise ValueError(
                f"per-item counts sum to {sum(tiles)} tiles/frames but got "
                f"{pixel_values_flat.shape[0]} tile images"
            )

        # Allocate every item's blocks first, then hand the whole flat batch to the
        # compiled tower in one call: it scatter-writes into the buffer in-graph.
        write_block_ids: list[int] = []
        for mm_hash, n_tiles in zip(mm_hashes, tiles):
            num_tokens = n_tiles * per_tile
            num_blocks = math.ceil(num_tokens / block_size)
            tokens_per_block = [
                min(block_size, num_tokens - b * block_size) for b in range(num_blocks)
            ]
            write_block_ids.extend(encoder_cache.allocate(mm_hash, tokens_per_block))

        # The compiled graph has a static tile count, so pad the batch up to the
        # selected bucket and send the padding blocks to the scratch block, which
        # exists precisely to absorb writes that are not real cache entries.
        real_tiles = pixel_values_flat.shape[0]
        bucket = self._select_vision_bucket(real_tiles)
        padded_tiles = self._padded_tile_count(
            bucket, self.config.vision_neuron_config
        )
        # The runner hands multimodal kwargs over on CPU (they come straight off
        # the mm processor), so the move to device happens here — same as
        # qwen3_vl's embed_multimodal. Without it the traced graph mixes a CPU
        # activation with device parameters and dynamo fails on the first matmul.
        device = next(self.visual.parameters()).device
        px = pixel_values_flat.to(device=device, dtype=self.text_config.torch_dtype)
        if padded_tiles > real_tiles:
            pad = px.new_zeros((padded_tiles - real_tiles, *px.shape[1:]))
            px = torch.cat([px, pad], dim=0)

        scratch = encoder_cache.scratch_block_id
        while len(write_block_ids) < padded_tiles * per_tile // block_size:
            write_block_ids.append(scratch)

        ids = torch.tensor(
            write_block_ids, dtype=torch.int64, device=encoder_cache.buffer.device
        )
        self.visual(px, encoder_cache.buffer, ids)

    def build_vision_synthetic_inputs(
        self,
        bucket: int,
        vision_neuron_config: VisionNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Shape-only inputs matching ``visual.forward``, for per-bucket warmup.

        Declaring SupportsVisionWarmup is what makes the runner pre-compile the
        vision graph; without it the graph is only built on the first real request,
        where it looks like a hang.

        Buckets count raw (pre-merge) patches, so a bucket is just a tile count:
        ``bucket / patches_per_tile`` tiles of ``[3, image_size, image_size]``.
        """
        vc = self.config.vision_config
        patches_per_tile = vc.num_patches_per_tile
        num_tiles = self._padded_tile_count(bucket, vision_neuron_config)
        return {
            "pixel_values": torch.zeros(
                num_tiles,
                vc.num_channels,
                vc.image_size,
                vc.image_size,
                dtype=self.text_config.torch_dtype,
                device=device,
            ),
            # The runner overwrites write_block_ids with
            #   zeros(ceil(bucket / vision_attention_block_size) padded to dp_size)
            # so this only has to agree on the length.
            "write_block_ids": torch.zeros(num_tiles, dtype=torch.int64, device=device),
        }

    def _padded_tile_count(
        self, bucket: int, vision_neuron_config: VisionNeuronConfig
    ) -> int:
        """Tiles the compiled vision graph consumes for a bucket.

        Must match the block count the runner derives, because it builds
        write_block_ids itself as
        ``ceil(bucket / vision_attention_block_size)`` padded up to ``dp_size``.
        One InternVL tile is one VE block, so tiles and blocks are the same number.
        A mismatch shows up as an XLA lowering failure on index_put_ ("Input
        dimension should be either 1 or equal to the output dimension"), because the
        value tensor then has fewer rows than the index.
        """
        block = vision_neuron_config.vision_attention_block_size
        dp = max(1, vision_neuron_config.dp_size)
        return math.ceil(math.ceil(bucket / block) / dp) * dp

    def _select_vision_bucket(self, num_tiles: int) -> int:
        """Smallest configured bucket that holds ``num_tiles`` tiles."""
        vnc = self.config.vision_neuron_config
        patches = num_tiles * self.config.vision_config.num_patches_per_tile
        buckets = sorted(vnc.num_vision_tokens_buckets or [])
        for b in buckets:
            if b >= patches:
                return b
        raise ValueError(
            f"{num_tiles} tiles need {patches} raw vision patches, above the "
            f"largest configured bucket ({buckets[-1] if buckets else None})"
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

        # Vision weights load separately: the tower has its own TP group, so it
        # must be sharded by the vision rank rather than the text rank.
        self.visual.load_weights(checkpoint_path, device="cpu", cpu_mode=True)

        self._assert_no_meta_params()

    def _assert_no_meta_params(self) -> None:
        """Fail with the offending names if any parameter is still on meta.

        The model is constructed on the meta device and materialized by
        load_weights. A name missing from the checkpoint mapping stays on meta and
        surfaces much later as torch's "Cannot copy out of meta tensor" from
        model.to(device), which names nothing. List them here instead.
        """
        stranded = [
            name
            for name, tensor in list(self.named_parameters())
            + list(self.named_buffers())
            if tensor.device.type == "meta"
        ]
        if stranded:
            raise RuntimeError(
                f"{len(stranded)} parameter(s) were never loaded and remain on "
                f"the meta device: {stranded[:12]}"
                + (" ..." if len(stranded) > 12 else "")
            )
