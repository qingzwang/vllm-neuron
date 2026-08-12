# InternVL3-8B onboarding — state and remaining work

Branch `model/InternVL3-8B`, cut from `release-0.21.0.1.0.0`. No overlap with
`benchmark/Qwen3-VL-8B`.

## Done

| Piece | File | Status |
|---|---|---|
| Configs | `config.py` | used by the verified pieces below |
| InternViT-300M tower | `vision_encoder.py` | **verified vs HF**, rel 8.0e-6 |
| pixel shuffle + projector | `projector.py` | **verified vs HF**, rel 2.5e-6 |
| Qwen2 decoder layers | `text_model.py` | written, imports, **not executed** |
| Factory | `factory.py` | written, **not executed** (imports `.model`, absent) |

Validator: `examples/vllm_neuron/models/internvl/validate_vision_encoder.py`
(CPU, float32, real checkpoint). Run with `VLLM_NEURON_CPU_MODE=1`; takes seconds.

## Remaining

1. **`model.py`** — top-level `InternVLChatModel`, the only blocker for a first
   run. Needs: `InternVLTextModel` (embed_tokens + 28 `Qwen2DecoderLayer` + norm),
   `lm_head`, `forward`, `embed_multimodal`, `get_kv_spec`, `bind_kv_cache`,
   `load_weights`, `from_configs`.
2. **Register** `("InternVLChatModel", InternVLChatModel)` in
   `vllm_neuron/model/registry.py` and export from `__init__.py`.
3. **On-device TP=4 smoke test.** This is also what validates `text_model.py`
   numerically and the vision encoder's TP sharding — a wrong shard shows up
   immediately as garbage output, so **read the generated text, not just the
   latency numbers.**

## Contract notes (the expensive part to rediscover)

### Where registration happens, and why the mm processor is free

Model registration runs **only in the worker process**
(`vllm_neuron/vllm/worker/neuron_worker.py:251`);
`platform.pre_register_and_update` registers nothing but the synthetic test model.
So the frontend keeps vLLM's own `InternVLChatModel` and its multimodal
processor. **Dynamic tiling, prompt replacement and `get_num_image_tokens` need
no reimplementation.** The plugin replaces execution only.

### What arrives in `embed_multimodal`

The runner calls, per modality group
(`neuron_model_runner.py` around line 2265):

```python
model.embed_multimodal(encoder_cache=..., mm_hashes=[...], **mm_kwargs_batch)
```

For InternVL, `mm_kwargs_batch` carries vLLM's InternVL schema:

- `pixel_values_flat`: `[total_tiles, 3, 448, 448]` — flat across all images
- `image_num_patches`: `[num_images]` — tiles per image

The method must run the tower + projector and scatter results into the on-device
encoder cache (`EncoderCacheBlocks`), then the runner marks each `mm_hash`
written. Follow `qwen3_vl/model_bf16.py:1051` `embed_multimodal` — allocate per
item by merged token count, let the write go straight into the cache buffer, no
device→host round trip.

Sizing: **256 embed tokens per tile**, so an item with `n` tiles needs
`ceil(n * 256 / cache_block_size)` blocks. Note the block-vs-token budget hazard
recorded for Qwen3-VL: the scheduler admits items on a token budget while the
allocator works in blocks, so per-item padding waste can exhaust the allocator
mid-stream. Expect to have to set `encoder_cache_num_blocks` explicitly.

### `forward` signature and the vision merge

Mirror `qwen3_vl/model_bf16.py:971`, minus the M-RoPE argument — InternVL uses
plain 1-D RoPE, so there is no `rotary_position_ids` and **no**
`SupportsMRoPE`/`get_mrope_input_positions`. Positions come in as the ordinary
`positions` tensor.

Vision embeddings reach `forward` as `vision_embedding_blocks`
(tuple of `[block_size, fat_dim]` cache views) plus `vision_positions`
(`[max_num_vision_blocks, block_size]`). Reuse
`qwen3_vl/utils/merge_vision_embeds.py::merge_vision_embeddings` — it infers the
deepstack count from `fat_dim // visual_dim - 1`, which is **0** for InternVL, so
it returns `deepstack=None` and the layer loop needs no deepstack injection.

`fat_dim` is therefore just `llm_hidden` (3584), unlike Qwen3-VL where it is
`out_hidden_size * (1 + num_deepstack_levels)`.

### SP layout

Prefill runs sequence-parallel: `embed_tokens(..., scatter_tokens=is_prefill)`,
each layer reduce-scatters, and the backbone all-gathers before `norm`. Decode
all-reduces instead. `text_model.py` already follows this; the top-level model
must keep the `is_prefill = max_query_len > decode_token_threshold` test that
drives it.

### Checkpoint key prefixes

- text: `language_model.model.layers.{i}.*`, `language_model.model.embed_tokens.weight`,
  `language_model.model.norm.weight`, `language_model.lm_head.weight`
  (**note**: no `model.` prefix in front of `language_model`, unlike Qwen3-VL's
  `model.language_model`)
- vision: `vision_model.*` — mapping built by
  `InternVisionModel.build_weight_mappings()`
- projector: `mlp1.{0,1,3}.*` — `InternVLProjector.build_weight_mappings()`

`tie_word_embeddings` is False, so `lm_head.weight` is a real tensor.

## Architecture facts worth not re-deriving

| | value |
|---|---|
| LLM | Qwen2, hidden 3584, 28 layers, 28 Q / 4 KV heads, head_dim 128, inter 18944 |
| LLM specifics | q/k/v **have bias**; o_proj does not; **no** QK norm; rope_theta 1e6 |
| Vision | InternViT-300M, hidden 1024, 24 layers, 16 heads, **head_dim 64**, inter 4096 |
| Vision specifics | patch **14** (not 16), `qkv_bias=True`, **LayerScale** `ls1`/`ls2`, plain MLP |
| Tiles | always 448x448 -> 32x32 = 1024 patches -> 256 embed tokens |
| TP=4 | q/kv per rank 7/1; `fused_qkv_dim` = 1152 |

TP=8 does not divide 28 Q heads. At TP=1 `fused_qkv_dim` is 4608, which the
existing `> 4096` guard sends to the PyTorch fallback.

> The QKV-kernel fix from the `benchmark/Qwen3-VL-8B` branch is deliberately
> **not** on this branch. InternVL3 at TP=4 sits at 1152, far below the broken
> 3072–3583 range, so it is not needed. If you ever try other TP values here,
> that bug exists: the kernel silently returns garbage for
> `fused_qkv_dim >= 3072`.

## Gotchas already paid for

- **LayerScale** — miss `ls1`/`ls2` and weights load cleanly while output is wrong.
- **`pixel_shuffle`'s trailing permute** — the only v1/v2 difference; the two
  permutes do not cancel. Dropping it silently reorders every vision token.
- **Position-embedding interpolation** — HF bicubic-interpolates it, but source
  and target grids are both `image_size/patch_size`, so it is the identity and is
  skipped. `config.py` raises on configs where that would not hold.
- **`timm`** — not in the DLAMI venv, and the HF reference imports `DropPath` from
  it. `drop_path_rate=0.0` here, so the validator stubs it as the identity. It
  also has to import the checkpoint's modeling file as a synthetic package
  because of its relative imports, and it must let `transformers` finish its
  lazy init before the stub is installed.
