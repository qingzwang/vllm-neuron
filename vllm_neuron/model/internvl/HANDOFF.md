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
| Factory | `factory.py` | logic verified against the real config |
| Top-level model | `model.py` | written, **not executed** |
| Registry entry | `registry.py`, `__init__.py` | import chain verified |

Validator: `examples/vllm_neuron/models/internvl/validate_vision_encoder.py`
(CPU, float32, real checkpoint). Run with `VLLM_NEURON_CPU_MODE=1`; takes seconds.

## Vision integration: done, matches the reference design

Restructured after `Qwen3VLVisionModel` and it now compiles and warms up on
device:

    Vision warmup [1/1]: bucket=13312
    Successfully warmed up vision bucket 13312
    init engine (profile, create kv cache, warmup model) took 192.38 s

What changed:

1. `InternVLProjector` moved **inside** `InternVisionModel`, so one compiled
   module spans pixels -> `[tiles, 256, llm_hidden]`. `tower()` and
   `encode_tiles()` are kept as staged helpers purely so the CPU validator can
   still compare each stage against HF — it still reports 8.0e-6 / 2.5e-6.
2. `forward(pixel_values, encoder_cache_buffer, write_block_ids)` scatter-writes
   with `index_put_` **inside** the graph.
3. `build_vision_synthetic_inputs` added and `SupportsVisionWarmup` declared, so
   the runner pre-compiles per bucket instead of the graph appearing on the first
   request.
4. `embed_multimodal` reduced to: allocate blocks, pad the batch to the bucket,
   route padding blocks to `scratch_block_id`, call `visual` once.

Two traps worth keeping in mind here:

- **The runner overwrites `write_block_ids`.** It builds its own as
  `zeros(ceil(bucket / vision_attention_block_size))` padded up to `dp_size`, so
  `build_vision_synthetic_inputs` only has to agree on the *length*. Getting that
  length wrong (13 vs the runner's dp-padded 16) surfaces as an XLA lowering
  failure on `index_put_` — "Input dimension should be either 1 or equal to the
  output dimension it is broadcasting into" — because the value tensor then has
  fewer rows than the index.
- **The compiled vision graph has a static tile count**, so a real request has to
  pad `pixel_values` up to the bucket and send the padding writes to the scratch
  block. That is what the scratch block is for.

Sizing for InternVL is unusually tidy: one 448x448 tile is 1024 raw patches, i.e.
one VE block at the default `vision_attention_block_size`, and produces exactly
256 merged tokens, i.e. exactly one cache block. Tiles, VE blocks and cache blocks
are all the same count.

## Remaining: EngineCore still never signals ready

**This is independent of the vision work above and survives it.** With vision
warmup succeeding and engine init logging completion at 192.38 s, EngineCore still
goes quiet right after the `NeuronAsyncScheduler` warning and never sends the ready
message. The parent stays blocked in `zmq.poll` under `wait_for_engine_startup`,
the four workers poll at ~8% with nothing to do, and EngineCore itself sits at
0.2% CPU — alive, blocked, not spinning.

So the ordering is now clear: weights load, all graphs (text **and** vision)
compile, warmup completes, and then the bring-up handshake does not finish. No
request is ever submitted.

**Next step, and the only one that will answer it:** get a stack out of the
EngineCore *process*. The parent's `faulthandler` dump does not cover it, and
`kill -ABRT` on it produced the parent's stack rather than EngineCore's. Options:

- `faulthandler.dump_traceback_later` installed from inside code that runs in
  EngineCore — `vllm_neuron/vllm/platform.py` class methods do, and are a cheap
  injection point for a temporary probe.
- Or set `VLLM_ENABLE_V1_MULTIPROCESSING=0` if this vLLM honours it, which
  collapses EngineCore into the parent and makes one stack dump cover everything.
  That is the fastest experiment left to try.

Compare against a Qwen3-VL run at the same point: there the scheduler warning is
followed immediately by request activity, so whatever EngineCore does between
scheduler construction and the ready message is the whole search space.

## Earlier (superseded) diagnosis: EngineCore never signals ready

**Corrected diagnosis.** An earlier note in this file said the *request* hangs.
A `faulthandler` stack dump of the parent process shows otherwise: it is still
inside `LLM.__init__`, blocked in `zmq.poll` under
`vllm/v1/engine/utils.py::wait_for_engine_startup`. No request has been submitted
at all.

    smoke_test.py:50 in main            <- still constructing LLM(...)
    vllm/entrypoints/llm.py:381
    vllm/v1/engine/llm_engine.py:170 / 104
    vllm/v1/engine/core_client.py:723 / 535
    vllm/v1/engine/utils.py:1128 launch_core_engines
    vllm/v1/engine/utils.py:1169 wait_for_engine_startup
    zmq/sugar/poll.py                   <- waiting for the ready handshake

Meanwhile EngineCore **is alive** (0.3% CPU) and has already logged
`init engine (profile, create kv cache, warmup model) took 46.92 s`, followed by
the `NeuronAsyncScheduler` warning — and then nothing. So weights, compilation and
warmup all succeeded; what never happens is the ready message back to the parent.

The four workers poll at ~11% waiting for work, which is consistent: they are
fine, nobody has asked them to do anything.

**Next step.** Instrument EngineCore between scheduler construction and the ready
handshake, since that is the only remaining window:

1. `faulthandler.dump_traceback_later` inside the EngineCore process (the parent's
   dump does not cover it) to see what it is blocked on after the scheduler
   warning.
2. Suspect anything the ready payload needs from the model — `get_kv_spec()` and
   the KV-cache config exchange are the obvious candidates, and both are
   model-supplied code paths that are still unexecuted elsewhere.
3. Compare against a Qwen3-VL run: in that log the scheduler warning is
   immediately followed by request activity, so a side-by-side of the two
   EngineCore processes at that point should isolate the difference quickly.

**This is a much tighter target than "the request hangs"** — the failure is in
engine bring-up, after warmup, before the first request.

**Everything else in this file still holds**, and the eight fixes above are real:
each was found by this same run failing earlier and louder.

### Original plan for this step

**On-device TP=4 smoke test** validates everything not yet executed:
`text_model.py`, `model.py`, the encoder-cache write path, and the vision
encoder's TP sharding.

    llm = LLM(model="/mnt/nvme/models/InternVL3-8B-Instruct",
              tensor_parallel_size=4, max_model_len=4096,
              max_num_batched_tokens=2048, max_num_seqs=1,
              additional_config={...})   # see BENCHMARK_REPORT.md on the other
                                        # branch for the neuron_config shape

**Read the generated text before any latency number.** A wrong TP shard, a missed
LayerScale or a mis-ordered pixel shuffle all produce plausible timings with
garbage output — that failure mode cost real time on the Qwen3-VL branch.

Expect to iterate on: `num_vision_tokens_buckets` (tile count driven, 256 embed
tokens per tile), `encoder_cache_num_blocks` (see the block-vs-token hazard
below), and `max_model_len` (size it to the actual prompt; oversizing it cost 3x
TPOT on Qwen3-VL).

### Bugs the first on-device run already found and fixed

Each of these failed at engine init or graph trace, i.e. long before any output
could be inspected:

1. **`vocab_size` 151674 is not divisible by TP=4.** `ColumnParallelLinear`
   asserts divisibility (`nn/cpl.py` still has "TODO: Add flag to enable
   padding"), and every other model in this plugin happens to have a divisible
   vocab. The LM head is now rounded up to `ceil(vocab/ws)*ws` with the padded
   tail columns forced to `-inf` — zero-weight columns score 0, which can beat
   genuinely negative logits and emit an out-of-vocab token id.
2. **`VocabDimShardedEmbedding` already attaches its own loader** with
   `pad_shard=True`, which this model needs for the same reason. Overriding it
   with a plain loader made rank 3's slice run past the end of the tensor;
   `strict=False` then silently left `embed_tokens.weight` on the meta device.
   `_assert_no_meta_params()` now names such parameters instead of letting torch
   raise a nameless "Cannot copy out of meta tensor" from `model.to(device)`.
3. **The vision tower must load with the *vision* rank.** Folding its mapping
   into the top-level `load_weights` fed it the text rank (0..3) while the vision
   TP group is size 1, so slices ran off the end and produced short shards.
   `InternVisionModel.load_weights` now does its own rank-correct load, matching
   what qwen3_vl does.
4. **The shared `fused_qkv_weight_loader` asserts 2-D slices**, so Qwen2's 1-D
   q/k/v biases need `fused_qkv_bias_loader` (added here).
5. **The projector's checkpoint tensors are HF `[out, in]`** while the params are
   `[in, out]`; it is replicated, so it needs a transpose-only loader.
6. **The QKV kernels require bias shaped `[1, I]`**, not `[I]` — both
   `NF.qkv_proj(bias=)` and `NF.attention_decode(bias_qkv=)`.
7. **SP all-gather at the entry of attention *and* MLP.** Prefill runs both over
   the full sequence and reduce-scatters at the end, so each has to undo
   `embed_tokens`' scatter first. Without it the residual add sees `T/ws` vs
   `T/ws**2`. Note this also means cos/sin must cover the **full** sequence, not
   the rank's slice — an earlier attempt "fixed" the length mismatch by slicing
   positions, which was the wrong direction.
8. **`num_vision_tokens_buckets` gates schedulability.** The runner derives
   `max_vision_blocks_per_request = ceil(bucket / merge_factor / cache_block_size)`
   and a request needing more blocks can never be scheduled — the scheduler spins
   with **no log output at all**, which is indistinguishable from a hang. Size the
   bucket for the worst-case tile count (max_dynamic_patch + thumbnail = 13 tiles
   -> 13312 raw patches), not for the test image.

### Known-unverified specifics to check first if it misbehaves

- `Qwen2Attention.forward_prefill` passes `bias=` to `NF.qkv_proj` and
  `forward_decode` passes `bias_qkv=` to `NF.attention_decode`. Both kwargs exist,
  but neither call has been executed.
- `embed_multimodal` writes into `encoder_cache.buffer[block_id, :n_rows]`
  directly. `allocate(mm_hash, tokens_per_block)` takes a **per-block token list**,
  not a count — an earlier draft got that wrong. Compare against
  `qwen3_vl/model_bf16.py:1051`, which instead has the vision NEFF scatter-write
  into the buffer; the direct write here is simpler but unproven on device.
- `merge_vision_embeddings` is reused from the qwen3_vl utils. It should infer
  zero deepstack levels from `fat_dim == visual_dim`; confirm it returns
  `(hidden, None)` rather than raising.

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
