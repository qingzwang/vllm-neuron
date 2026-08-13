# InternVL3-8B onboarding — state and remaining work

Branch `model/InternVL3-8B`, cut from `release-0.21.0.1.0.0`. No overlap with
`benchmark/Qwen3-VL-8B`.

## Status: working on device at TP=4

`examples/vllm_neuron/models/internvl/smoke_test.py` runs end to end on the
default (multiprocess) path and generates correct text for both tile counts:

    [1 tile, flat blue]
      'The image is a solid blue color with a uniform texture. There are no
       discernible objects, patterns, or variations in color. ...'
    [3 tiles (2x1 + thumbnail), red left / blue right]
      'The left half of the image is red, and the right half is blue.'

The second case is the one that matters for correctness: it covers dynamic
tiling, multiple cache blocks per item, padding up to the compiled bucket, and —
because the halves are different colours — **tile order**, which is easy to get
wrong in a way that still produces fluent output.

## Done

| Piece | File | Status |
|---|---|---|
| Configs | `config.py` | used by the verified pieces below |
| InternViT-300M tower | `vision_encoder.py` | **verified vs HF**, rel 8.0e-6 |
| pixel shuffle + projector | `projector.py` | **verified vs HF**, rel 2.5e-6 |
| Qwen2 decoder layers | `text_model.py` | **executed on device**, output correct |
| Factory | `factory.py` | logic verified against the real config |
| Top-level model | `model.py` | **executed on device**, 1 and 3 tiles |
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

## Solved: the bring-up stall was an OpenMP-after-fork deadlock

For a long time this looked like a lost ready handshake: engine init logged
completion, EngineCore went quiet right after the `NeuronAsyncScheduler` warning
and never sent the ready message, the parent sat in `zmq.poll` under
`wait_for_engine_startup`, and EngineCore itself sat at 0.2% CPU — alive, blocked,
not spinning. **It was none of that.** It was blocked *before* the handshake, in
the scheduler constructor, on a `permute().contiguous()`.

    torchvision/transforms/functional.py:174 in to_tensor        <- blocked here
    vllm/transformers_utils/processors/internvl.py:187 image_to_pixel_values_internvl
    vllm/multimodal/.../processor.py  apply
    vllm/multimodal/encoder_budget.py:87  MultiModalBudget.__init__
    vllm/v1/core/sched/scheduler.py:187   Scheduler.__init__
    vllm_neuron/vllm/core/scheduler.py    NeuronAsyncScheduler.__init__

The chain, which is worth understanding because it is one image processor away
from biting any multimodal model on this plugin:

1. EngineCore is created with `fork()`. Torch's intra-op thread pool does not
   survive a fork — the child still reports the parent's thread count but has no
   live worker threads, so the first CPU op large enough to reach
   `at::parallel_for` blocks forever on a barrier. Repro, no vLLM involved:

       work()                 # any CPU op over ~32k elements, in the parent
       if os.fork() == 0:
           work()             # hangs forever in the child

   Only `set_num_threads(1)` escapes it; setting the count back to N
   re-deadlocks, so the pool genuinely cannot be rebuilt post-fork.
2. Constructing the base `Scheduler` builds a `MultiModalBudget`, which runs the
   HF processor over a worst-case dummy image. vLLM knows this hangs and wraps it
   in `set_default_torch_num_threads()` — the comment there is literally
   "Avoid hang during startup".
3. **That guard resolves its thread count from `OMP_NUM_THREADS`**, and
   `NeuronPlatform.check_and_update_config` raises that variable to
   min(8, cpu_count) for multimodal models so frontend patchify parallelizes.
   EngineCore inherits the raised value across the fork, so upstream's guard
   becomes a no-op.
4. Whether a model trips it comes down to its image processor. Qwen3-VL's is
   numpy-only and slips through. InternVL's goes through torchvision, whose
   `to_tensor` does a `permute().contiguous()` over 448*448*3 elements — just
   past the `parallel_for` grain size.

Fix: `_pin_engine_core_cpu_threads()` in `vllm_neuron/vllm/core/scheduler.py`,
called before `super().__init__()`. It sets the **environment variable** back to
1 for this process; a bare `torch.set_num_threads(1)` does not work, because
step 2's guard overwrites it straight back to 8. The frontend keeps its raised
value, so the patchify optimisation the bump exists for is unaffected. Verified:
Qwen3-VL still starts and generates correctly with this change in place.

### Why this took so long, and the tooling that ended it

Every stack dump of the parent showed `zmq.poll` in `wait_for_engine_startup`,
which reads as "EngineCore never replied" and points away from EngineCore's own
work. `kill -ABRT` on the EngineCore pid produced the parent's stack. And a
`faulthandler` timer armed at import time in the parent never fires in EngineCore:
the fork does not carry the timer thread over, and the module is already in
`sys.modules` so it is not re-imported.

Two things that do work, both kept:

- `VLLM_NEURON_DEBUG_STACK_SECONDS=<n>` — arms a repeating `faulthandler` dump
  from inside `NeuronScheduler.__init__`, which is plugin code that reliably runs
  *in* EngineCore. This is what produced the stack above.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` — collapses EngineCore into the parent.
  Bypasses the fork entirely (so it hides this class of bug), but it turns worker
  errors into a plain traceback in the foreground, which is how the CPU/device
  mismatch below was found in one run instead of several.

### The other bug the same session found

With multiprocessing off, the request actually ran and failed loudly:

    Unhandled FakeTensor Device Propagation for aten.mm.default,
    found two different devices cpu, neuron:0
      ... vision_encoder.py:106 in forward
          patches = torch.matmul(x, self.proj_weight) + self.proj_bias

The runner hands multimodal kwargs over **on CPU** — they come straight off the
mm processor. Vision warmup passed because `build_vision_synthetic_inputs` builds
its tensors on device, so only real requests hit it. `embed_multimodal` now moves
`pixel_values_flat` to the visual tower's device, same as qwen3_vl does.

**Everything else in this file still holds**, and the earlier fixes are real:
each was found by this same run failing earlier and louder.

### Measured and optimised — see `examples/vllm_neuron/models/internvl/BENCHMARK_REPORT.md`

Latency is measured, correctness is checked against HF on real photos, and the
vision encoder has been through one round of optimisation. Headlines:

- 13 tiles, batch 1: TTFT **4733 -> 562 ms**; the vision encoder alone
  **4546 -> 375 ms (12.1x)**. Batch 8 throughput 0.60 -> 1.07 req/s.
- The win is `NF.flash_attention` in `InternVisionAttention`. The naive
  `softmax(q@k^T)@v` was correct but materialised `[tiles, heads, s, s]` and was
  measured **quadratic in the per-rank tile count** even though the FLOPs are
  linear. It is linear now.
- Two traps in that port: s = 1 + grid_size**2 = **1025 is never a multiple of
  128**, so the tower must pad (the kernel reads bounds in whole 128-tiles and an
  unaligned s bakes an out-of-bounds DMA into the NEFF that the runtime rejects at
  load); and the bounds here exist **only** to mask that padding, unlike qwen3_vl
  where they also do frame packing.
- **Keep vision tp=1 / dp=4** (the default). tp=4/dp=1 is 4.2x *slower* — the
  per-rank s x s tensor barely shrinks while every layer gains two all-reduces —
  and tp=2/dp=2 hangs the vision graph on device with
  `FATAL-RT-UNDEFINED-STATE`.
- TPOT is 12.2-12.9 ms at batch 1 **regardless of image size**. At batch > 1 the
  reported TPOT includes prefill interference, not just decode: it is
  `(e2e - ttft) / (n-1)`, and one request's decode is interrupted by the others'
  prefills.

**Read the generated text before any latency number.** A wrong TP shard, a missed
LayerScale or a mis-ordered pixel shuffle all produce plausible timings with
garbage output — that failure mode cost real time on the Qwen3-VL branch.
`compare_vs_hf.py` now automates that check against the HF reference.

Still open: vision cost is still slightly superlinear at 4 tiles/rank (4.48x vs
4x); the vision bucket is sized for one image so batched prefills serialise; and
`encoder_cache_num_blocks` has not been stressed with many distinct images.

### Bugs the first on-device runs found and fixed

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

### Specifics now confirmed on device

These were the open questions before the model ran; all of them held.

- `Qwen2Attention.forward_prefill` passes `bias=` to `NF.qkv_proj` and
  `forward_decode` passes `bias_qkv=` to `NF.attention_decode`. Both execute, and
  both want the bias as `[1, I]` rather than `[I]`.
- `allocate(mm_hash, tokens_per_block)` takes a **per-block token list**, not a
  count. `embed_multimodal` no longer writes into the buffer itself: like
  `qwen3_vl/model_bf16.py:1051`, the vision NEFF scatter-writes with `index_put_`
  in-graph, and `embed_multimodal` only allocates, pads to the bucket and routes
  padding blocks to the scratch block.
- `merge_vision_embeddings`, reused from the qwen3_vl utils, does infer zero
  deepstack levels from `fat_dim == visual_dim` and returns `(hidden, None)`.

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
