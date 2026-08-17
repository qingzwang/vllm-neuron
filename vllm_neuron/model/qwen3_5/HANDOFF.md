# Qwen3.5-2B onboarding — findings, decisions, plan

Branch `model/Qwen3.5-2B`, cut from `release-0.21.0.1.0.0`. No overlap with
`model/InternVL3-8B` or `benchmark/Qwen3-VL-8B`.

Checkpoint: `/mnt/nvme/models/Qwen3.5-2B` (4.3 GB, downloaded).
NxDI reference: `git clone --depth 1 -b qwen3.5-2b-hybrid-deltanet
https://github.com/qingzwang/neuronx-distributed-inference` →
`contrib/models/Qwen3.5-2B` (cloned to `/tmp/nxdi_ref` during bring-up; re-clone
as needed, it is not vendored here).

## What makes this model different from everything else in this plugin

**18 of its 24 layers are not attention layers.** `text_config.layer_types` is
`[linear, linear, linear, full] x 6`, so full attention lives only at layer
indices **3, 7, 11, 15, 19, 23**. The other 18 are recurrent gated-DeltaNet
blocks: a depthwise conv over projected q/k/v plus a delta-rule state update.

That one fact drives the whole port. A DeltaNet layer keeps a **fixed-size**
state regardless of sequence length, so it has no paged KV cache, no block
table, and no notion of "context length" — while the 6 attention layers have all
three. Every piece of this plugin's machinery that assumes "layer == attention
layer with a KV cache" has to learn about a second kind of layer.

## Architecture (read from the real config.json, not the reference README)

| | value |
|---|---|
| Text | 24 layers, hidden 2048, intermediate 6144, vocab 248320 |
| Full attention (6 layers) | 8 Q heads, 2 KV heads, **head_dim 256**, `attn_output_gate: true` |
| Rotary | **partial**, factor 0.25 → rotary_dim **64** of 256; mRoPE interleaved, section `[11, 11, 10]`, theta 1e7 |
| DeltaNet (18 layers) | 16 key heads + 16 value heads, head_dim 128, conv kernel 4, state accumulated in **float32** (`mamba_ssm_dtype`) |
| DeltaNet per-layer weights | `A_log`, `dt_bias`, `conv1d.weight`, `in_proj_{qkv,a,b,z}`, `norm`, `out_proj` |
| Derived | `conv_dim` 6144 (= 2·16·128 + 16·128), conv state `[6144, 3]`, recurrent state `[16, 128, 128]` |
| Per-seq state | 18 × (1.05 MB fp32 recurrent + 0.04 MB conv) ≈ **19.6 MB** at TP=1 |
| Vision | ViT 24 layers, hidden 1024 → out 2048, patch 16, spatial_merge 2, temporal_patch 2, 16 heads, **`deepstack_visual_indexes: []`** |
| Weights | `tie_word_embeddings: true` and **no `lm_head` in the checkpoint** — confirmed by reading the safetensors index |
| Extra | An **MTP head** is present (`mtp.*`, 1 layer, its own `self_attn` + `mlp` + `fc` + norms) for multi-token prediction. Not needed for bring-up. |

Checkpoint prefixes: `model.language_model.layers.{i}.*`,
`model.language_model.{embed_tokens,norm}`, `model.visual.*`, `mtp.*`.

`config.py` here re-derives all of this from the HF config and raises on every
variant that has not been validated (non-default rope_type, non-interleaved
mRoPE, `mlp_only_layers`, `attn_output_gate=False`, deepstack). Verified against
the real checkpoint — see the numbers in the table above, which were printed from
it rather than copied from the README.

## Environment: better than the reference's, in one important way

The NxDI reference README works around `transformers==4.57.6` not knowing the
`qwen3_5` architecture (it builds a separate venv just to run the HF oracle).
**That problem does not exist here:**

| | reference (NxDI DLAMI) | here (vLLM DLAMI) |
|---|---|---|
| transformers | 4.57.6 (no `qwen3_5`) | **5.14.1 — `qwen3_5` is native** |
| HF oracle | needs a separate venv | runs in the same venv |

Also already present and directly useful:

- vLLM 0.21's registry has **`Qwen3_5ForConditionalGeneration` → `('qwen3_5', ...)`**,
  so the frontend supplies the config and the multimodal processor for free.
  Registration happens only in the worker process, exactly as for InternVL, so
  **only execution needs replacing**.
- vLLM ships **`MambaSpec`** in `v1/kv_cache_interface.py` and
  **`Qwen3NextForCausalLM`** — the gated-DeltaNet sibling. That is a second
  reference for the DeltaNet math, in vLLM's own idiom.

## Decisions taken (with the user, 2026-08-13)

1. **Text-only first**, then VL. The reference did the same, and VL additionally
   needs the numerically-unstable-kernel workaround below.
2. **Integrate with vLLM's `MambaSpec`** rather than keeping private static state
   buffers. The reference takes the static route
   (`use_hybrid_cache_manager=False`); we deliberately do not, so prefix caching
   and preemption stay correct instead of silently corrupting state. Cost: this
   touches shared plugin code (`LayerSpec`/`KVSpec`, the runner's
   `get_kv_cache_spec`, cache allocation), which the static route would avoid.

## The infrastructure gap this implies

`vllm_neuron/model/kv_cache.py` has only `LayerSpec` (name, num_kv_heads,
head_size, dtype, sliding_window_size, chunk_size) and `KVSpec`. The runner's
`get_kv_cache_spec` (`neuron_model_runner.py:7815`) walks
`model.get_kv_spec().layers` and emits `FullAttentionSpec` or
`SlidingWindowSpec` — nothing else. Confirmed by grep: the plugin has **no**
`MambaSpec`, no `conv_state`, no `recurrent_state`, no DeltaNet anywhere. (The
`recurrent_state` hits in `llama3/eagle3_model.py` are the speculative drafter's
hidden state, unrelated. `mamba_ssm_size` in the NIXL connector is the only
mention of mamba at all.)

So the work splits into:

1. **Spec plumbing** — extend `LayerSpec`/`KVSpec` so a layer can describe
   recurrent state (shapes + dtypes) instead of KV geometry, and add the
   `MambaSpec` branch to `get_kv_cache_spec`. Then follow where
   `initialize_kv_cache` / `initialize_from_config` puts the buffers.
2. **The model** — 18 DeltaNet layers, 6 gated-GQA layers, partial interleaved
   mRoPE, tied embeddings.
3. **Kernels** — port the DeltaNet NKI kernels from the reference (see map
   below).
4. **Bring-up at TP=4** and numerical validation against HF.

## Constraints specific to this box

- **4 NeuronCores** (trn2.3xlarge, LNC=2), so **TP=4 is the ceiling** — the
  reference validated TP=8 on a trn2.48xlarge. It does publish TP=4 text numbers
  (TTFT 42.2 ms, TPOT 4.75 ms, 210 tok/s at seq_len 1024) which is the number to
  beat/compare.
- **2 KV heads < TP=4**, so the attention layers need KV replication
  (`num_kv_replicas`, as used by the InternVL loaders). The 16 DeltaNet key/value
  heads divide by 4 cleanly.
- **`head_dim` is 256 but the plugin's attention kernel caps at 128.**
  `functional/attention/attention_cte.py` sets `MAX_HEAD_DIM = 128` and
  `_can_use_flash_attention_kernel` returns False above it, so the 6 full
  attention layers will fall back to torch. On InternVL, materialising the
  `s x s` attention matrix cost 12.1x on the vision encoder — so verify this
  early and expect it to set the perf ceiling. Only 6 of 24 layers are affected.
- Per-sequence DeltaNet state is small (19.6 MB at TP=1, ~4.9 MB/rank at TP=4),
  so state memory is not a concern at this scale.

## Reference code map (`/tmp/nxdi_ref/contrib/models/Qwen3.5-2B`)

| file | lines | what to take |
|---|---|---|
| `src/modeling_qwen35.py` | 8148 | text decoder: DeltaNet + GQA + mRoPE. The primary reference. |
| `src/nki_kernels/nki_deltanet_fused.py` | 2991 | fused chunked forward — the reference's default CTE path |
| `src/nki_kernels/nki_deltanet_fused_legacy.py` | 613 | **the one VL must use** (see hazard below) |
| `src/nki_kernels/nki_deltanet_chunked.py` | 431 | per-chunk step |
| `src/nki_kernels/nki_deltanet.py` | 607 | recurrent step / recurrent forward |
| `src/nki_kernels/qwen_qk_norm_rope.py` | 230 | optional fused QK-norm + RoPE |
| `src/modeling_qwen35_vision.py` | 985 | ViT wrapper (VL stage) |
| `src/modeling_qwen35_vl.py` | 701 | VL orchestrator (VL stage) |
| `src/hybrid_apc.py` | 1798 | APC — disabled in the reference, skip |

The reference states its modeling code came verbatim from
[NxDI PR #173](https://github.com/aws-neuron/neuronx-distributed-inference/pull/173)
(the 27B sibling); only tied-weight loading and config validation were adapted
for 2B.

## Hazards the reference already paid for

- **The default fused multihead DeltaNet NKI kernel is numerically unstable on
  real vision embeddings** — it produces degenerate repeated tokens. The
  reference forces `QWEN36_DELTANET_CTE_IMPL=legacy_direct` and
  `QWEN36_DELTANET_MULTIHEAD_CTE=0` for VL. Text-only is fine on the fused
  kernel. Root cause unknown: the reference notes random vectors at the same std
  (0.17) decode cleanly, so it is a specific numeric interaction with structured
  signal, not magnitude. **For the VL stage here, start from the legacy kernel.**
- **Only batch size 1 is validated** in the reference. State buffers are indexed
  by `seq_ids` for continuous batching but that path was never exercised.
- The reference's own accuracy check against HF is **53/80 tokens (66%) greedy
  match, 3/5 prompts exact**. Treat that as the bar to meet, not a target to
  beat — and note the InternVL experience: a synonym-level divergence after a
  matching prefix is normal for bf16 vs an independent implementation, whereas
  divergence in the first few tokens is a bug.

## Spec plumbing: done, and most of it was free

vLLM already does the hard part. Measured on this box with the real checkpoint at
TP=4:

    is_hybrid: True                  <- derived from the HF config
    mamba_block_size: 1024           <- set on the Neuron path (= max_model_len,
                                        i.e. one block per sequence)
    mamba_page_size_padded: None
    mamba_cache_mode: 'none'

`ModelConfig.is_hybrid` comes from the HF config, and
`Platform._align_hybrid_block_size` fills in `mamba_block_size` /
`mamba_page_size_padded` before the runner is asked for specs. **Do not recompute
either** — the planner has already sized the pages from those values.

What was added here:

- `RecurrentLayerSpec` (name, shapes, dtypes) and
  `KVSpec.recurrent_layers` in `vllm_neuron/model/kv_cache.py`. Purely additive
  with a default of `[]`, so the five existing models that build
  `KVSpec(layers=...)` are untouched.
- A `MambaSpec` branch in the runner's `get_kv_cache_spec`, reading block size and
  padding from `cache_config`.
- A `MambaSpec` branch in the runner's `initialize_kv_cache`, striding each state
  tensor out of the layer's raw page-aligned buffer (block dim strides by a whole
  page; successive states sit at increasing offsets within a page). Copied from
  vLLM's own `gpu_model_runner` branch on purpose.

Verified end to end: 6 attention + 18 recurrent layers, per-rank state
`conv (3, 1536) bf16` and `recurrent (4, 128, 128) fp32`, 0.271 MB per recurrent
layer per sequence per rank → **4.88 MB per sequence per rank** for all 18.

### The mistake worth not repeating

The first version of `config.py` derived the state shapes by hand and got the conv
state **transposed and unsharded**: `[conv_dim, kernel-1]` = `(6144, 3)` where vLLM
uses `[kernel-1, conv_dim/tp]` = `(3, 1536)`. Because vLLM sizes the state *pages*
from `MambaStateShapeCalculator` via
`model_cls.get_mamba_state_shape_from_config`, a divergent layout would not raise
— it would alias memory. `Qwen3_5TextConfig.state_shapes(tp_size)` now delegates
to that same calculator, which is the only way to be sure the two agree.

Also note vLLM ships a full `vllm/model_executor/models/qwen3_5.py` implementing
both `get_mamba_state_{shape,dtype}_from_config`. It is a second reference for the
DeltaNet and gated-attention math, in vLLM's idiom, and unlike the NxDI reference
it is guaranteed consistent with the cache layout above. NxDI remains the
reference for the **NKI kernels**, which vLLM has no equivalent of.

## The model: implemented in torch, validated against HF on CPU

`transformers` 5.14.1 ships `modeling_qwen3_5.py`, and it is a *better* oracle
than the NxDI reference for everything except the NKI kernels: it has
`torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule` as plain
functions, so the delta rule can be compared piece by piece rather than
end-to-end. Both layer kinds were built against it first, in float32 on CPU, so
that implementation error was ruled out before compilation ever entered the
picture. Two check scripts under
`examples/vllm_neuron/models/qwen3_5/` are the record of that:

| script | covers |
|---|---|
| `check_deltanet_vs_hf.py` | our kernels vs HF's; module prefill vs HF's module, padded and not; decode replayed token-by-token vs prefill incl. both state carries; TP=4 vs TP=1 through the real loaders |
| `check_attention_vs_hf.py` | zero-centred RMSNorm; partial interleaved mRoPE; attention prefill vs HF's module; MLP; decode replay vs prefill (paged KV scatter/gather); TP=4 vs TP=1 |

Everything agrees to <= 1e-6 relative, and most of the attention pieces are
bit-exact. Run both after any change to either layer — they are cheap (seconds,
CPU) and they catch the whole class of "fluent but wrong" bug this model is
prone to.

**Run them with `PYTHONPATH=/mnt/nvme/vllm-neuron`.** The venv's editable
install of `vllm_neuron` resolves through a static module map, so subpackages
added after install time (`vllm_neuron.model.qwen3_5`) are invisible without it.

### Things that would have silently produced wrong output

- **`Qwen3_5RMSNorm` scales by `1 + weight`,** and the checkpoint's norm tensors
  are distributed around 0 to match. The usual `weight * x` would multiply
  activations by roughly zero. Applies to `input_layernorm`,
  `post_attention_layernorm`, `q_norm`, `k_norm` and the final `norm` — but
  **not** to the DeltaNet's output norm, which is HF's `Qwen3_5RMSNormGated` and
  uses plain `weight`. Two different norms in one model.
- **`q_proj` emits `[query | gate]` per head** (twice the query width) and the
  attention output is scaled by `sigmoid(gate)` before `o_proj`. Because each
  head's contribution is a contiguous `2 * head_dim` block, sharding q_proj's
  output dim by whole heads keeps query and gate together.
- **Rotary is partial**: only the first 64 of each head's 256 dims rotate. The
  remaining 192 pass through. mRoPE sections `[11, 11, 10]` sum to
  `rotary_dim / 2`, not `head_dim / 2`.
- **The prefill padding mask comes from `positions`, not from new metadata.**
  The runner pads a prefill by appending token id 0 with `positions` frozen at
  the last real value, so a token is real iff
  `positions[i] - positions[0] == i`. Zeroing `g` and `beta` at the pads makes
  them exact no-ops (decay `exp(0) == 1`, delta 0), so the state written out is
  the state after the last real token. Verified: a short prompt padded to a
  bucket reproduces HF on the unpadded prompt.

### The UT-transform inverse: do not use the Neumann series

The in-chunk `(I - A)^-1` (HF's `attn` loop) looks like an easy win: `A` is
strictly lower triangular, so `sum_j A^j` terminates and could be summed by
repeated squaring as `prod (I + A^(2^i))` — 10 matmuls instead of 63 dependent
slice-assignments, and slice-assignment-into-a-shared-tensor is exactly the
pattern the NxDI reference flags as hitting neuronx-cc codegen failures.

**It does not work.** With this checkpoint's weights `|A^16|` reaches 2.7e6
while the true inverse has entries of magnitude 1, so the product loses every
significant digit: measured 0.57 absolute error in float32, against 1.2e-7 for
elimination. Symptom at the module level was a 8e-4 relative output error that
looked like plausible rounding.

`deltanet.py` uses the **blocked** form of the same elimination instead: keep
`T` block diagonal and double the block width, so `T_L A_LU T_U` is a masked
quadrant of `T @ A @ T`. Two matmuls per round, `log2(64) = 6` rounds, every
intermediate a true inverse of a sub-problem so nothing grows. This is worth
remembering when the NKI port lands: the kernel has to be stable in this same
way, and a "clever" series expansion will pass a random-input test and fail on
real weights.

### Deliberately deferred

- **No NKI kernels yet.** The delta rule is torch. The chunked prefill is
  matmul-heavy so it should map reasonably, but this is the first thing to
  measure. `nki_deltanet.py` (recurrent step) is the simplest starting point;
  the fused multihead kernel is the one that breaks on vision embeddings later.
- **Attention runs in torch too**, because `head_dim` 256 exceeds the flash
  kernel's `MAX_HEAD_DIM` of 128 and the decode megakernel fuses a QKV
  projection and full-width RoPE that do not match this layer's gate and partial
  rotary. Only 6 of 24 layers.
- **Padded decode rows** read and write the *last* state block, chosen because
  `slot_mapping` is `-1` there and that is where `index_put_`'s negative-index
  wrap already sends the attention path's padded writes. Verify at bring-up that
  the last mamba block is not also handed to a live sequence.
- **No prefix caching / chunked prefill.** Prefill starts from a zero state,
  which matches what the attention path already assumes (its prefill is plain
  causal flash attention with no prefix).
- **Speculative decode** raises: the DeltaNet decode path wants one token per
  request, and multi-token decode needs the recurrence stepped per draft token.

## Bring-up: it runs, and the output is wrong

`examples/vllm_neuron/models/qwen3_5/run.py` boots, compiles, warms up both KV
cache groups on all 4 ranks and generates, exit 0. **Generation is wrong from the
first token** — "The capital of France is" → " most". By the rule below, first-token
divergence is a bug, not bf16 noise.

Five plumbing fixes were needed to get that far; the first two are the
interesting ones and both are corrections to what an earlier version of this
document claimed.

1. **vLLM's hybrid page-size alignment never runs on the Neuron path.**
   `Platform._align_hybrid_block_size` grows the attention block size until its
   page is at least the state page, then pads the state page to match — but vLLM
   gates it on `_find_non_ssm_backend` finding one of *its own* attention
   backends, and this platform registers none. Without it startup dies in
   `unify_kv_cache_spec_page_size`, because a DeltaNet state page is 271360
   bytes per rank and that factors as `1024 * 5 * 53`, so no sane block size
   divides it. `Platform.update_block_size_for_backend` now calls vLLM's helper
   directly with a stub backend supplying the one method it reads. Effect at
   TP=4: **block_size 32 → 288**, `mamba_page_size_padded` None → 294912, 8.68%
   state padding. Call vLLM's implementation rather than copying the arithmetic:
   the same helper sizes the pages the planner allocates.
2. **Neuron device tensors reject vLLM's state layout.** Its `gpu_model_runner`
   strides each state tensor so the block dim steps by a whole page; the runtime
   answers "Detected non-contiguous slicing for requested Device Tensor". The
   runner now lays the states out one after another so each is contiguous. Sound
   only because nothing outside the model reads them — the page size is
   unchanged so the planner's accounting holds, and the per-block state-copy
   hooks are inactive at `mamba_cache_mode == "none"`.
3. Vision bucket resolution ran for a text-only launch of a vision-capable
   checkpoint and then failed validating a 2048 vision block against a 1024 text
   `max_model_len`. Now skipped when every `limit_mm_per_prompt` count is 0. Note
   those counts are `BaseDummyOptions` objects, not ints.
4. vLLM derives `uses_mrope` from the HF config and then *requires* the
   `SupportsMRoPE` protocol, so even a text-only port must implement it.
5. The runner passes `vision_embedding_blocks`/`vision_positions` to any model
   whose HF config has a `vision_config`. Accepted and ignored.

### What is ruled out, with evidence

- **Both layer kinds.** Bit-exact or ~1e-7 against HF on CPU in float32,
  including TP=4 through the real weight loaders. See the check scripts above.
- **Weight loading completeness.** All 320 referenced checkpoint keys exist, and
  `load_weights` now raises if any parameter got no tensor — it does not fire.
  This mattered to check: the loader must be `strict=False` (it does not know
  about buffers like rotary `inv_freq`), so a wrong mapping key would leave a
  parameter at its uninitialised `torch.empty` value and produce exactly this
  symptom with no error anywhere.
- **Compiler downcasting.** The runner passes `--auto-cast=none`, so the float32
  delta rule stays float32. This mattered because the in-chunk matrix has a
  dynamic range that bf16 could not survive (see the Neumann note above).
- **Packed prefills.** The DeltaNet padding mask and the attention causal mask
  both assume one sequence per prefill forward. That assumption is safe: the
  Qwen3-VL decoder next door makes it too (plain causal `NF.flash_attention`,
  no segmentation) and works.

### Localised: the spine is correct on device, a mixer is not

Bisected with the ablation switch in `model.py`
(`VLLM_NEURON_QWEN35_ABLATE_MIXERS`), comparing the *same reduced model* between
device and CPU. Output text is meaningless under ablation; only the agreement
matters. CPU references come from a short driver over
`check_model_vs_hf.build_ours`.

| experiment | result |
|---|---|
| SP disabled (`VLLM_NEURON_QWEN35_DISABLE_SP=1`) | **same first tokens** as SP on -> the collectives are not the bug |
| depthwise `F.conv1d` -> unrolled taps, `cumsum` -> matmul | output unchanged (2 of 4 prompts byte-identical) -> those ops were not wrong |
| all mixers ablated | device **matches CPU**: prompt 1 `'atches'` exactly, prompt 3 `'icals'` = CPU's #2 in a near-tie (bf16 vs fp32) |
| DeltaNet ablated, attention kept | **does not compile** (`neuronx-cc` 70) |
| attention ablated, DeltaNet kept | **does not compile** (`neuronx-cc` 70) |

So embedding, MLP, norms, the vocab-sharded LM head, the sampler and every
collective are **correct on device**, and the fault is inside a mixer. Note only
"all mixers on" and "all mixers off" compile, so the two mixer kinds cannot be
separated by ablation — bisect by *simplifying* a mixer instead, adding pieces
back on top of the known-good ablated model (e.g. keep the DeltaNet's
projections, conv and output stage but set `core = v` to remove the recurrence).

Ranked suspects inside the DeltaNet prefill, which is where the first token is
decided (the state write cannot matter until decode):

1. **`_strictly_lower_inverse`** — 6 dependent rounds of masked batched matmuls
   over `[1, 4, 16, 64, 64]`. Already made its identity operand contiguous
   rather than an `expand_as` broadcast view, untested on device.
2. **The 16-iteration chunk loop** carrying `state` — long dependent chain of
   batched matmuls.
3. **The `is_real` padding mask**, derived from `positions`. Worth noting *no
   other model in this plugin reads `positions` during prefill* (Qwen3-VL uses it
   only in decode), so the runner's prefill `positions` have never been exercised
   before. Print them from inside the graph to confirm they are `0..L-1` then
   frozen.
4. Extreme dynamic range: `g` reaches -4.6 per token, so the in-chunk cumsum
   reaches -297 and `exp` underflows to 0. Fine in float32 (CPU agrees), but
   worth confirming the device really keeps these tensors in float32 rather than
   demoting them despite `--auto-cast=none`.

## Status

- [x] Branch, checkpoint, reference clone
- [x] `config.py`, validated against the real checkpoint
- [x] `LayerSpec`/`KVSpec` + `MambaSpec` plumbing (spec emission + allocation)
- [x] DeltaNet layer, matched to HF in float32 (`deltanet.py`)
- [x] Gated GQA layer with partial interleaved mRoPE (`model.py`)
- [x] Text decoder, factory, registry entry
- [x] Boots, compiles and generates at TP=4
- [x] Spine (embed / MLP / norms / head / sampler / collectives) verified on device
- [ ] **Correct output** — wrong from the first token; localised to a mixer
- [ ] NKI kernel port + latency measurement
- [ ] VL stage
