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
   below). *Done and measured: they lose to torch. See "Two kernels vendored".*
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
  attention layers fall back to torch. This was predicted to set the perf ceiling
  by analogy with InternVL (12.1x on its vision encoder). **Measured, it does
  not** — torch attention is 6-8% of prefill; see "Where prefill time actually
  goes" below. With only 2 query heads per rank at TP=4 the `[2, seq, seq]` score
  matrix is cheap.
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

### Known limits of the shipped configuration

- **Both mixers run in torch, on purpose.** Attention because `head_dim` 256
  exceeds the flash kernel's `MAX_HEAD_DIM` of 128 and the decode megakernel
  fuses a QKV projection and full-width RoPE that do not match this layer's gate
  and partial rotary. The delta rule because the vendored NKI kernels *lose* to
  the batched torch path — see "Two kernels vendored" below, which is the measured
  answer to the "port the kernels" item that this section used to list as
  outstanding.
- **Padded decode rows** read and write the *last* state block, chosen because
  `slot_mapping` is `-1` there and that is where `index_put_`'s negative-index
  wrap already sends the attention path's padded writes. Still worth confirming
  the last mamba block is never also handed to a live sequence — nothing observed,
  but nothing proves it either.
- **No prefix caching / chunked prefill.** Prefill starts from a zero state,
  which matches what the attention path already assumes (its prefill is plain
  causal flash attention with no prefix).
- **Speculative decode** raises: the DeltaNet decode path wants one token per
  request, and multi-token decode needs the recurrence stepped per draft token.
- **One `num_seqs_bucket` per engine.** Not this port's constraint — passing
  several batch-size buckets to one engine hangs on the first request for other
  models on this box too — but it is why the benchmarks below measure each batch
  size in a separate process.

## Bring-up: working at TP=4

`examples/vllm_neuron/models/qwen3_5/run.py` generates correct text:

    'The capital of France is'            -> ' Paris.'
    'I am gonna keep counting forever...' -> ' 6 7 8 9 10 11 12 13 14 15 16 17'
    'def fibonacci(n):'                   -> a correct recursive base-case chain
    'Once upon a time, there was a'       -> ' little boy named Tom. Tom loved...'

Getting there took five plumbing fixes and then two genuine bugs that only exist
on the device. The plumbing first, briefly:

1. **vLLM's hybrid page-size alignment never runs on the Neuron path.**
   `Platform._align_hybrid_block_size` grows the attention block size until its
   page is at least the state page, then pads the state page to match — but vLLM
   gates it on `_find_non_ssm_backend` finding one of *its own* attention
   backends, and this platform registers none. Without it startup dies in
   `unify_kv_cache_spec_page_size`: a DeltaNet state page is 271360 bytes per
   rank, which factors as `1024 * 5 * 53`, so no sane block size divides it.
   `update_block_size_for_backend` now calls vLLM's helper directly with a stub
   backend. Effect at TP=4: **block_size 32 -> 288**, state page padded 8.68%.
2. **Neuron device tensors reject vLLM's state layout** (its `gpu_model_runner`
   strides the block dim by a whole page). The runner now lays the states out
   contiguously one after another. Sound only because nothing outside the model
   reads them and `mamba_cache_mode` is `"none"`.
3. Vision bucket resolution ran for a text-only launch of a vision-capable
   checkpoint; now skipped when every `limit_mm_per_prompt` count is 0. Those
   counts are `BaseDummyOptions` objects, not ints.
4. vLLM derives `uses_mrope` from the HF config and then *requires* the
   `SupportsMRoPE` protocol, so even a text-only port must implement it.
5. `load_weights` asserts every parameter received a tensor — the loader has to
   be `strict=False`, so a wrong mapping key would otherwise leave a parameter at
   its uninitialised `torch.empty` value and produce fluent garbage.

### Probe on the device, do not guess through the engine

An engine run costs ~9 minutes and only ever reports "the whole model is wrong".
Two probes replaced that with seconds-long, op-level answers, and they are the
reason the bugs below were found at all:

| script | what it does |
|---|---|
| `probe_device_ops.py` | Hands the plugin's own `vllm_neuron.compile.backend.compile` to `torch.compile`, puts inputs on `neuron:0`, and diffs one function or module against CPU. Reports `OK` / `WRONG` (compiles but disagrees) / `FAIL` (does not compile). |
| `probe_device_model.py` | The same for a whole `Qwen3_5TextModel`, scalable by `--layers`, `--tp`, `--dtype`, `--head` and `--state-views`, so model-level and scale-dependent codegen problems reproduce cheaply. |

This is the same trick as the CPU checks — compare against a reference you trust
— but with the compiler in the loop. Reach for it first next time.

Two harness gotchas worth knowing: `expand(...).contiguous()` is unsupported on
device tensors (build block tables on CPU, then move), and the rotary module
cannot run eagerly on device (`Expected self.dtype() == dst.dtype()`), so
precompute `cos`/`sin` on CPU.

### Bug 1: `Tensor.split(sizes, dim=-1)` silently miscompiles

The symptom was perfect: per-stage diffing showed the padding mask, the
projections and the post-conv activations all matching to ~1e-6, and then `q`
and `v` wrong by relative error 1.2-1.3, with `beta`/`g` wrong too. The one thing
those had in common was a last-dim `.split()`.

Probed the alternatives directly:

| form | device vs CPU |
|---|---|
| `t.split((512, 512, 512), -1)[1]` | **rel 1.4** |
| `t[..., 512:1024]` | exact |
| `torch.chunk(t, 3, dim=-1)[1]` | exact |
| `t.reshape(N, 3, 512)[:, 1]` | exact |

Both uses were in the DeltaNet — the q/k/v split and the fused `b|a` split — so
all 18 layers were affected. Now plain slices. **Do not use `Tensor.split` in
device code in this plugin.**

### Bug 2: the runner's `--modular-flow-mac-threshold=10` breaks codegen

The decode graph failed with `NCC_IBTN006`: a `pftranspose` whose `Copy` fails
the `start_addr_active_channels` assertion. Isolated *without the engine* by
recompiling the cached HLO by hand — the failed compile leaves its `graph.hlo`
and the exact `command.txt` under `$VLLM_CACHE_ROOT/neuron/compile_cache/<hash>/`
(the failing directory is the one with no `.neff`):

| flags | result |
|---|---|
| as the runner passes them | fails, exit 70 |
| `-O2` instead of `-O1` | compiles |
| `-O1`, no internal options | compiles |
| `-O1` + `--modular-flow-mac-threshold=10` alone | **fails** |
| `-O1` + `--enable-verifier=false` alone | compiles |

The runner passes that threshold unconditionally, with a FIXME saying it is only
needed until NKI kernels report MAC counts — and this model has no NKI kernels.
So `NeuronConfig.hlo2tensorizer_options` now overrides those extra options;
`""` means none, and `run.py` sets it. Default behaviour for every other model is
unchanged. Note this had to be added in **two** places: the dataclass field *and*
`from_dict`'s explicit whitelist, which silently drops unknown keys.

### Two more device-only hazards, same probes

- **Data-dependent `index_select` miscompiles.** The conv-state tail was gathered
  at `L - 3 .. L - 1` with `L` from a device-side `sum`. Replaced by `_tail_rows`,
  which selects by arithmetic: `r[t] - r[t+1]` is a one-hot at the last real
  token, and shifting it builds a static `[count, T]` selector applied with one
  matmul. Every index is a compile-time constant, and prompts shorter than the
  window zero-fill for free.
- **Rank-5 `permute` is what emitted the bad `pftranspose`.** Decode attention
  used to index the cache with the block table and then permute to move the
  KV-head dim in front of the block dim. It now flattens the cache to
  `[blocks * kv_heads, block_size, dim]` and gathers rows by an index computed per
  (request, query head) — dropping both the permute and the `repeat_interleave`
  that expanded KV heads to query heads.

### The rank-3 rewrite, and what it was not

The delta rule is written entirely at rank 3 — `(b, h)` and the chunk index fold
into one batch dim, chunks are taken by `unbind` rather than sliced out of a
rank-5 tensor. That followed the reference's note at `modeling_qwen35.py:3114`
that it had to collapse attention to `(B*H, S, d)` to avoid
`NCC_INLA001: Expected 2D tensor but got 4D AP`.

Worth being honest: this was **not** the bug. The rank-5 form also compiled and
was also wrong, for the `split` reason above. The rewrite is kept because it is
good practice on this compiler and marginally simpler, not because it fixed
anything. The reference's warning that its PyTorch `_chunk_forward` "can hit
neuronx-cc codegen ICE with these DeltaNet dimensions" did not reproduce here
once the real bugs were out of the way — **the pure-torch chunked delta rule
compiles and runs correctly on Neuron.** The NKI port is back to being a
performance item.

### Also not the bug, ruled out with evidence

Sequence parallelism (disabling it changed nothing), the grouped depthwise
`F.conv1d` and `torch.cumsum` (replaced with unrolled taps and a triangular
matmul — exact, kept for simplicity, but the output did not change), compiler
downcasting (`--auto-cast=none` is passed), packed prefills, weight-loading
completeness, and the offset state views.

## Measured: accuracy and latency at TP=4

### Accuracy vs HF (`check_generation_vs_hf.py`)

32 greedy tokens per prompt, device bf16 against HF float32 on CPU:

    prefix 9    9/32    'The capital of France is'
    EXACT      32/32    'I am gonna keep counting forever, 1 2 3 4 5'
    EXACT      32/32    'def fibonacci(n):'
    prefix 6    6/32    'Once upon a time, there was a'
    EXACT      32/32    'The three primary colours are'

    total 111/160 tokens (69.4%), 3/5 prompts exact

Slightly ahead of the reference port's published bar (53/80 = 66%, 3/5 exact).
**No first-token mismatches**, and both divergences are the benign kind — a
matching prefix then a coin flip:

* `' Paris.\nA. True\nB. '` then `True` (ours) vs `False` (HF)
* `' little boy named Tom. Tom '` then `loved to play with his toys` vs
  `was very curious`

That is what bf16 against an independent float32 implementation looks like: the
two agree until a near-tie, and greedy decoding then amplifies the difference. A
mismatch on the *first* token would be a bug instead, which is why the script
fails loudly on that case specifically.

### Latency (`benchmark_latency.py`)

896-token prompt + 128 output tokens (1024 total, the whole `max_model_len`),
median of 3 rounds after a discarded warmup round, `AsyncLLM` streaming so TTFT
is the true time to first token:

| | batch 1 | batch 4 | reference, TP=4 |
|---|---|---|---|
| TTFT | **108.9 ms** | 262 ms mean (109 min / 412 max) | 42.2 ms |
| TPOT | **3.72 ms** | 6.23 ms | 4.75 ms |
| E2E | 581.7 ms | 1053.5 ms | — |
| decode, per stream | 268.7 tok/s | 160.5 tok/s | 210 tok/s |
| decode, aggregate | 268.7 tok/s | **482.6 tok/s** | — |

So **decode is already faster than the reference** — TPOT 3.72 ms against
4.75 ms, 268.7 tok/s against 210 — and **prefill is 2.6x slower**. That split is
exactly the ceiling this document predicted before any of it ran, and it says
where the NKI port would pay:

* The 6 attention layers run in torch because `head_dim` 256 exceeds the flash
  kernel's `MAX_HEAD_DIM` of 128, so prefill materialises a
  `[heads, 1024, 1024]` score matrix per layer. On InternVL the same problem cost
  12.1x on a vision encoder.
* The chunked delta rule is pure torch where the reference uses NKI. Decode does
  not care (a single recurrent step is tiny and memory-bound, which is why decode
  wins), but prefill does all the chunked matmul work.

The batch-4 TTFT spread (109 -> 412 ms) is prefills serialising: one prefill
graph per forward, so request *n* waits for the *n-1* before it. Aggregate
throughput still scales 1.8x from batch 1 to batch 4.

## VL: working, by reusing Qwen3-VL's vision tower

`examples/vllm_neuron/models/qwen3_5/run_vl.py` describes an image correctly on
device. With the `cherry_blossom` asset at 224x224 (a 14x14 patch grid, 196 raw
-> 49 merged tokens):

> "This is a vibrant, vertically oriented photograph that captures a striking
> contrast between natural beauty and urban architecture. **Foreground:** The
> image is dominated by branches of cherry blossom trees (sakura) in full
> bloom..."

That is the asset described accurately, monument included — the tower is
perceiving the image, not confabulating from the prompt.

### The vision half is reused, not reimplemented

Upstream, HF's `Qwen3_5VisionModel` is literally `Qwen3VLVisionModel` with the
deepstack mergers deleted (`modular_qwen3_5.py:437`), the checkpoint's
`model.visual.*` tensor names are identical, and every attribute the plugin's
Qwen3-VL encoder reads off a vision config already exists on
`Qwen3_5VisionConfig`. So `vl.py` reuses:

| reused | from |
|---|---|
| the ViT itself and its weight loading | `qwen3_vl/vision_encoder_bf16.py` |
| `embed_multimodal` (packing + encoder cache) | `qwen3_vl/model_bf16.py` |
| `build_vision_synthetic_inputs` (warmup) | same |
| `merge_vision_embeds`, `mrope`, packing/preprocessing utils | `qwen3_vl/utils/` |

`embed_multimodal` and `build_vision_synthetic_inputs` are taken as plain function
references because their whole contract is `self.visual`,
`self.config.vision_config` and `self._vision_captures`. Subclassing the Qwen3-VL
*model* would drag in its text decoder, which is the one part Qwen3.5 does not
share. `deepstack_visual_indexes` is empty, so the encoder builds no deepstack
mergers and cache rows are exactly `out_hidden_size` wide; the text model raises
if a deepstack tensor ever appears.

### Text-only and VL are two different classes, on purpose

The factory picks by whether the runner supplied a `VisionNeuronConfig`, which it
does exactly when the engine was configured for images or video (the platform
skips vision bucket resolution when every `limit_mm_per_prompt` count is 0, and
without that dict the runner leaves it None). A text-only launch therefore pays
neither the tower's weights nor its compile time.

### Protocol classmethods belong on the factory

`get_vision_token_merge_factor` and `get_max_pixels_token_count` must live on the
**factory**, because `vision_utils` resolves them through the model *registry* and
the registry holds the factory. With them only on the VL class the lookup falls
back to a merge factor of 1, which sizes the encoder cache blocks 4x too large
and dies at graph capture with

    aten::index_put, xla_shape=bf16[65,256,2048]

— the cache rows (256) not matching the encoder's merged output (64). It reads
like a vision bug and is not one. The same applies to
`get_mamba_state_shape_from_config`, already on the factory for the same reason.

### What the CPU vision check can and cannot say

`check_vision_vs_hf.py` establishes: all 297 vision parameters load from Qwen3.5's
checkpoint, no deepstack mergers are built, 62/64 merged tokens are closest to
their own HF row, magnitude ratio 1.000. It deliberately does **not** assert
numerical equality — the vision attention runs on NKI kernels and
`can_run_kernel("cpu")` is False, so off-device it takes a fallback whose
arithmetic differs op-for-op (the first block already differs by 4e-3, compounding
over 24 blocks to ~1e-1). That is a property of running kernel code on CPU, not
evidence about Qwen3.5. Numerical validation of the vision path belongs on device.

## Longer contexts

The text stack compiles and matches CPU on device at **seq 2048 and 4096**, not
just 1024, in both phases (`probe_device_model.py --seq N`). Compile time grows
noticeably at 4096 — the chunked recurrence unrolls 64 chunks per DeltaNet layer
and the attention score matrix is `[heads, 4096, 4096]`.

## Test results

### VL accuracy vs HF

`check_generation_vs_hf.py --vl`, 32 greedy tokens, one image fed at 224x224
(which the processor turns into a 16x16 grid = 256 raw / 64 merged tokens), device
bf16 against HF float32 on CPU: **prefix 12, 16/32 tokens matched**. The
divergence is a synonym choice, not a different reading of the image:

    neuron: "...a striking contrast between natural beauty and urban
             architecture. **Foreground:** The image is dominated by branches of"
    hf    : "...a striking juxtaposition of nature and urban architecture. The
             composition is dominated by delicate pink cherry blossoms (sak"

Both sides go through the same `AutoProcessor` — which is also what vLLM's
frontend uses for this architecture — so pixel values and placeholder expansion
match and a divergence means the model, not preprocessing. Agreement is lower
than text-only's 69% because the tower adds 24 more layers of bf16 error ahead of
the decoder. What matters is the prefix being 12 and not 0.

### Latency at TP=4

896-token prompt (or one image) + 128 output tokens, `AsyncLLM` streaming,
median of 3 rounds after a discarded warmup:

| | batch 1 | batch 4 | batch 8 | VL, 1 image | reference TP=4 |
|---|---|---|---|---|---|
| TTFT | **108.9 ms** | 262 ms | 457 ms | **111.7 ms** | 42.2 ms |
| TPOT | **3.72 ms** | 6.23 ms | 8.90 ms | **3.71 ms** | 4.75 ms |
| decode/stream | 268.7 t/s | 160.5 t/s | 112.4 t/s | 269.5 t/s | 210 t/s |
| aggregate | 268.7 t/s | 482.6 t/s | **640.7 t/s** | — | — |

Three things to take from this:

* **Decode beats the reference** (TPOT 3.72 vs 4.75 ms) and **prefill is 2.6x
  slower** (108.9 vs 42.2 ms) — the split this document predicted before anything
  ran. The NKI port is purely a TTFT win; decode needs nothing.
* **The VL column here is wrong — it measured a cache, not the tower.** It reads
  111.7 vs 108.9 ms text-only, i.e. "the tower costs 3 ms", because this harness
  used to re-send one identical image every round and vLLM served the encoder
  output from its multimodal cache. Re-measured with a unique image per request,
  the tower costs ~15 ms at this size and 176 ms at 1024x1024. See "Latency vs
  image resolution" below; the TPOT figure is unaffected, since decode never runs
  the tower.
* Throughput scales sublinearly (1.00 / 1.80 / 2.38x at batch 1/4/8) and TTFT
  spreads badly with batch (108 -> 804 ms at batch 8) because prefills serialise:
  one prefill graph per forward, so request *n* waits for *n-1*.

### Latency vs image resolution — and the multimodal cache that hid it

**Read the warning first: the earlier version of this section, and the "tower
costs ~3 ms" claim above it, were both wrong, and wrong in the same way.**
`benchmark_latency.py` used to send one byte-identical image every round while
salting only the text. vLLM caches multimodal encoder *outputs* keyed by a hash of
the image, so every measured round after the warmup was served from that cache and
never ran the tower at all. The vision cost came out 7-40x too low. The harness now
salts a 4x4 corner of pixels per request, which changes the hash and leaves the
token count — and therefore the compute — untouched; `--reuse-image` restores the
old behaviour if you actually want to measure the cache.

The size of the mistake, at 1024x1024:

| | TTFT |
|---|---|
| unique image per request | **325.23 ms** |
| identical image reused | 164.21 ms |

What tipped it off was the reference's own VL benchmark (`run_vl_benchmark.py` on
the `qwen3.5-2b-hybrid-deltanet` branch), which reports a TP=4 end-to-end TTFT of
488 ms at 1024x1024 against the 172 ms I had published. A 16x disagreement on a
number that is mostly one ViT should never have been written down as a finding.

Corrected sweep. Batch 1, one image, 128 output tokens, unique image per request,
median of 3 rounds after a discarded warmup. `max_model_len` is pinned at 2048 for
**every** row including the text-only baseline, so the fixed overhead and the text
prefill bucket cancel and what is left is the vision half:

| fed size | grid | raw | merged | block | TTFT | TPOT | vision cost |
|---|---|---|---|---|---|---|---|
| *text-only, no tower* | — | — | — | — | *148.96 ms* | *3.98 ms* | *baseline* |
| 224x224 | 1,16,16 | 256 | 64 | 256 | **163.80 ms** | 3.97 ms | +14.8 ms |
| 512x512 | 1,32,32 | 1024 | 256 | 1024 | **183.36 ms** | 3.99 ms | +34.4 ms |
| 768x768 | 1,48,48 | 2304 | 576 | 2304 | **232.32 ms** | 4.00 ms | +83.4 ms |
| 1024x1024 | 1,64,64 | 4096 | 1024 | 4096 | **325.23 ms** | 3.99 ms | +176.3 ms |

So resolution matters a great deal: the tower goes from 10% of TTFT at 224x224 to
54% at 1024x1024. Fitting the tight-block rows,

    vision cost ~= 9.7 ms + 18.6 us/token + 5.39 us per 1000 token^2

which predicts the held-out 2304-token row at 81.1 ms against 83.4 measured. At
4096 tokens the quadratic term alone is 90 ms, **51% of the vision cost** — the
encoder's attention, and the reason this curve steepens. Extrapolating to
2048x2048 (16384 raw) gives roughly 1.7 s, so the reference's decision to tile
that case into four 4096-patch blocks is not optional.

**TPOT stays flat at 3.97-4.00 ms** at every resolution, which is the one thing the
broken measurement also got right, and for a sound reason: decode never re-runs the
tower.

#### The compiled block size dominates, not the image

The oversized-block controls are where the practical advice lives, and the effect
is far bigger than the ~5 ms I originally reported:

| control | block | raw | TTFT | vision cost | vs tight block |
|---|---|---|---|---|---|
| 512x512 in an oversized block | 4096 | 1024 | 289.25 ms | +140.3 ms | **4.08x** |
| 768x768 in an oversized block | 4096 | 2304 | 306.05 ms | +157.1 ms | **1.88x** |

A 4x oversized block costs 4.08x, and a 1.78x oversized block costs 1.88x — the
tower pays for the whole padded block almost as if it were real content. Splitting
the two terms at block 4096 gives

    cost at block 4096 ~= 128 ms + 11.7 us per *real* token

so 128 of the 176 ms at 1024x1024 is the block, not the picture. Hence:
**size `vision_attention_block_size` to the image you actually send.** For a
512x512 image, a tight 1024 block instead of 4096 saves 106 ms of TTFT — far more
than anything left to win in the text decoder. Non-power-of-two blocks are fine;
2304 compiled and ran.

#### The encoder was running unsharded — `vision_neuron_config.tp_size`

The single biggest win found so far, and it is a configuration default rather than
anything in this port's code. `VisionNeuronConfig.resolve_tp_dp` reads:

    Both at default (1) -> tp_size=1, dp_size=world_size

So leaving both unset — which every script here did, and which the numbers above
were measured with — gives **tp=1, dp=4**: the vision encoder is *replicated* on
each rank with all 16 heads, and vision data-parallelism splits work per *item*, so
a one-image request has three of four ranks idle. Setting `tp_size: 4, dp_size: 1`
shards heads, MLP and merger as `vision_encoder_bf16.py` already supports:

| 1024x1024 | default (tp=1, dp=4) | `tp_size=4` | |
|---|---|---|---|
| TTFT | 325.23 ms | **230.40 ms** | −95 ms, −29% |
| vision cost | +176.3 ms | **+81.4 ms** | **2.17x faster** |

| 512x512 | default | `tp_size=4` | |
|---|---|---|---|
| TTFT | 183.36 ms | 179.65 ms | −3.7 ms |
| vision cost | +34.4 ms | +30.7 ms | 1.12x |

2.17x rather than 4x, because sharding adds two all-reduces per layer over 24
layers; the reference measured a comparable 3.05x (308 -> 101 ms standalone). The
win is worth little at 512x512, where vision is a small share of TTFT, and large
wherever the image is big.

Output stays correct — the same image description either way, differing only in
wording, which is what a changed reduction order does to bf16:

    tp=4:    "...juxtaposition of nature and urban architecture against a clear
              blue sky... dominated by an abundance of delicate pink"
    default: "...juxtaposition of nature and urban architecture under a clear
              blue sky... dominated by delicate, pink cherry blossom (sakura)"

This is deliberately *not* forced on in the model code, because the default is the
right one for a different workload: with several images in flight, `dp` encodes
several items concurrently, while `tp` only makes each one faster. Rule of thumb —
**`tp_size=4` for single-image latency, the `dp` default for many-image
throughput.** `run_vl.py` and `benchmark_latency.py` both take `--vision-tp`.

#### Against the reference — cloned and run on this box

The README comparison that used to sit here was second-hand and, as it turned out,
wrong about which half was slow. The reference is now cloned at
`/mnt/nvme/nxdi_ref` (branch `qwen3.5-2b-hybrid-deltanet`) and run on *this*
trn2.3xlarge, at TP=4, on the same `cherry_blossom` source image. Use the
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` venv — it has NxDI 0.10.0 and
transformers 4.57.6, exactly what their README asks for. Two setup notes: their
`bin` must be on `PATH` (`libneuronpjrt-path` is an executable there, not a
library), and `run_vl_benchmark.py` expects images pre-staged at
`/tmp/test_image_<size>.jpg`.

Fair head-to-head — tight vision bucket both sides, and each given the *smallest
text bucket that fits* rather than a pinned one:

| image | NxDI | here | |
|---|---|---|---|
| 512x512 | 137.6 ms | **119.9 ms** | 1.15x faster here |
| 1024x1024 | 575.1 ms | **230.4 ms** | **2.50x faster here** |
| TPOT | 4.83-4.91 ms | 3.58-3.99 ms | ~1.25x faster here |

Their published figures roughly reproduce (126 -> 137.6, 488 -> 575.1); the residual
is this box against the trn2.48xlarge their README used. Note also that their TTFT
*excludes* image preprocessing — `apply_chat_template` sits outside their timer —
while these rows include it, so the gap is understated rather than flattered.

Subtracting each side's own text-only TTFT at the same bucket splits it:

| | NxDI | here | |
|---|---|---|---|
| vision, 1024 patches | 106.7 ms | 30.7 ms | **3.47x faster here** |
| vision, 4096 patches | 484.5 ms | 81.4 ms | **5.95x faster here** |
| text prefill, bucket 512 | **30.9 ms** | 89.2 ms | 2.89x faster *there* |
| text prefill, bucket 2048 | **90.6 ms** | 149.0 ms | 1.64x faster *there* |

So the whole advantage is the vision encoder, and the text decoder is where this
port is *behind* — the opposite of the story the README numbers suggested. Two
checks that the split is real: the implied text cost at bucket 512 (119.9 − 30.7 =
89.2 ms) lands on the independent `69 + 0.039/token` fit's 89.0 ms; and their
text-only TTFT is *not* flat with bucket size (30.9 ms at 512, 90.6 ms at 2048),
contrary to the README's claim, which is measured here directly.

#### One graph or two, and the encoder with the glue removed

**NxDI compiles the two halves separately.** From
`NeuronQwen35VLForCausalLM`'s own docstring: *"Separate compilation/loading of
vision encoder and text decoder"* — `compile_vision_encoder_tp.py` emits
`bucket_<N>/tp_*.pt`, `parallel_model_load` restores it, and
`NeuronQwen35VLForCausalLM` composes the two in Python. This port instead runs the
tower inside the same compiled graph as the text model.

Timing their vision path at two nested levels (`/tmp/nxdi_ve_probe.py`, median of
5 after a warmup) separates encoder from glue. L1 is the raw traced graph
(`_TPVisionAdapter.__call__` on pre-built inputs), L2 adds
`wrapper.forward`'s CPU prep:

| | 512x512 (1024 patches) | 1024x1024 (4096 patches) |
|---|---|---|
| L1 raw graph, device only | 15.0 ms | **101.1 ms** |
| L2 + wrapper prep | 41.9 ms | 220.1 ms |
| prep alone | 26.9 ms | 119.0 ms |

L1 at 4096 patches reproduces their README's "101 ms standalone TP=4 encoder"
exactly, which is good evidence L1 is the number that claim refers to. Their
1024x1024 TTFT of 575.1 ms therefore decomposes as:

| ms | share | |
|---|---|---|
| 101.1 | 18% | vision graph, device |
| 119.0 | 21% | wrapper prep — patch-embed, RoPE, a 4096x4096 bf16 mask built per call, pad, merge |
| 90.6 | 16% | text prefill, bucket 2048 |
| 264.4 | 46% | VL orchestration left over: mRoPE, embedding injection, `generate()` setup |

**So 383 ms of their 575 ms — 67% — is glue, and only 101 ms is the encoder.** That
is the real answer to "why is this port faster end-to-end", and it is an
integration cost, not a modelling one.

Encoder against encoder, with their glue removed and none of ours removed:

| 4096 patches | |
|---|---|
| NxDI, device graph only | 101.1 ms |
| this port, *all* vision-related work (CPU preprocessing + graph + scatter) | **81.4 ms** |

So this port is at least 1.24x faster even measured against their glue-free number,
and by more than that graph-to-graph, since 81.4 ms still carries preprocessing.
The likely cause is visible in their `_forward_attention`: it is plain PyTorch over
an explicit `(1, 1, bucket, bucket)` mask — 33 MB of bf16 at bucket 4096, rebuilt
on CPU every call, which is also most of the 119 ms prep — where this port uses
`NF.flash_attention` and never materialises a full score matrix. A graph-to-graph
number for this port has *not* been measured; the tower is fused into the text
graph, so isolating it needs a separate harness.

#### What the reference's glue actually does

The 264 ms of "VL orchestration" attributed here earlier **did not exist**. It was
an artefact of assuming the reference's VL text prefill costs what its standalone
text smoke test costs at bucket 2048 (90.6 ms). Profiling one prefill
(`probe_nxdi_glue.py`, cProfile over `generate(max_new_tokens=1)`) shows the text
call is 368 ms, and nothing is unaccounted for:

| ms | what |
|---|---|
| **368.0** | text CTE forward |
| 101.1 | vision graph, device |
| 85.0 | `patch_embed` as a **CPU `torch.conv3d`** |
| 14.0 | pos-embed interpolate (11) + mask build (3), both grid-only |
| 7.0 | mRoPE (`get_rope_index` is only ~1 ms), vision scatter, misc |

Sums to 575 ms. So the glue is three things, in order of size:

1. **The text half runs at the wrong bucket.** Their code pads `vision_embeddings`
   and `vision_mask` to `text_config.neuron_config.seq_len`, which is the *maximum*
   (4096), not the bucket the 1048 real tokens need. Measured text-only: 90.6 ms at
   bucket 2048 against 173.8 ms at 4096, so **~83 ms is thrown away** on bucket
   over-selection.
2. **~194 ms of VL-specific cost inside the text call**, beyond text-only at the
   same 4096 bucket. Not decomposed; the obvious suspect is the padded
   `vision_embeddings` — 4096 x 2048 bf16 is 16 MB copied to device per request —
   plus the injection itself.
3. **99 ms of CPU work that belongs on the device**: an 85 ms `conv3d` patch
   embedding, plus 14 ms of grid-only work that could simply be cached.

Almost none of this needs one graph to fix — (1) is a bucket-selection bug, (3) is
moving `patch_embed` into the *existing* vision graph and memoising three helpers.

**Correcting the claim made here earlier:** "strip their glue and they land at
191.7 ms and beat this port" assumed text = 90.6 ms. With text measured at 368 ms,
fixing the CPU work and the bucket still leaves ~386 ms, well behind the 230.4 ms
here. They only reach ~192 ms if the unexplained 194 ms goes too. So the lead here
is more robust than that correction feared — but it is still a lead over an
integration, and the 194 ms is unidentified, so do not treat it as settled.

#### The one number worth taking from all of this

The reference's text-only prefill against bucket size, TP=4, measured here:

| bucket | 512 | 2048 | 4096 |
|---|---|---|---|
| TTFT | 30.9 ms | 90.6 ms | 173.8 ms |

which fits **9.9 ms fixed + 39.9 us/token**, against this port's
**69 ms fixed + 39.0 us/token**. The per-token slopes are the same to within 2%.
The entire text-prefill gap is the *fixed* cost — 69 ms against 10 ms — not the
model, not the DeltaNet, not the torch attention. Everything already known about
that 69 ms says it is per-request framework overhead, and this is independent
confirmation from a second implementation on the same hardware that ~10 ms is
achievable. That makes it unambiguously the top item.

The practical read for this port: the text prefill is the gap worth closing, and
69 ms of its 89-149 ms is the fixed per-request overhead already flagged below.

Caveats: every row is one image at batch 1, so nothing here measures several large
images in one request, where the block cost is paid per block. And the 2048x2048
case the reference tiles has not been run at all here.

### Feature coverage, all on device

| | result |
|---|---|
| text, seq 1024 / 2048 / 4096 / 8192 | works (2048+ device-verified via `probe_device_model.py`) |
| batch 1 / 4 / 8 | works |
| one image, 256 raw / 64 merged tokens | described correctly |
| two images | **both** described correctly and distinctly — cherry blossoms with a tower, then "a small, dark-colored SUV parked on a city street, with a prominent red STOP sign" |
| video, 4 frames (640x360, 440 merged tokens) | "A baby wearing glasses sits on a bed and reads a book... holding the book with both hands and turning the pages" |
| image at 448x448 (784 raw / 196 merged) | richer description: "clusters of small, delicate flowers and slender green stems ... vivid, clear blue sky" |
| image at 672x672 (1764 raw / 441 merged) | richer again: "upward-looking photograph ... dense canopy ... branches, dark brown and slender, crisscross the frame" |
| VL at batch 2 | works — TTFT 160 ms, TPOT 5.05 ms, 317.6 tok/s aggregate |
| text, seq 8192 | works (device-verified, both phases) |

Two sizing traps, both in configuration rather than model code:

* `vision_attention_block_size` is **per block** and one block holds exactly one
  item (`ffd_pack_images(..., one_item_per_block=True)`), while
  `num_vision_tokens_buckets` is the **total** budget and caps blocks per request.
  Setting them equal works for one image and fails for two.
* A video is packed per temporal group and **`mm_processor_kwargs["max_pixels"]`
  does not reach the video processor** — `baby_reading` still reported 440
  embedding tokens with `max_pixels=65536`. Size block/bucket for the native grid
  instead: 4 frames needed `block_size=1024, bucket=2048`.

## Where prefill time actually goes — and it is not what this doc predicted

Measured with `probe_device_model.py --time N` plus the ablation switch, 24
layers, TP=4 shapes, bf16 (collectives excluded, so these are one rank's compute
rather than engine TTFT — the *ratios* are the point):

| | seq 1024 | seq 4096 |
|---|---|---|
| all mixers on | 27.79 ms | 147.27 ms |
| mixers off (embed + MLP + norms + head) | 4.76 ms | 25.35 ms |
| attention ablated -> **DeltaNet cost** | **21.70 ms** (78%) | **92.35 ms** (72%) |
| DeltaNet ablated -> **attention cost** | **1.81 ms** (6.5%) | **10.59 ms** (8%) |
| per layer | 1.21 ms delta / 0.30 ms attn | 5.13 / 1.77 |

**This contradicts what the "Constraints specific to this box" section above
predicted.** That section reasoned that because `head_dim` is 256 and the flash
kernel caps at `MAX_HEAD_DIM = 128`, the 6 torch-fallback attention layers would
set the performance ceiling — citing InternVL, where materialising an `s x s`
score matrix cost 12.1x. It is wrong for this model: torch attention is only
6-8% of prefill. The pure-torch chunked delta rule is ~3/4 of it, in 18 layers.

The reason the analogy failed: only 2 query heads per rank at TP=4, so the
`[2, seq, seq]` score matrix is small, while the delta rule runs a 12-matmul
blocked inverse plus a per-chunk carried recurrence over `seq/64` chunks in each
of 18 layers.

Scaling 1024 -> 4096 (4x tokens): DeltaNet 4.26x (roughly linear, as a linear-
attention kernel should be), attention 5.85x (super-linear but far from the 16x a
pure quadratic would give), spine 5.33x. So the balance barely shifts with context
length and **the DeltaNet NKI port is the right optimisation across the range**,
targeting ~72-78% of prefill. Decode still needs nothing.

## The NKI port: correct, slower, and not where the time is

Two results, both worth keeping.

### Two kernels vendored; the inverse algorithm is most of the story

Both of the reference's chunked CTE kernels are vendored — `nki_deltanet.py` (its
"legacy" one) and `nki_deltanet_fused.py` (its current one) — behind
`VLLM_NEURON_QWEN35_ENABLE_NKI=1` with `VLLM_NEURON_QWEN35_NKI_VARIANT` selecting
between them. Torch still ships. They differ only in how they apply the in-chunk
`(I - A)^-1`, and that difference is 5.2x:

| variant | in-chunk inverse | ms/call | vs torch |
|---|---|---|---|
| torch | blocked elimination, 12 batched matmuls over all `(head, chunk)` pairs | **0.81** | — |
| `fused` | **Neumann by power-doubling on 32x32 leaves** (`_leaf_inverse32_t`, 5 squaring rounds) then **Schur composition** 32 -> 64 -> 128 (`_offdiag_combine_t` = `left @ cross @ right`) | 1.86 | 0.43x |
| `legacy` | forward substitution: 128 sequential steps, each a full 128x128x128 matmul with all but one row of the result masked away | 9.74 | 0.08x |

(4 heads, one rank, seq 1024. Both kernels are *correct*: output within 1.2e-05 of
the torch reference, final state within 2.5e-06.)

**Note the reference does use a Neumann series** — on 32x32 leaves, inside a
hierarchical blocked scheme. That is not the same thing as the full-matrix
repeated-squaring this port rejected in `_strictly_lower_inverse`: at 64x64
unmasked, `|A^16|` reaches 2.7e6 and the product loses every digit (0.57 absolute
error). Confined to a 32x32 leaf the same trick costs about 1.2e-3 — measured with
this checkpoint's weights, against 1.2e-7 for elimination — which the recurrence
then damps to 1.2e-05 end to end. Both statements are true and the earlier wording
here, which said flatly that Neumann "is not usable", was too broad.

**There is no library on either side.** The "torch" path is not eager PyTorch:
dynamo traces it to an FX graph (336 `matmul` nodes for this function), the
plugin's backend lowers it to HLO, and *neuronx-cc* — the same compiler that
builds the NKI kernels — emits a NEFF for the same Tensor Engine. Counting the
in-chunk inverse plus its application exactly, per 128 tokens of one head:

| | inverse FLOPs | vs torch | time vs torch | achieved throughput |
|---|---|---|---|---|
| torch | 16.8 MFLOP | 1.00x | 1.00x | 1.00x |
| `fused` | 7.9 MFLOP | **0.47x** | 2.30x | **0.20x** |
| `legacy` | 536.9 MFLOP | **32x** | 12.02x | **2.66x** |

So the two kernels lose for opposite reasons, and neither is "torch has a faster
library":

* `legacy` loses on **arithmetic** — 32x the FLOPs, while achieving 2.7x *better*
  hardware utilisation than the torch path. A well-engineered kernel running a
  wasteful algorithm.
* `fused` loses on **utilisation** — it does less than half the arithmetic and
  still takes 2.3x the time, so it reaches a fifth of the throughput.

**Why the fused kernel still loses.** Not the inverse any more — the serial
structure. Each kernel is one launch per `(batch, head)` and walks chunks with
`sequential_range`, because its design goal is keeping the 128x128 state in SBUF
across chunks. The torch path folds every `(head, chunk)` pair into one batch
dimension and issues batched matmuls, so its serial region holds only four small
matmuls per chunk. HF's factorisation is what permits that:
`v_new = T @ v_beta - (T @ (k_beta * e^g)) @ state` keeps both `T` applications
free of `state`, so they hoist out of the loop; the kernels solve
`(I - A) v_new = rhs(state)` instead, which pins the solve inside it.

The reference's *multihead* variant batches head groups — the missing ~4x, which
could plausibly close the remaining 2.3x. It is also the variant it documents as
numerically unstable on real vision embeddings, so this port does not use it.

**One hypothesis tested and refuted.** If leaf Neumann were the cause of that
instability, vision embeddings should produce a worse-conditioned `A`. Measured
with the real tower's output fed into layer 0: they do not — `|A_leaf^8|` is 4907
on vision against 7027 on text, and the leaf-Neumann error is 1.17e-3 against
1.27e-3. The reason is that `k` is L2-normalised, so `A` is nearly invariant to
input magnitude, even though the vision embeddings themselves are 10x larger in RMS
(0.152 vs 0.015). That independently corroborates the reference's own observation
that random vectors at the same std decode cleanly, and leaves its hazard still
unexplained.

### TTFT is 63% fixed overhead, so the model was never the main cost

Measured engine TTFT against prefill bucket size, batch 1:

| prefill bucket | TTFT | marginal |
|---|---|---|
| 512 | 89.20 ms | — |
| 1024 | 108.90 ms | 0.0385 ms/token |
| 2048 | 149.85 ms | 0.0400 ms/token |

Linear, and the implied intercept agrees to +-1 ms from all three points:

    TTFT ~= 69 ms fixed + 0.039 ms/token

So at seq 1024 the ~109 ms TTFT decomposes roughly as

| | ms | share |
|---|---|---|
| fixed per-request overhead | ~69 | 63% |
| model compute (one rank, from `probe_device_model --time`) | ~28 | 26% |
| collectives (the residual) | ~12 | 11% |

and the DeltaNet's ~22 ms is **20% of TTFT**, not the 78% the earlier "share of
model compute" figure might suggest. Read the two numbers together: the delta rule
dominates *compute*, and compute does not dominate *TTFT*.

That reframes the comparison with the reference's 42.2 ms. Most of our gap is the
69 ms fixed term — vLLM's per-request path (scheduling, CPU-side slot-mapping /
block-table / mRoPE construction and the transfers to device, NEFF launch,
sampling, AsyncLLM IPC) — not the mixers, and the reference measures under NxDI's
own harness rather than this one. **Anyone chasing TTFT should profile that fixed
term first;** there is more there than in any kernel.

Decode is unaffected by all of this and already beats the reference (TPOT 3.72 vs
4.75 ms).

## Status

- [x] Branch, checkpoint, reference clone
- [x] `config.py`, validated against the real checkpoint
- [x] `LayerSpec`/`KVSpec` + `MambaSpec` plumbing (spec emission + allocation)
- [x] DeltaNet layer, matched to HF in float32 (`deltanet.py`)
- [x] Gated GQA layer with partial interleaved mRoPE (`model.py`)
- [x] Text decoder, factory, registry entry
- [x] Boots, compiles and generates at TP=4
- [x] **Correct text-only output at TP=4**
- [x] Accuracy cross-check against HF on device — 69.4% greedy tokens, 3/5
      prompts exact, no first-token mismatches (reference bar: 66%, 3/5)
- [x] Latency at TP=4 — decode beats the reference (TPOT 3.72 vs 4.75 ms),
      prefill is 2.6x slower (TTFT 108.9 vs 42.2 ms)
- [x] Longer contexts — device-verified at seq 2048 and 4096
- [x] **VL stage** — image described correctly on device
- [x] VL accuracy against HF, and VL latency — but the first latency pass measured
      the multimodal cache instead of the tower; see the correction below
- [x] Video input, multi-image, batches up to 8
- [x] NKI kernel port — done, and it *loses* to the batched torch path (0.10x);
      vendored and off by default, with the measurement recorded
- [ ] Profile the ~69 ms fixed per-request overhead, which is 63% of TTFT and the
      real gap to the reference — bigger than anything left in the model
- [x] Contexts to 8192, VL at batch > 1, images to 672x672
- [x] Latency vs image resolution — swept 224 to 1024x1024 with a unique image per
      request; TPOT is flat, but vision costs +176 ms at 1024x1024 (54% of TTFT) and
      an oversized `vision_attention_block_size` multiplies that by up to 4x
- [x] Speed up the vision encoder — it was running *unsharded* (`resolve_tp_dp`
      defaults to tp=1/dp=world_size); `vision_neuron_config.tp_size=4` cuts vision
      cost 2.17x and TTFT 29% at 1024x1024
- [x] Run the reference on this box rather than quoting its README — cloned at
      `/mnt/nvme/nxdi_ref`, TP=4: this port is 2.50x faster end-to-end at
      1024x1024, entirely on the vision side (5.95x), while its text prefill is
      1.64x *slower* than the reference's
- [ ] **Kill the 69 ms fixed per-request overhead — the highest-value item left,
      and now precisely bounded.** The reference's text prefill fits
      `9.9 ms + 39.9 us/token` against ours at `69 ms + 39.0 us/token`: the
      *per-token slopes match to 2%*, so the entire text gap is fixed cost, and a
      second implementation on this same box shows ~10 ms is achievable. Nothing in
      the model needs touching.
- [ ] 2048x2048, which the reference handles by tiling into four 4096-patch blocks
- [x] Push the branch
- [ ] Several *large* images in one request — the per-block cost should scale with
      the block count, but only single-image blocks have been measured
