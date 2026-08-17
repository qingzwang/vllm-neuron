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

## Status

- [x] Branch, checkpoint, reference clone
- [x] `config.py`, validated against the real checkpoint
- [x] `LayerSpec`/`KVSpec` + `MambaSpec` plumbing (spec emission + allocation)
- [ ] DeltaNet layer + NKI kernel port
- [ ] Gated GQA layer with partial interleaved mRoPE
- [ ] Text-only bring-up at TP=4, HF cross-check
- [ ] VL stage

Next: the DeltaNet layer. The state buffers now arrive as
`kv_caches[layer_name] = [conv_state, recurrent_state]`, so the layer needs to read
and write those in place, and the NKI kernel port decides how. Start from the
reference's `nki_deltanet.py` (recurrent step, simplest) to get decode correct
before taking on the chunked/fused prefill path — and remember the fused
multihead kernel is the one that breaks on vision embeddings later.
