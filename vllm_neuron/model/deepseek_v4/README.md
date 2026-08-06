# DeepSeek-V4

DeepSeek-V4-Flash / Pro: a Mixture-of-Experts model with hybrid sparse attention
and manifold-constrained hyper-connections, supporting a 1M-token context.

Validated against `deepseek-ai/DeepSeek-V4-Flash-0731` (284B total / 13B
activated, FP4 experts + FP8 elsewhere).

## Architecture

| Parameter                 | Value                                        |
| ------------------------- | -------------------------------------------- |
| hidden_size               | 4096                                         |
| num_hidden_layers         | 43                                           |
| num_attention_heads       | 64                                           |
| num_key_value_heads       | 1 (MLA latent KV, shared by all Q heads)     |
| head_dim                  | 512 (448 content + 64 RoPE)                  |
| q_lora_rank / o_lora_rank | 1024 / 1024                                  |
| o_groups                  | 8                                            |
| moe_intermediate_size     | 2048                                         |
| n_routed_experts          | 256 (top-6) + 1 shared                       |
| num_hash_layers           | 3                                            |
| vocab_size                | 129280                                       |
| RoPE                      | YaRN, **interleaved pairs** (GPT-J style)    |
| Activation                | SwiGLU, clamped at ±10 (gate: upper only)    |
| Normalization             | RMSNorm (fp32 accumulate)                    |
| Residual                  | Hyper-connections, `hc_mult=4`               |
| tie_word_embeddings       | false                                        |

### Hybrid attention

Every layer attends over a 128-position sliding window of latent KV. Layers
additionally attend over a *compressed* KV stream, selected by
`compress_ratios[i]`:

| `compress_ratio` | Layers | Mode                                              |
| ---------------- | ------ | ------------------------------------------------- |
| 0                | 2      | Sliding window only                               |
| 4                | 21     | **CSA** — indexer picks top-512 compressed slots  |
| 128              | 20     | **HCA** — attends over all compressed slots       |

The compressor pools `compress_ratio` consecutive positions into one latent slot
via a learned gate. Ratio-4 compressors use *overlapping* windows (double-width
projection); ratio-128 do not. CSA layers additionally run an `Indexer`, which
maintains its own 128-wide compressed stream purely for scoring.

### Key differences from the reference implementation

The checkpoint ships a reference implementation in `inference/model.py` (CUDA +
tilelang). This port differs as follows:

- **FP4 and FP8 weights are dequantized to bf16 at load time.** Trainium has no
  FP4 datapath. See the memory note below.
- **Quantization-aware fake-quant is preserved in activations.** The reference
  simulates the FP8 / FP4 grids on the latent KV, the compressor outputs and the
  indexer Q (`act_quant(..., inplace=True)`). Dropping this shifts activations
  ~2-3% relative per layer, so `layers.fake_quant_fp8` / `fake_quant_fp4`
  reproduce it exactly.
- **Hadamard rotation is written in torch** (`layers.hadamard_rotate`) rather
  than `fast_hadamard_transform`, so it traces on Neuron.
- **Attention and the hyper-connection Sinkhorn are purpose-built NKI kernels.**
  The plugin's flash-attention kernel caps `head_dim` at 128 and MLA needs 512,
  so `functional/attention/sparse_latent.py` and `functional/hc_sinkhorn.py`
  implement them directly. Both fall back to torch for decode — see the
  hardware-status section.
- **DSpark speculative decoding is not implemented.** The `-0731` revision
  replaces the single MTP block with 3 DSpark stages plus Markov and confidence
  heads (`mtp.*` in the checkpoint, `dspark_*` in `config.json`). The backbone is
  served without them; requesting speculative decoding raises.

## Compressed KV state is model-owned

vLLM's paged KV cache addresses one slot per token via `slot_mapping`. The
compressed streams advance one slot per `compress_ratio` tokens, so the runner's
slot mapping cannot address them. Only the sliding-window stream is declared in
`get_kv_spec()` (as a `SlidingWindowSpec`, which is what keeps a 1M context
affordable); the compressed streams live in model-owned buffers
(`compressed_state.py`).

Consequences:

- **Prefix caching does not cover compressed state.** Cached blocks are reused
  for the window, but the compressed stream is rebuilt from the prompt on every
  prefill.
- **No cross-request block sharing** of compressed state.
- Buffers hold **one request's** stream, so `--max-num-seqs 1` is enforced
  whenever any layer has compression. Concurrent requests would otherwise read
  each other's compressed KV.

Folding these into the paged allocator needs `LayerSpec` to express a per-group
token-to-slot stride, which the current runner contract does not have.

## Routing is near-degenerate in low precision

With real weights, the median relative gap between the 6th and 7th ranked expert
score is **~0.2%**, and the minimum is ~0. Any bf16-level perturbation flips
top-k selection for a small fraction of tokens. Router scoring is therefore done
in fp32, and **token-exact agreement with the reference is not a valid accuracy
gate** — use logit cosine similarity and argmax agreement instead.

Measured against the reference implementation on real checkpoint weights
(6 layers spanning all three attention types, 40 tokens, CPU):

| Metric              | Value    |
| ------------------- | -------- |
| hidden-state cosine | 0.99997  |
| logit cosine        | ~1.0000  |
| argmax agreement    | 100%     |

Component-level agreement with the reference is exact (maxdiff 0.0) for RMSNorm,
YaRN frequencies, interleaved RoPE, the hyper-connection pre/post/head mixes,
both compressor variants, the sparse latent attention, and the FP8/FP4
fake-quant and Hadamard helpers. The MoE agrees to bf16 accumulation order
(~5e-3 relative).

## Neuron compiler constraints

Six neuronx-cc 2.26 behaviours shape the implementation. All are load-bearing —
reverting any of them reproduces a compile failure or wrong numbers. Note that
the NKI CPU simulator accepts every one of the kernel-side three, so they only
surface when tracing on device.

### In the traced torch graph

* **No `sort`.** XLA lowers a small `torch.topk` to `sort`. The MoE router uses
  `layers.topk_mask` (iterated max-and-mask) instead, and the indexer skips
  top-k entirely when every slot fits the budget.
* **No f64.** A bare Python float in a tensor expression is typed f64. Scales
  and mask fill values are materialized as same-dtype tensors, and `-inf` is
  replaced by `finfo.min`.
* **No conditional in-place buffer updates** (internal error `NCC_ILSA902`).
  Slice assignment into a preallocated buffer, and `torch.where` feeding an
  in-place `index_put_` / `index_copy_`, both trip it. Buffers are built whole
  with `cat` and written with a single flat `copy_`; index steering uses
  `clamp` or arithmetic blends rather than selects.

### Inside the NKI kernels

* **`nl.*` expressions are lazy.** An online-softmax rescale left as an
  expression over `running_max` is re-evaluated after the writeback and
  collapses to `exp(0) = 1`, silently dropping the correction. Intermediates are
  materialized with `tensor_copy`.
* **Broadcasting a `[H, 1]` tile against a partition slice of a 3-D SBUF tile
  misaligns the partition axis.** Accumulators are laid out 2-D as
  `[H, blocks * 128]` instead. A scalar operand also has to be widened with
  `nl.broadcast_to` to the destination's partition count.
* **`nl.divide` is not a supported activation operator, and gather indices must
  be `uint32`.** Normalization takes a reciprocal and multiplies; indices are
  clamped (making them non-negative) and cast.

Tensor Engine mechanics add two more shapes to work around rather than
constraints to avoid: transpose writes PSUM (the Vector Engine path caps at
32x32) and matmul operands must be SBUF, so each transpose routes through PSUM
and copies back.

## Parallelism layout

`TP=N` and `EP=N` describe two different partitions of the *same* N ranks, not
N x N of anything:

| Degree  | Partitions                                         | Per rank at N=64    |
| ------- | -------------------------------------------------- | ------------------- |
| `TP=N`  | Q heads, embedding, LM head, attention projections | 64 / 64 = 1 Q head  |
| `EP=N`  | The 256 routed experts                             | 256 / 64 = 4 experts |

Every rank runs attention for its own heads and the FFN for its own experts. The
latent KV is a single head shared by all Q heads, so it is replicated rather
than sharded.

The two interact through `tp_degree = world_size / ep_degree` (see `moe.py`):
`EP == world_size` leaves `tp_degree = 1`, so each rank holds *whole* experts
with an unsharded intermediate dim. `EP=8` at world size 64 would instead give
every rank 32 experts at an eighth of the intermediate width each. Setting
`EP == TP` is preferred because the decode path evaluates every local expert
densely, so fewer local experts is faster.

TP=64 is the full machine and the configuration to use. A trn2.48xlarge reports
`logical-neuroncore-config: 2`, pairing physical cores into logical ones: 16
devices x 8 physical cores = 128 physical, exposed as **64 logical** cores, and
TP addresses logical cores. At TP=64 the weight share is 574 GiB bf16 / 64 = 9.0
GiB per rank against 48 GiB of HBM per logical core, and the attention sharding
reaches its limit at 1 Q head per rank (64 heads / 64 ranks). Going wider would
need `num_attention_heads` to divide further.

Setting `NEURON_LOGICAL_NC_CONFIG=1` unpairs the cores and exposes 128 logical
cores, but TP is capped at 64 by the Q-head count regardless.

## Hardware status

The full 43-layer `DeepSeek-V4-Flash-0731` checkpoint loads, compiles and
generates correctly on a trn2.48xlarge at `--tensor-parallel-size 64` with
expert parallelism, and identically at TP=32.

```
prompt:  "The capital of France is"
output:  " Paris. The capital of Spain is Madrid. The capital of Italy is Rome."
```

64-token prompt, 32 output tokens, `--max-model-len 128`, `--max-num-seqs 1`,
best of 3 runs per shape.

Scaling across the machine (both with the NKI-prefill / torch-decode path):

| Config          | TTFT       | TPOT       | Decode throughput |
| --------------- | ---------- | ---------- | ----------------- |
| TP=32, EP=32    | 504 ms     | 88.2 ms    | 11.3 tok/s        |
| **TP=64, EP=64** | **467 ms** | **69.4 ms** | **14.4 tok/s** |

TP=64 uses all 64 logical cores and is the configuration to prefer: 27% better
TPOT and 7% better TTFT. TPOT gains more than TTFT because decode runs the dense
MoE path, and doubling EP halves the local experts each rank evaluates (8 -> 4).

Kernel-path comparison at TP=32:

| Path                          | TTFT       | TPOT        | Decode throughput |
| ----------------------------- | ---------- | ----------- | ----------------- |
| torch attention + HC          | 653 ms     | 88.3 ms     | 11.3 tok/s        |
| NKI everywhere                | 507 ms     | 106.2 ms    | 9.4 tok/s         |
| **NKI prefill, torch decode** | **504 ms** | **88.2 ms** | **11.3 tok/s**    |

The kernels win on prefill and lose on decode, so both ops fall back to torch
below 8 tokens — see `functional/attention/sparse_latent.py`.

TTFT is flat across prompt lengths because both pad to the single compiled
128-token prefill bucket. Weight load takes ~1980 s at TP=64 and ~2330 s at
TP=32 (FP4/FP8 dequantized to bf16 on every rank); compilation adds ~1800 s.

These are first-working numbers, not a tuned configuration. Decode is where the
headroom is: it runs the dense MoE path (every local expert evaluated for one
token) and a torch sparse attention, both bounded by the compiler constraints
below rather than by hardware throughput.

Two environment variables are required on a host without EFA (this instance has
none): `NEURON_SKIP_EFA_AFFINITY=1`, since EFA affinity is a CPU optimization
rather than a correctness requirement, and `neuronx-cc` on `PATH` — the venv
ships it in `bin/` but the worker resolves it by name.

### Compiler instruction budget bounds the context length

`neuronx-cc` caps a graph at 5M machine instructions (`NCC_ELUR015`). The
binding term is the **context length**, not the layer count: a CSA layer's
compressed stream holds `max_model_len / 4` slots, and the sparse attention
gathers over `sliding_window + slots` per query.

With the sparse attention and the hyper-connection Sinkhorn both in NKI, the
instruction count no longer scales with the selection width or the round count:

| `max_model_len` | torch path                 | with NKI kernels     |
| --------------- | -------------------------- | -------------------- |
| 128             | all ranks                  | all ranks (32 and 64) |
| 512             | 30/32 (2 over by 9%)       | **all ranks; decode faults (above)** |
| 2048            | fails (6.9M instructions)  | not yet measured     |

The instruction count is per-rank and driven by the context length, not the rank
count, so widening TP does not change this envelope.

Beyond `max_model_len 2048` the indexer's selection also has to engage: a CSA
layer's compressed stream then holds more than `index_topk = 512` slots, so the
top-k is no longer a no-op. `torch.topk` lowers to `sort`, which neuronx-cc
rejects, so `layers.large_topk_mask` selects by bisecting the k-th largest
*value* instead — a fixed ~24 compare-and-count passes regardless of `top_k`,
versus the `top_k` passes `topk_mask` costs (fine for the MoE router's k=6,
hopeless for k=512). `_compact_mask_to_indices` then turns the mask into
fixed-width indices with a cumsum scatter, also sort-free.

Verified against `scores.topk` on CPU: identical selection (100% agreement,
exactly `top_k` entries per row) for streams from 513 up to 131072 slots, and
the lowered HLO contains no `sort`, `topk`, or `f64`. Two subtleties were
load-bearing — the bisection must bracket over *finite* scores only, because
callers mask disallowed slots to `finfo.min` and a 3e38-wide bracket never
converges; and the index tie-break must scale as `1 / num_slots`, or at 32k
slots it reorders genuinely distinct scores. **Not yet exercised on device**:
the 512 decode fault blocks reaching a context where the top-k engages.

At `max_model_len 512` all graphs compile and warmup completes. Two environment
variables are needed, and neither is the value its name suggests:

* `VLLM_NEURON_BARRIER_TIMEOUT` (default 3600) is what bounds warmup — *not*
  `--distributed-timeout-seconds`. Rank 0 compiles every graph serially while
  the other ranks sit in `tp_barrier`, so the barrier must outlast the whole
  compile. 512 needs ~2 h; an earlier report that "512 warmup is too slow" was
  this timeout, not kernel throughput.
* `VLLM_NEURON_PARALLEL_COMPILE_WORKERS` (default 8) caps compile concurrency.
  This host has 192 cores, so 24 is a better default.

### Known bug: gather out-of-bounds at 512 decode

`max_model_len 512` compiles and warms up, then the first real decode step
reports

```
TDRV: scatter/gather (indirect memory copy via scalar DGE) out-of-bound access
RuntimeError: nrta status=5
```

on all 64 ranks. It does not reproduce at 128. Two real defects were found and
fixed while chasing it (see below), but neither is the cause, and every
`index_select`/`gather` in the traced decode graph has been checked against its
operand extent — the clamps are all correct. At decode `T=1`, so no NKI kernel
runs (all three fall back to torch below their token thresholds), which narrows
it to a torch-lowered gather.

Setting `NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK=0` silences the
notification; whether the offending read is harmless (a masked row) or actually
corrupts output was not established. **Do not treat that flag as a fix.**

## Memory

Dequantizing to bf16 inflates the checkpoint's on-device footprint:

| Component            | Checkpoint | bf16 on device |
| -------------------- | ---------- | -------------- |
| Routed experts (FP4) | ~132 GiB   | ~528 GiB       |
| Other weights (FP8)  | ~23 GiB    | ~46 GiB        |
| **Total**            | ~155 GiB   | **~574 GiB**   |

A trn2.48xlarge has 1536 GiB of HBM (16 devices x 96 GiB), so the model fits at
TP=32 or above with room for KV cache — 17.9 GiB per rank at TP=32 and 9.0 GiB at
TP=64, against 48 GiB per logical core. Expert parallelism is strongly
recommended to keep the per-rank weight share reasonable. Serving the FP8 `-Base`
variant instead halves the expert footprint.

## Feature status

| Feature                 | Status | Notes                                          |
| ----------------------- | ------ | ---------------------------------------------- |
| TP (Q-head sharding)    | ✅     | Latent KV replicated (single head)             |
| EP (expert parallelism) | ✅     | `ep_degree` in `neuron_config`                 |
| SP (sequence parallel)  | ❌     | Hyper-connection streams not sharded           |
| Attention DP            | ❌     |                                                |
| Prefix caching          | ⚠️     | Window only; compressed state rebuilt          |
| Continuous batching     | ❌     | `--max-num-seqs 1` enforced (see above)        |
| Long context            | ❌     | 128 serves; 512 faults at decode (see above)   |
| Segmented prefill       | ❌     | Compressors need the whole prompt              |
| DSpark spec decode      | ❌     | `mtp.*` weights unused                         |
| FP8 KV cache            | ❌     | Latent KV is bf16                              |
| On-device sampling      | ✅     |                                                |
| MoE blockwise kernel    | ✅     | Prefill via `NF.moe_cte`; decode stays dense   |
| Sparse-attention kernel | ✅     | NKI for prefill; torch fallback for decode     |
| HC Sinkhorn kernel      | ✅     | NKI for prefill; torch fallback for decode     |

## Serving

```bash
export NEURON_SKIP_EFA_AFFINITY=1   # only on hosts without EFA

vllm serve /path/to/DeepSeek-V4-Flash-0731 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --additional-config '{"neuron_config": {"ep_degree": 64}}'
```

`--max-num-seqs 1` is enforced by the factory (the compressed streams are
per-request buffers) and raises a clear error rather than returning wrong
output. **128 is the only length that serves correctly today** — 512 compiles
and warms up but faults on the first decode step; see the known-bug section.

TP=64 / EP=64 uses the whole machine and is the fastest measured configuration;
TP=32 / EP=32 also works and halves the core count. See the parallelism-layout
section for what each degree partitions.

Longer contexts additionally need the barrier and compile-worker environment
variables from the compile-envelope section, or warmup times out waiting on
rank 0's serial compile.

## TODO

Ordered by what blocks what.

1. **Find the 512 decode gather out-of-bounds.** Blocks every context above
   128, and therefore blocks all quantitative accuracy work. Ruled out so far:
   every `index_select`/`gather` clamp in the traced decode graph (all correct
   against their operand extents), the NKI kernels (none run at `T=1`), the
   sharded embedding (its shard mask is sound), and `tid2eid`. Next: bisect by
   disabling layer groups, or dump the runtime's faulting DMA descriptor rather
   than reasoning from the graph. Do not ship
   `NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK=0` as a workaround without first
   proving the read is harmless.
2. **Benchmark accuracy on GSM8K.** The harness is written and the dataset is
   local, but GSM8K needs ~200 tokens median (only 14/150 problems fit in 128),
   so it cannot run until (1) is fixed. A prior NxD run of the same checkpoint
   scored 97.3% (146/150) at TPOT 93 ms — directly comparable once this runs.
   Current accuracy evidence is qualitative only: 6 prompts correct, and
   identical tokens across 5 configurations (torch / NKI / hybrid / TP=32 /
   TP=64).
3. **Exercise `large_topk_mask` on device.** Verified exact against
   `scores.topk` on CPU up to 131072 slots and confirmed sort-free in the
   lowered HLO, but no context above 2048 has run on hardware, so its
   instruction cost and numerics there are unmeasured.
4. **Batch the NKI attention kernel's token loop.** It walks tokens with
   `nl.sequential_range`, so a long prefill is that many serial passes. Tiling
   tokens onto the partition axis alongside `head_dim` is the fix. This is a
   throughput item, not a correctness one.
5. **Widen the CPU reference comparison.** The 0.99997 hidden-state cosine
   covers 6 of 43 layers and 8 of 256 experts. A full-depth, full-expert
   comparison has never run.
6. Fold the compressed streams into the paged allocator (needs `LayerSpec` to
   express a per-group token-to-slot stride), which would lift
   `--max-num-seqs 1`; and implement DSpark speculative decoding (`mtp.*`
   weights are loaded but unused).
