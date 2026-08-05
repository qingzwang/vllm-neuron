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
- **Attention is a gather-based sparse kernel in torch.** The NKI flash-attention
  kernel caps `head_dim` at 128; MLA needs 512.
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

Three neuronx-cc 2.26 limitations shape the implementation. All are load-bearing
— reverting any of them reproduces a compile failure:

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

## Hardware status

Compiles and generates on trn2 (`tensor_parallel_size=1`, synthetic checkpoint)
for all three attention layer types — sliding-window-only, CSA and HCA — with
FP4 experts and hash routing. End-to-end serving of the full 43-layer 284B
checkpoint has not been run; see the memory budget below for the TP degree it
requires.

## Memory

Dequantizing to bf16 inflates the checkpoint's on-device footprint:

| Component            | Checkpoint | bf16 on device |
| -------------------- | ---------- | -------------- |
| Routed experts (FP4) | ~132 GiB   | ~528 GiB       |
| Other weights (FP8)  | ~23 GiB    | ~46 GiB        |
| **Total**            | ~155 GiB   | **~574 GiB**   |

A trn2.48xlarge has 1536 GiB of HBM (16 devices x 96 GiB), so the model fits at
TP=32 or above with room for KV cache — but expert parallelism is strongly
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
| Long context on CSA     | ⚠️     | `--max-model-len <= index_topk * 4` = 2048     |
| Segmented prefill       | ❌     | Compressors need the whole prompt              |
| DSpark spec decode      | ❌     | `mtp.*` weights unused                         |
| FP8 KV cache            | ❌     | Latent KV is bf16                              |
| On-device sampling      | ✅     |                                                |
| MoE NKI kernels         | ❌     | Dense-masked dispatch; see `moe.py` TODO       |

## Serving

```bash
vllm serve /path/to/DeepSeek-V4-Flash-0731 \
    --tensor-parallel-size 32 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --additional-config '{"neuron_config": {"ep_degree": 8}}'
```

`--max-num-seqs 1` and `--max-model-len <= 2048` are enforced by the factory for
the reasons given above; both raise a clear error rather than miscompiling or
returning wrong output.
