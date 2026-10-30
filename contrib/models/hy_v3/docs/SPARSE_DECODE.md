# Block-sparse decode attention (experimental, off)

At 85k context, attention is 55% of an 8-request MTP decode step and 27% of a
single-request one ([PROFILE_85K.md](PROFILE_85K.md)); most of it is reading
the KV history. This study asks whether decode can attend to a fraction of
the history without losing accuracy, before building a fast implementation.
Status: the study below found no accuracy loss at 12.5% of the blocks. The
implementation that reads only the kept blocks
([Implementation](#implementation)) keeps that accuracy on the full model
with FP8 KV + MTP (LongBench v2 55% vs 56%, 97/100 same answers, needle
12/12) and makes 8-request decode steps **14% faster** (ITL 76.7 -> 66.2 ms,
TPOT 51.0 -> 47.4 ms); 4 requests break even and 1 request is 7% slower (it
stays dense but keeps the summaries up to date). Off by default.

## Accuracy study

The study implementation (commit 80c4bdb, since replaced) read the whole K
every step: `HY3_SPARSE_DECODE=<fraction>`, decode
only (prefill stays dense), BF16 KV, no DCP, one token per request per step:

1. In every layer the decode query is computed as the decode kernel does it
   (RMSNorm, STATIC FP8 QKV projection, QK-norm, RoPE).
2. Every 32-token KV block of the request gets the Quest upper bound of
   `q . k` over its tokens: `sum_d max(q_d min_d, q_d max_d)` from the block's
   per-dimension K min / max (each rank holds one KV head and selects for it).
3. `ceil(fraction x blocks)` blocks are kept per request: the first block
   (`HY3_SPARSE_SINK_BLOCKS`, 1) and the last 4 (`HY3_SPARSE_RECENT_BLOCKS`)
   always, the rest by bound.
4. The kept blocks are moved to the front of the block table in their original
   order and the history length is set to the tokens they hold, so the
   unchanged decode megakernel computes exact attention over them.

Evaluation (`scripts/sparse_decode_study.sh`, which now runs the
implementation below: max_model_len 86016, prefix
caching, chunked 4096-token prefill, one request at a time; each fraction is
its own server, the dense run is the reference):

- **needle**: `scripts/needle.py`, passkey at depth 0.1 / 0.5 / 0.9 of 16k /
  32k / 64k / 85k-token prompts (12 cases);
- **continuations**: `scripts/sparse_decode_eval.py`, greedy 64-token
  continuations of 19 long real-text prompts (11 x 85k with a 78k shared
  prefix, 4 x 32k, 4 x 64k ShareGPT), compared to dense;
- **LongBench v2**: `scripts/longbench_v2.py`, 100 multiple-choice items
  (THUDM/LongBench-v2, official no-CoT template) of 10k-81k tokens (median
  30k), greedy; accuracy and the share of items answered like dense;
- **control**: a fraction of 1e-6 keeps only the sink and the 4 recent blocks
  (no selection by the bound).

## Results

| decode attends to | needle | continuations identical to dense | LongBench v2 accuracy | LongBench v2 same answer as dense |
|---|---|---|---|---|
| all blocks (dense) | 12/12 | (reference) | 57% | (reference) |
| 50% of the blocks | 12/12 | 18/19 | 59% | 98% |
| 25% | 12/12 | 19/19 | 60% | 97% |
| **12.5%** | **12/12** | **19/19** | **60%** | **97%** |
| sink + 4 recent blocks only (control) | **0/12** | 4/19 (25% common prefix) | | |

Raw outputs: `results/sparse_decode/`.

- At 12.5% of the blocks (4k of a 32k history, 10.6k of 85k) nothing measured
  drops: all needles found, every 64-token continuation identical to dense,
  and LongBench v2 answers differ on 3 of 100 items, all three from a wrong
  dense answer to the right one (9 items have no parsable answer in any run).
- The control shows that the decode kernel really reads only the compacted
  blocks (no needle, 4/19 identical continuations), so the selection by the
  bound is what keeps the accuracy. In the control the first digits of a key
  are often right: the first answer token comes from the (dense) prefill.
- The one different continuation at 50% (an open-ended 85k ShareGPT prompt,
  diverging after 81 characters) is not repeated at 25% or 12.5%: run-to-run
  variation of the near-tie kind seen elsewhere, not a trend.

Limits: decode only (the prefill, where the needle is read in, is dense); one
model, these task types (retrieval, continuation, long-document multiple
choice); generations of 16-64 tokens; BF16 KV (FP8 packed K would need the
summaries from the unpacked values).

## Implementation

`HY3_SPARSE_DECODE=<fraction>` (src/model.py; no DCP), decode only:

| step | study | implementation |
|---|---|---|
| K min / max | per 32-token block, gathered from the K cache every step | per group of `HY3_SPARSE_GROUP` blocks (4: 128 tokens), `k_sum` [blocks, 2, kv heads, 128] BF16 next to the KV cache, at the row of the group's first block (`bind_attention_kv`); written by prefill (`update_k_summary_prefill`, from the K before FP8 quantization) and by every decode step (`update_k_summary_decode`); 1/16 of the FP8 KV bytes |
| query (and K) for the bound | the kernel's | computed again in torch (`_sparse_decode_blocks`) from own copies of the Q and K weight columns (dequantized to BF16) and the norm gammas, with RoPE from the positions (`prepare_sparse_decode`) |
| kept | a fraction of each request's blocks | `sparse_budget_blocks`: the fraction of the decode context bucket, rounded up to a multiple of 256 blocks (0.125 of 2688 -> 512 blocks = 128 groups, 16k tokens): the first group and the last 2 (`HY3_SPARSE_SINK_GROUPS`, `HY3_SPARSE_RECENT_GROUPS`) and the rest by bound (top-k) |
| decode attention | full-width table, masked | the kept groups' blocks as a [B, 512] block table and `s_prior` 16384: the kernel reads only them |

With MTP the bound is the max over the 4 verify tokens, and the draft layer
selects for its own cache too (`HY3_SPARSE_DRAFT=0` turns that off). Steps
of fewer than `HY3_SPARSE_MIN_BATCH` (4) requests attend densely (the decode
NEFFs are per batch bucket): the selection costs about the same at any batch
size, the attention it saves grows with it. They still update the summaries.

Why groups: the selection is a handful of small torch ops per layer whose
cost is mostly fixed. `scripts/sparse_select_bench.py` times them on one
NeuronCore at the full-model per-rank shapes (8 requests x 4 verify tokens,
2688 blocks):

| selection (`sparse_select_blocks` and its parts) | time |
|---|---|
| launch + sync of a trivial graph | 0.10 ms |
| per block (group 1, the first version): summary gather / + bound / + top-511 / + sort | 0.24 / 0.27 / 0.86 / 0.98 ms |
| per block, top-k replaced by a 16-step threshold bisection, + scatter or compare compaction | 0.57, + 1.8 or + 0.3 ms |
| per block, whole function | 1.07 ms |
| **groups of 4 (672 groups, top-127), whole function** | **0.35 ms** |

By the 85k profile, attending to 1/5 of the blocks saves ~0.46 ms per layer
at 8 requests, ~0.23 ms at 4 and ~0.06 ms at 1.

Pitfalls found on the way (covered by `test/unit/test_sparse_decode.py` or
comments in the code):

- **Torch ops reading tensors that also feed the decode megakernel** (the
  cos / sin, the QKV weight, the norm gammas) gave wrong attention results;
  hence the own weight copies and RoPE.
- **The kernel reads a step's earlier verify tokens back through the block
  table**: when the 4 MTP tokens cross a block boundary, the block after the
  history must be in the compacted table. At most `n_sel / group - 1` groups
  are kept for that. Without it, LongBench v2 lost 5 of 78 items (3 of them
  unparsable answers), while the needle test still passed.
- **Recycled blocks keep the summary of their previous request**: with
  summaries written at prefill only (the first version), blocks decode wrote
  looked like valid history and could be dropped once an output outgrew the
  forced recent blocks. Decode now folds its K into the summaries (a step
  starting a group restarts it; rejected verify tokens only loosen the bound).
- A block-table width that is not a multiple of 256 makes the kernel split
  the 32-token blocks, which the compacted table does not survive.
- HBM: the summaries (~0.9 GB per rank at 697k tokens) do not fit next to
  the 32768-bucket decode NEFFs (`nrt_tensor_allocate` at runtime), so
  `serve_85k.sh` drops that bucket when `HY3_SPARSE_DECODE` is set.
- The 4-layer mini model (`hy3_mini4_mtp`) does not show such bugs in greedy
  tokens: with MTP even a recompiled dense run flips near-tie tokens there.
  Compare captured activations (without MTP) or use the full model.

### Results (full model, `serve_85k.sh` FP8 KV + MTP k=3, 0.125)

Accuracy (`results/sparse_decode/lb_full_fp8mtp_*.json`, needle logs):

| | LongBench v2 | same answer as dense | needle |
|---|---|---|---|
| dense | 56% | | |
| first version (per block, prefill summaries) | 55% | 95/100 | 12/12 (before the verify-token fix) |
| **groups + decode summaries** | **55%** | **97/100** | **12/12** |

The differing items are coherent near-ties ("The correct answer is (B)");
8 unparsable answers in every run.

Speed: P/D split, `bench_pd_split.sh "1 4 8"` (85k prompts with a 78k shared
prefix, 256 output tokens, pass 2; `results/bench/pd_split_85k_nodcp_fp8kv_mtp3`,
`..._sparse` for the first version, `..._sparse_g4`):

| concurrency | ITL dense | first version | groups | TPOT dense | first version | groups | output tok/s dense / groups |
|---|---|---|---|---|---|---|---|
| 1 | 47.4 ms | 100.5 ms | 50.8 ms | 23.4 ms | 49.4 ms | 24.9 ms | 37.9 / 35.7 |
| 4 | 66.6 ms | 128.1 ms | 66.9 ms | 39.2 ms | 69.2 ms | 39.3 ms | 92.0 / 91.5 |
| 8 | 76.7 ms | 137.1 ms | **66.2 ms** | 51.0 ms | 81.5 ms | **47.4 ms** | 138.3 / **150.5** |

MTP accepts 2.0 tokens per step in every run. TPOT gains less than ITL at 4
and 8 requests because it also carries the (dense) prefill steps of arriving
requests that interrupt decode (p99 ITL ~600 ms in all runs). At 1 request
the step stays dense; the 3.4 ms are the Q / K projection and the summary
update in 80 + 3 layer calls. The 8-request step saves ~10 ms of the ~20 ms
the single-core timings predict: the selection costs more inside the model
graph than alone. Next steps: profile the sparse step, and an NKI kernel for
the selection and the summary update.
