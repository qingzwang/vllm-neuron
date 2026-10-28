# Block-sparse decode attention: accuracy study (experimental)

At 85k context, attention is 55% of an 8-request MTP decode step and 27% of a
single-request one ([PROFILE_85K.md](PROFILE_85K.md)); most of it is reading
the KV history. This study asks whether decode can attend to a fraction of
the history without losing accuracy, before building a fast implementation.
Status: **accuracy measured; no speed-up yet** (the study implementation
reads the whole K every step). Off by default.

## Method

`HY3_SPARSE_DECODE=<fraction>` (src/model.py, `sparse_keep_blocks`), decode
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

`test/unit/test_sparse_decode.py` checks the selection and the compaction on
CPU (fraction 1 is dense; the block holding a matching key is kept; the
compacted table reads exactly the kept tokens).

Evaluation (`scripts/sparse_decode_study.sh`: max_model_len 86016, prefix
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

## Towards a fast implementation

What the study does every step, a real implementation keeps or avoids:

| step | study | fast version |
|---|---|---|
| per-block K min / max | gathers every K block of the request | stored per block (2 x 128 values per block and KV head, 1/16 of K) and updated at KV write (prefill blocks, decode token) |
| bound | over the gathered summary | over the stored summaries: reads 1/16 of the K bytes |
| decode attention | full-length context bucket, masked | a context bucket sized to the kept tokens (e.g. 16384 for 12.5% of 86016): the kernel reads only those blocks |

The last row needs the runner to pick the decode context bucket from the kept
tokens instead of the context length (it otherwise slices the block table to
the context bucket), so it touches the framework. Expected effect from the
85k profile (attention grows ~5.8 ms per 85k-token request per step): at 8
requests with MTP the attention part would drop from ~42 to ~15-20 ms of a
76.6 ms step (~-30%); at one request ~-9%. With MTP the selection takes the
union over the 4 verify tokens.
