# Contrib Model: Hy3 (tencent/Hy3-FP8)

Serves [tencent/Hy3-FP8](https://huggingface.co/tencent/Hy3-FP8) with vLLM on
AWS Trainium through the `vllm-neuron` plugin. Hy3 is a ~295B-parameter MoE
decoder (80 layers, 192 routed experts top-8 + 1 shared expert, GQA 64Q / 8KV)
with a static per-tensor FP8 checkpoint. All linear layers run with the
checkpoint's FP8 weights; the KV cache is BF16.

The architecture, kernel mapping and feature status are in
[src/README.md](src/README.md).

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

The model needs the full trn2.48xlarge (TP=64 over 64 logical NeuronCores,
`--enable-expert-parallel`, full EP=64).

### Install / register

vLLM already has a GPU implementation registered as `HYV3ForCausalLM`, and the
vLLM engine core and the Neuron workers run in spawned processes, so the Neuron
implementation has to be registered in every process. This folder is a small
Python package that does that through a `vllm.general_plugins` entry point:

```bash
pip install -e contrib/models/hy_v3     # imports as `hy_v3_neuron`
```

After that, both `vllm serve` and the offline `LLM(...)` API pick up the Neuron
implementation with no further code. The plugin also adds `"fp8"` (the
checkpoint's `quant_method`) to the Neuron platform's supported quantizations;
the model consumes the FP8 weights and scales itself. Nothing outside this
folder is modified.

### Serve with the OpenAI-compatible server

`scripts/serve.sh` and `scripts/serve_latency.sh` wrap the configurations used
for the results below. A minimal launch:

```bash
export NEURON_SKIP_EFA_AFFINITY=1          # only on instances without EFA
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.45
vllm serve tencent/Hy3-FP8 \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 2048 \
    --block-size 32 \
    --enable-prefix-caching \
    --additional-config '{"neuron_config": {
        "num_batched_tokens_buckets": [2048],
        "kv_segment_size_buckets": [2048],
        "num_seqs_buckets": [1, 4, 16, 32],
        "decode_context_length_buckets": [4096, 8192]}}'
```

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "hy3", "messages": [{"role": "user", "content": "What is 17*23?"}],
         "max_tokens": 64}'
```

The first start compiles the model (about 50 minutes on a cold cache).

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="tencent/Hy3-FP8",
    tensor_parallel_size=64,
    enable_expert_parallel=True,
    max_model_len=4096,
    max_num_seqs=4,
    block_size=32,
)
outputs = llm.generate(["The capital of France is"], SamplingParams(max_tokens=32))
print(outputs[0].outputs[0].text)
```

### Configuration notes

- Keep `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` at 0.45 or lower. Some NEFFs
  are loaded lazily on first use and need device memory for their DMA rings;
  at 0.6 the server ran out of memory after about 40 minutes.
- `max_model_len` is always an implicit decode context bucket, so
  `decode_context_length_buckets` must stay strictly below it.
- Set `decode_context_length_buckets` whenever `max_model_len` is well above
  the typical context. Without it each decode batch bucket is a single NEFF
  that reads `max_model_len` worth of KV blocks per request on every step,
  whatever the request's real length (vllm-neuron 0.21+ makes this opt-in;
  the NxDI-based 0.2.x bucketed decode by sequence length by default). The
  configs below all set it.
- Keep every decode context bucket (and `max_model_len`, the implicit last
  one) a multiple of 256 whose quotient by 256 shares a large factor with 32,
  e.g. 8192, 10240, 12288, 16384. The decode kernel resizes the cache block to
  `gcd(32, bucket / 256)`; 8448 or 9472 give block length 1, one DMA per KV
  row, which cost up to 33% TPOT at 8K context (see "8K in / 1K out at high
  concurrency"). The kernel prints "reducing block length by ...x" per bucket.
- Not supported: data parallel / attention DP, MTP together with DCP, an FP8
  KV cache together with DCP. They are rejected at startup.
- MTP speculative decoding (`--speculative-config '{"method": "mtp",
  "num_speculative_tokens": 3}'`, `MTP_K=3` for the serve scripts) uses the
  checkpoint's MTP layer as the draft. It needs a vllm-neuron with the MTP
  proposer (`vllm_neuron/vllm/spec_decode/mtp.py`). With `moe_tkg_hits` it
  cuts TPOT at concurrency 1 from 26.3 to 13.8 ms and triples throughput at
  concurrency 4; see "MTP speculative decoding" under Results.
- `--decode-context-parallel-size 8 --cp-kv-cache-interleave-size 32` shards
  the replicated KV heads 8 ways (8x KV capacity) for many concurrent
  long-context requests. On 0.29 it is validated on the 4-layer model
  (`scripts/validate_features.py <ckpt> <refs> 8`: logits cos >= 0.9988 with
  segmented prefill and prefix-cache hits); its full-model performance was
  measured only on 0.24, where it cost ~22% TPOT and ~30% prefill speed at
  equal concurrency. See [src/README.md](src/README.md).
- Decode steps of up to 4 tokens per 3 local experts (EP=64: 4, EP=32: 8)
  compute only the routed experts that live on each rank
  (`src/kernels/moe_tkg_hits.py` for one token, `moe_tkg_hits_batch.py` for
  more; on by default). Larger steps keep the all-expert `moe_tkg`.
  `HY3_DECODE_HITS_MAX_T` changes the limit and `HY3_DECODE_HITS=0` turns
  both paths off. See "Decode: only the routed experts" under Results. With
  MTP, raise the limit to the verify step's size (max-num-seqs x (k + 1));
  `scripts/serve_latency.sh` does this when `MTP_K` is set.
- `"ep_degree": 32` (`EP=32` for `scripts/serve.sh` and
  `scripts/serve_latency.sh`) trades TTFT for TPOT. In config B it gives
  0.4-1.8 ms lower TPOT at concurrency 1 and 4, but 3-6% higher TPOT at
  concurrency 15 and 10-22% higher TTFT everywhere
  (`results/bench/B_multibucket_ep32/`). EP=64 stays the default.
- Below EP=32 the prefill MoE kernel (`moe_cte` `shard_on_i_hybrid`) does
  not compile on neuronx-cc 2.28. `HY3_MOE_CTE_IMPL=shard_on_i` makes EP=16 /
  8 work, at about 2.5x the TTFT and no TPOT gain over EP=32.
- On-device sampling is on by default. If every request is greedy,
  `"on_device_sampling_config": {"all_greedy": true}` swaps the general
  sampler for an argmax and saves about 0.5 ms per token; requests that set
  a temperature are then decoded greedily too.

## Compatibility Matrix

| Instance / software | vllm-neuron 0.29.0.1.2.0 (vLLM 0.29.0) | earlier |
| ------------------- | -------------------------------------- | ------- |
| Trn3                | Not tested                             | Not tested |
| Trn2 (trn2.48xlarge) | Working                               | Not tested with this folder |

Tested component versions: neuronx-cc 2.28.7631, nki 0.7.0,
libtorch-neuronx-lite 2.13.0.1.0, Neuron runtime 2.35.30, collectives 2.35.43,
driver 2.31.7, torch 2.13.0, transformers 5.17.0, Python 3.12.

## Example Checkpoints

- <https://huggingface.co/tencent/Hy3-FP8>

## Results

All numbers: trn2.48xlarge, TP=64, full EP=64 unless noted, FP8 weights, BF16 KV cache,
`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.45`, on-device sampling, block size 32.

### Accuracy

| Test | Result |
| --- | --- |
| MATH-500 no_think (full model, temp 0.9, max 4096 tokens, config B) | **97.0%** (485/500), 3 truncated |
| 4-layer real-weight model vs HF FP32, flash prefill (`scripts/compare_hf_logits.py`) | logits cosine >= 0.9993; greedy splits only at near-ties (top-2 gap <= 0.14 nats) |
| same, segmented prefill 5000 / 3100-token prompts, cold and prefix-cache hit (4992 / 3072 tokens cached) (`scripts/validate_features.py`) | cosine >= 0.9989, identical cold vs cache hit |
| batch 128 vs batch 4 decode | greedy identical, cosine >= 0.9993 |
| DCP=8, same checks (`validate_features.py ... 8`) | cosine >= 0.9988; greedy splits only at near-ties (gap 0.011-0.18 nats) |
| Integration test (`test/integration/test_model.py`, both cases) | passed: top-K logit error <= 0.06 vs the BF16 golden; batch-1 case (`moe_tkg_hits`) <= 0.045, batch-4 case (`moe_tkg_hits_batch`) <= 0.057 |
| Module tests (`test/module/`) | routed experts rel. error ~3e-3 (`moe_tkg_hits`, 1-3 local hits: 2.5-3.3e-3, 0 hits: exactly 0; `moe_tkg_hits_batch`, 2 / 4 tokens: 2.9e-3); MLP: BF16-activation (TKG) paths ~2e-3, FP8-activation (CTE) paths within 1-2% of an ideal FP8 quantizer |

### Performance

`vllm bench serve`, random dataset, 256 output tokens (`ignore_eos`),
4 x concurrency requests per cell (min 8), a distinct seed per cell
(`scripts/bench_matrix.sh`). TPOT includes the decode stalls caused by other
requests' prefills (prefill and decode never share a step on Neuron); ITL
median is close to the pure decode step time. Raw JSON is in `results/bench/`.

**Recommended: prefix caching + several prefill buckets** (`results/bench/B_multibucket/`):

```bash
PREFILL_BUCKETS='[512, 1024, 2048, 4096, 8192]' MODEL=<ckpt> \
    scripts/serve.sh 16384 32 '[1, 4, 16, 32]' '[4096, 8192]'
```

| input | conc | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | ITL med (ms) | out tok/s | peak out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.23 | 0.25 | 22.2 | 21.7 | 44 | 47 | 218 |
| 1024 | 4 | 0.54 | 0.85 | 34.1 | 32.7 | 111 | 124 | 554 |
| 1024 | 15 | 1.59 | 2.94 | 44.9 | 39.4 | 295 | 390 | 1473 |
| 4096 | 1 | 0.57 | 0.58 | 22.2 | 21.9 | 41 | 46 | 697 |
| 4096 | 4 | 1.37 | 2.17 | 37.8 | 34.5 | 93 | 120 | 1583 |
| 4096 | 15 | 4.26 | 7.96 | 56.9 | 42.2 | 205 | 360 | 3479 |
| 8192 | 1 | 1.01 | 1.02 | 23.7 | 23.5 | 36 | 43 | 1198 |
| 8192 | 4 | 2.45 | 3.90 | 42.8 | 36.9 | 77 | 112 | 2528 |
| 8192 | 15 | 7.71 | 14.42 | 78.2 | 51.8 | 139 | 300 | 4582 |
| 16000 | 1 | 2.05 | 2.06 | 23.8 | 23.5 | 32 | 43 | 2007 |
| 16000 | 4 | 4.99 | 7.94 | 48.6 | 36.8 | 59 | 112 | 3746 |
| 16000 | 15 | 15.79 | 29.51 | 105.7 | 51.7 | 90 | 300 | 5707 |

**High concurrency, short context** (max_model_len 2048, 128 sequences,
prefill buckets 512 / 1024 / 2048; `results/bench/C_multibucket/`):

```bash
PREFILL_BUCKETS='[512, 1024, 2048]' MODEL=<ckpt> \
    scripts/serve.sh 2048 128 '[1, 4, 16, 32, 64, 128]' '[1024]'
```

| input | conc | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | ITL med (ms) | out tok/s | peak out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 512 | 32 | 2.37 | 4.44 | 45.0 | 36.4 | 593 | 896 | 1779 |
| 512 | 63 | 4.48 | 8.76 | 61.3 | 44.2 | 802 | 1449 | 2405 |
| 512 | 127 | 8.89 | 17.49 | 95.7 | 61.1 | 976 | 2159 | 2927 |
| 1024 | 32 | 3.23 | 6.20 | 51.1 | 39.3 | 504 | 832 | 2518 |
| 1024 | 63 | 6.19 | 12.13 | 71.1 | 47.5 | 663 | 1380 | 3314 |
| 1024 | 127 | 12.31 | 24.24 | 116.5 | 68.6 | 773 | 1905 | 3866 |

**Prefix-cache hit rate** (same server as the recommended config;
`scripts/bench_prefix_cache.sh "8192 16000" "0 25 50 75 90" "1 8"`,
`results/bench/prefix_cache/`). Each cell sends 16 requests that share one
random prefix of the given share of the input; the first request fills the
cache. The hit rate is measured from the server's `vllm:prefix_cache_hits` /
`vllm:prefix_cache_queries` counters: prefix share x 15/16, rounded down to
32-token blocks.

| input | conc | shared prefix | measured hit rate | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 0% | 0.0% | 1.01 | 1.05 | 27.7 | 32 | 1044 |
| 8192 | 1 | 25% | 23.4% | 1.02 | 1.03 | 27.7 | 32 | 1044 |
| 8192 | 1 | 50% | 46.9% | 0.61 | 0.96 | 27.7 | 33 | 1096 |
| 8192 | 1 | 75% | 70.3% | 0.39 | 0.96 | 27.7 | 34 | 1124 |
| 8192 | 1 | 90% | 84.2% | 0.29 | 0.92 | 27.7 | 35 | 1140 |
| 8192 | 8 | 0% | 0.0% | 4.36 | 7.72 | 65.1 | 98 | 3225 |
| 8192 | 8 | 25% | 23.4% | 4.39 | 7.76 | 65.1 | 98 | 3220 |
| 8192 | 8 | 50% | 46.9% | 2.76 | 4.86 | 59.5 | 114 | 3767 |
| 8192 | 8 | 75% | 70.3% | 1.88 | 3.32 | 56.5 | 126 | 4152 |
| 8192 | 8 | 90% | 84.2% | 1.49 | 2.63 | 55.0 | 132 | 4357 |
| 16000 | 1 | 0% | 0.0% | 2.06 | 2.08 | 27.7 | 28 | 1782 |
| 16000 | 1 | 25% | 23.4% | 1.66 | 2.00 | 27.7 | 29 | 1857 |
| 16000 | 1 | 50% | 46.9% | 1.12 | 1.92 | 27.7 | 31 | 1972 |
| 16000 | 1 | 75% | 70.3% | 0.70 | 1.85 | 27.7 | 33 | 2071 |
| 16000 | 1 | 90% | 84.4% | 0.48 | 1.83 | 27.7 | 34 | 2127 |
| 16000 | 8 | 0% | 0.0% | 8.94 | 15.79 | 78.7 | 71 | 4484 |
| 16000 | 8 | 25% | 23.4% | 7.38 | 13.04 | 73.5 | 78 | 4978 |
| 16000 | 8 | 50% | 46.9% | 5.21 | 9.11 | 66.0 | 93 | 5901 |
| 16000 | 8 | 75% | 70.3% | 3.51 | 6.22 | 60.2 | 109 | 6894 |
| 16000 | 8 | 90% | 84.4% | 2.62 | 4.67 | 57.1 | 119 | 7577 |

- At an 84% hit rate a 16k prompt's median TTFT drops from 2.06 s to 0.48 s
  (4.3x). P99 stays near the cold value because each cell's first request
  misses.
- At concurrency 8, shorter prefills also stall decode less: 16k TPOT drops from
  78.7 to 57.1 ms and total throughput rises from 4484 to 7577 tok/s.
- The uncached remainder is still padded to a prefill bucket. With 8k input
  and a 25% prefix, the remaining 6144 tokens pad to the 8192 bucket, so there
  is no gain. Denser buckets (subject to compile time) close such gaps.

**Single request, lowest latency** (`scripts/serve_latency.sh`: no prefix
caching, prefill buckets 512..16384 so a 16k prompt is one prefill step;
`results/bench/opt_latency/`):

| input | TTFT med | TPOT med |
| --- | --- | --- |
| 1024 | 0.22 s | 22.1 ms |
| 4096 | 0.54 s | 22.4 ms |
| 8192 | 0.97 s | 23.6 ms |
| 16000 | 1.83 s | 23.5 ms |

The multi-bucket prefix-caching config is within 3-12% of this TTFT and
about the same TPOT (within 2%), so it is the better default.

### Decode: only the routed experts (`moe_tkg_hits`, `moe_tkg_hits_batch`)

In a batch-1 decode step at EP=64 the routed experts took 45% of the device
time (`moe_tkg` all-expert, 12.6 of 27.8 ms). Every rank computes its 3 local
experts although a token's 8 experts hit at most one or two of them on any
rank. The step is bound by the weights read, not by FLOPs: a one-token
matmul streams its weight tiles through the PE array at about 180 GB/s per
physical core, so each expert computed costs its 18.9 MB of FP8 weights.
`moe_tkg`'s selective mode does not help: it unrolls all 8 top-k slots, and
the non-local ones run on a dummy expert with affinity 0.

`src/kernels/moe_tkg_hits.py` (forked from nkilib's `selective_expert_impl`,
Apache 2.0) takes the top-k slots with this rank's experts packed first plus
their count, and loops over that count at runtime (`nl.fori_loop` on a
register). A rank with no routed expert does no expert work; the MoE
all-reduce then waits for the busiest rank, which at EP=64 usually has one
expert and sometimes two. The runtime loop adds about 30 us of control-flow
barriers per layer.

`src/kernels/moe_tkg_hits_batch.py` (from nkilib's `all_expert_impl`) does the
same for decode steps of 2 to 4 tokens per 3 local experts (EP=64: up to 4,
EP=32: up to 8). The model packs this rank's experts that any token in the
step routed to first, reorders their affinity columns the same way, and the
kernel reads each hit expert once for the whole batch, scaling its output
per token (0 for tokens that did not route to it). Past that batch size the
busiest rank hits all its experts in most layers, where the runtime loop is
slower than `moe_tkg`'s unrolled all-expert path, so larger steps keep
`moe_tkg`. `HY3_DECODE_HITS_MAX_T` overrides the limit; `HY3_DECODE_HITS=0`
turns both paths off.

Latency config (`scripts/serve_latency.sh`), c=1, `moe_tkg` -> `moe_tkg_hits`
(`results/bench/opt_latency/`, `results/bench/opt_latency_ep32/`):

| input | TTFT, EP=64 | TPOT, EP=64 | TTFT, EP=32 | TPOT, EP=32 |
| --- | --- | --- | --- | --- |
| 1024 | 0.21 -> 0.22 s | 26.3 -> 22.1 ms | 0.26 s | 21.1 ms |
| 4096 | 0.54 -> 0.54 s | 26.3 -> 22.4 ms | 0.65 s | 20.9 ms |
| 8192 | 0.97 -> 0.97 s | 27.7 -> 23.6 ms | 1.08 s | 22.3 ms |
| 16000 | 1.83 -> 1.83 s | 27.7 -> 23.5 ms | 1.88 s | 22.2 ms |

Recommended config B (`scripts/serve.sh`, see above), `HY3_DECODE_HITS=0` ->
default (`results/bench/B_multibucket_moe_tkg/`, `results/bench/B_multibucket/`).
The two runs are on two trn2.48xlarge instances; the concurrency-15 cells,
which the change does not touch, agree within 0.2%. Concurrency 1 gets 14-15%
lower TPOT. At concurrency 4 the
decode steps have 4 tokens, and the 4-token step itself (ITL median) drops
from 42-50 ms to 33-37 ms: TPOT -11% to -35%, output throughput up to +50%.
Concurrency 15 decodes in the 16-token bucket and is unchanged:

| input | conc | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | ITL med (ms) | out tok/s | peak out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.22 -> 0.23 | 0.23 -> 0.25 | 26.3 -> 22.2 | 26.3 -> 21.7 | 37 -> 44 | 39 -> 47 | 185 -> 218 |
| 1024 | 4 | 0.54 -> 0.54 | 0.84 -> 0.85 | 52.1 -> 34.1 | 50.2 -> 32.7 | 74 -> 111 | 84 -> 124 | 370 -> 554 |
| 1024 | 15 | 1.59 -> 1.59 | 2.95 -> 2.94 | 44.9 -> 44.9 | 39.4 -> 39.4 | 295 -> 295 | 390 -> 390 | 1473 -> 1473 |
| 4096 | 1 | 0.56 -> 0.57 | 0.57 -> 0.58 | 26.3 -> 22.2 | 26.4 -> 21.9 | 35 -> 41 | 38 -> 46 | 597 -> 697 |
| 4096 | 4 | 1.38 -> 1.37 | 2.19 -> 2.17 | 53.2 -> 37.8 | 49.2 -> 34.5 | 69 -> 93 | 88 -> 120 | 1166 -> 1583 |
| 4096 | 15 | 4.26 -> 4.26 | 7.96 -> 7.96 | 56.9 -> 56.9 | 42.2 -> 42.2 | 205 -> 205 | 360 -> 360 | 3478 -> 3479 |
| 8192 | 1 | 1.01 -> 1.01 | 1.03 -> 1.02 | 27.7 -> 23.7 | 27.8 -> 23.5 | 32 -> 36 | 37 -> 43 | 1046 -> 1198 |
| 8192 | 4 | 2.45 -> 2.45 | 3.90 -> 3.90 | 55.4 -> 42.8 | 48.4 -> 36.9 | 62 -> 77 | 96 -> 112 | 2036 -> 2528 |
| 8192 | 15 | 7.71 -> 7.71 | 14.44 -> 14.42 | 78.3 -> 78.2 | 51.7 -> 51.8 | 139 -> 139 | 300 -> 300 | 4579 -> 4582 |
| 16000 | 1 | 2.06 -> 2.05 | 2.07 -> 2.06 | 27.7 -> 23.8 | 27.8 -> 23.5 | 28 -> 32 | 36 -> 43 | 1785 -> 2007 |
| 16000 | 4 | 5.00 -> 4.99 | 7.96 -> 7.94 | 54.5 -> 48.6 | 42.0 -> 36.8 | 54 -> 59 | 108 -> 112 | 3461 -> 3746 |
| 16000 | 15 | 15.77 -> 15.79 | 29.49 -> 29.51 | 105.6 -> 105.7 | 51.7 -> 51.7 | 90 -> 90 | 300 -> 300 | 5710 -> 5707 |

EP degree, small config (2048 context, batch 1, prefill buckets 1024 /
2048, `all_greedy`), 1024-token prompts:

| EP | prefill MoE | decode MoE | TTFT | TPOT |
| --- | --- | --- | --- | --- |
| 64 | `shard_on_i_hybrid` | `moe_tkg` all-expert | 219 ms | 25.8 ms |
| 64 | `shard_on_i_hybrid` | `moe_tkg_hits` | 211 ms | 21.6 ms |
| 32 | `shard_on_i_hybrid` | `moe_tkg_hits` | 259 ms | 20.2 ms |
| 16 | `shard_on_i` | `moe_tkg_hits` | 518 ms | 20.5 ms |
| 8 | `shard_on_i` | `moe_tkg_hits` | 564 ms | 20.8 ms |

Smaller EP makes each hit cheaper (an expert's I/tp slice) but the busiest
EP group gets more hits, and the per-layer fixed cost (router, shared expert,
loop barriers) stays, so TPOT flattens out at EP=32. The remaining batch-1
step at EP=32 is about 48% MoE fixed cost, 27% waiting for the busiest rank's
hits, 16% attention and 8% all-reduces.

### MTP speculative decoding

Latency config (`scripts/serve_latency.sh`, EP=64, `MTP_K=k`), MATH-500 chat
prompts (no_think, greedy, up to 512 tokens, `scripts/bench_real_prompts.py`):
20 problems one at a time (c=1) and 40 four at a time (c=4). Random-token
prompts are almost never accepted, so they do not show MTP's gain.

| config | TPOT c=1 | out tok/s c=1 | TPOT c=4 | out tok/s c=4 |
| --- | --- | --- | --- | --- |
| plain, `moe_tkg` | 26.3 ms | 38 | 53.2 ms | 72 |
| plain, `moe_tkg_hits` | 21.7 ms | 45 | 34.0 ms | 111 |
| MTP k=4, `moe_tkg` | 20.6 ms | 49 | 16.9 ms | 220 |
| MTP k=3, `moe_tkg_hits`, limit 4 (default) | 13.6 ms | 70 | 19.1 ms | 198 |
| **MTP k=3, `moe_tkg_hits`, limit 16 (`serve_latency.sh`)** | **13.8 ms** | **70** | **17.1 ms** | **216** |
| MTP k=4, `moe_tkg_hits`, limit 20 (`serve_latency.sh`) | 13.8 ms | 70 | 17.0 ms | 218 |

c=1 tok/s for the `moe_tkg` rows is 1000 / TPOT. Acceptance per draft
position is 85 / 60 / 34 / 16% (2.8 tokens per step at k=3, 2.9 at k=4), the
same with either MoE kernel. TTFT is 0.18-0.19 s at c=1 and 0.24-0.25 s at
c=4 (plain: 0.16 / 0.22-0.27 s). MTP k=3 with `moe_tkg_hits` is the
recommended latency setting: k=4 does not help (its fourth draft is accepted
16% of the time). Design, correctness checks and the per-step overhead:
[docs/MTP_DESIGN.md](docs/MTP_DESIGN.md).

### 8K in / 1K out at high concurrency: aligned decode buckets and an FP8 KV cache

`contrib/benchmark/bench_matrix.sh` (vllm-project/vllm-neuron#75): 8192 input tokens of ShareGPT text (+/-3%), exactly 1024 output tokens, 10 x concurrency requests, MTP k=3, `moe_tkg_hits` up to 128 tokens, 2048-token prefill chunks, BF16 unless noted (`results/bench/sharegpt8k_*`).

**Align the decode context buckets.** The decode kernel resizes the cache block length to `gcd(block_size, bucket / (128 * lnc))` so that the blocks per request divide evenly across partitions (`resize_cache_block_len_for_attention_tkg_kernel`). With `block_size` 32 at lnc 2, the 8448 / 9472 buckets resize it to 1 ("Token-gen bucket length of 9472: reducing block length by 32x"), and every KV row is then its own DMA. 8192 keeps 32 and 10240 gives 8. `serve.sh 10240 32 ... '[8192]'` instead of `9472 ... '[8448]'`:

| concurrency | TPOT 8448 / 9472 | TPOT 8192 / 10240 | total tok/s before -> after |
|---|---|---|---|
| 1 | 17.0 ms | 17.7 ms | 477 -> 468 |
| 4 | 29.9 ms | 25.1 ms | 1087 -> 1328 |
| 8 | 44.3 ms | 32.4 ms | 1499 -> 2040 |
| 16 | 76.0 ms | 50.7 ms | 1769 -> 2668 |
| KV limit (27 -> 25) | 102.8 ms | 72.9 ms | 2235 -> 2901 |

Much of the near-linear growth of TPOT with concurrency was this: the KV read per step grows with the batch, and at block length 1 it is DMA-descriptor bound. Pick buckets whose `bucket / 256` shares a large factor with 32 (8192, 12288, 16384 keep 32 or 16).

**FP8 KV cache** (`KV_DTYPE=fp8`): e4m3 with unit scales (the checkpoint has none; K is QK-normed), the packed K layout the decode and segmented kernels read, written in prefill with `scatter_packed_k`. On the 4-layer checkpoint, against the BF16 cache with the same prompts (5000 and 1100 tokens of text, 32 greedy tokens), every step's logits have cosine >= 0.9988 and KL ~0.001; the one greedy split is at a 0.06 near-tie. A 5.8k-token needle is answered on the full model. Same aligned buckets, `max_num_seqs` 48:

| concurrency | TTFT | TPOT (BF16) | ITL (BF16) | total tok/s (BF16) |
|---|---|---|---|---|
| 1 | 1.11 s | 22.0 ms (17.7) | 47.2 ms (40.2) | 394 (468) |
| 4 | 1.50 s | 23.7 ms (25.1) | 46.1 ms (48.2) | 1350 (1328) |
| 8 | 1.51 s | 30.9 ms (32.4) | 52.6 ms (55.9) | 2122 (2040) |
| 16 | 1.54 s | 47.5 ms (50.7) | 65.8 ms (73.5) | 2837 (2668) |
| 32 | 1.60 s | 78.6 ms | 94.9 ms | 3442 |
| 48 | 1.66 s | 106.9 ms | 112.3 ms | **3818** (2901 at the BF16 limit, 25) |

- The KV cache holds twice the requests (48+ against 25 at 10240), so the best throughput is 32% higher.
- From concurrency 4 up the decode step is shorter (half the KV bytes).
- At concurrency 1 the step is ~7 ms longer, presumably the in-kernel FP8 handling across 80 layers. For single-request latency keep BF16.
- The first FP8 prefill write unswizzled, wrote and reswizzled each layer's whole K cache; a per-lane `k[.., p // 2, :, p % 2]` write was worse (stride-2 byte scatter). Both doubled TTFT. `scatter_packed_k` writes whole token pairs and brings TTFT back to BF16's.

**Other findings at this shape**
- **MTP k=2 vs k=3** (same aligned BF16 config): equal at concurrency 16 (49.9 / 50.7 ms) and worse at 25 (76.8 / 72.9 ms). k=3 stays the setting.
- **TTFT** median stays at 1.1-1.7 s at every concurrency (prefill is chunked and scheduled first). P99 TTFT grows with concurrency because the benchmark starts all C requests at once and the scheduler runs one prefill per step: the last of the first wave waits for C prefills (~1.1 s each).
- **TPOT minus ITL** is the decode time lost to other requests' prefill steps (prefill and decode never share a step on Neuron); here it is a few ms per output token.
- **Short context, same buckets:** ISL 128 measures the same ITL as ISL 8192 (40.5 / 41.4 ms at concurrency 1, 58.2 / 60.5 ms at 4). The decode step's attention cost follows the context bucket, not the actual context.

### Comparison with the previous implementation (vllm-neuron 0.24, neuronx-cc 2.27)

The same model code previously ran on vllm-neuron 0.24. Same benchmark, same
seeds:

| config | 0.24 | 0.29 (this folder) |
| --- | --- | --- |
| B, prefill bucket 2048, 16k input, c=1: TTFT / TPOT | 4.40 s / 30.6 ms | 2.61 s / 27.7 ms |
| B, 16k input, c=15: TTFT / total tok/s | 34.4 s / 3078 | 20.3 s / 4756 (multi-bucket: 15.8 s / 5707) |
| B, 1k input, c=15: TTFT / total tok/s | 4.16 s / 1036 | 2.45 s / 1309 (multi-bucket: 1.59 s / 1475) |
| C, 1k input, c=127: TTFT / output tok/s | 32.8 s / 378 | 12.3 s / 773 (multi-bucket) |
| C, 512 input, c=127: TTFT / output tok/s | 32.8 s / 388 | 8.9 s / 976 (multi-bucket) |
| latency config, 16k input, c=1: TTFT / TPOT | 1.80 s / 26.9 ms | 1.83 s / 27.7 ms (`moe_tkg_hits`: 1.83 s / 23.5 ms) |

Where the gains come from:

1. **The dynamic-block MoE prefill kernel now works with prefix caching.** On
   neuronx-cc 2.27, segmented attention followed by any kernel with a
   register-driven loop (`moe_cte` `shard_on_i_hybrid`, `pack_tokens`) in the
   same graph failed to compile (`NCC_IBIR735` / `NCC_ICFG001`), so segmented
   prefill had to fall back to the static `shard_on_i` MoE, which computes the
   worst-case block count (~3x slower at 2048 tokens). On neuronx-cc 2.28 the
   minimal repro passes in every combination and the full model compiles, so
   every prefill path uses the hybrid kernel. This gives the ~40% TTFT drop of
   config B at an unchanged configuration.
2. **Prefill buckets no longer have to equal the KV segment size.** With
   prefix caching, 0.24 required `num_batched_tokens_buckets ==
   kv_segment_size_buckets`, so every prompt was padded to a 2048-token step.
   0.29 compiles each (Q bucket, segment size) pair, so short prompts use a
   512 / 1024 bucket and long prompts 4096 / 8192-token steps.
3. With the same all-expert `moe_tkg`, single-request decode is ~0.8 ms (3%)
   slower per token than on 0.24 in the latency config (same model code); the
   cause has not been isolated.

### Status of issues found on 0.24

| Issue on 0.24 | 0.29 |
| --- | --- |
| Segmented attention + register-loop kernel in one graph fails to compile (`NCC_IBIR735` / `NCC_ICFG001`) | **Fixed** (neuronx-cc 2.28): repro and full model compile and run |
| CTE MLP silently wrong when the per-rank intermediate is not a multiple of 128 (I=208: cos ~0.7) | **Fixed**: I=208 unpadded matches the padded result. The model still pads (validated path) |
| `build_blockwise_mapping` with an EP-TP group > 1 fails to trace (`torch.narrow`) | Fixed in the plugin (`index_select`) |
| `NF.moe_tkg` rejects plain `float8_e4m3fn` weights | Still rejected; the model calls the kernel wrapper directly |
| `"fp8"` not in the Neuron platform's supported quantizations | Still missing; `register()` adds it |
| DCP kernels build their `ReplicaGroup` from the caller's group only, so each DCP group compiles its own NEFFs | Unchanged (compile time only, not correctness) |
| Prefill and decode never share a step | Unchanged; at high concurrency TPOT reaches ~2x ITL |

## Testing

Install the package first (`pip install -e contrib/models/hy_v3`).

### Integration test (accuracy)

The full model is too large for an FP32 HuggingFace reference, so the test
builds a 4-layer checkpoint from the real weights (`scripts/make_mini.py`,
FP8 for vLLM plus a dequantized BF16 copy for HF) and runs the template's
three-way logit validation, once with flash-attention prefill and once with
prefix caching (segmented prefill) and batch-4 decode. It needs the whole
trn2.48xlarge and takes about 12 minutes.

Two settings differ from the template, because the model runs FP8 activations
against a BF16 golden: the top-K bounds are 0.1 / 0.12 / 0.15 / 0.15
(measured <= 0.06), and the argmax divergence tolerance is 0.25 logits instead
of 1 BF16 ULP. On the flat 4-layer distribution FP8 rounding flips near-ties
whose golden top-1 / top-2 gap is 0.06-0.125; a real defect shows gaps of
several logits.

```bash
HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/integration/test_model.py \
    -v -s --timeout=7200
```

### Module tests

Single-NeuronCore tests of the FP8 MLP (dense layer, shared expert, 8-way decode
copy; CTE and TKG paths) and the routed experts (`moe_cte` hybrid, `moe_tkg`,
`moe_tkg_hits` with 0-3 local hits, `moe_tkg_hits_batch` with 2 / 4 tokens).
They build the kernel inputs from the real checkpoint (rank-0 shard) through
the model's own weight loaders and compare against an FP32 fake-quant
reference.

```bash
HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/module/ -v -s
```

### Unit tests

CPU only: FP8 weight transforms (halving, scale layout, padding, expert
packing) and the router against the HuggingFace `HYV3TopKRouter`.

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hy_v3/test/unit/ -v
```

### Scripts

| Script | Purpose |
| --- | --- |
| `scripts/serve.sh` | `vllm serve` with prefix caching (configs B / C / DCP below); `EP=<n>` sets the EP degree |
| `scripts/serve_latency.sh` | single-request latency config (no prefix caching, prefill buckets 512..16384); `MTP_K=<k>` enables MTP |
| `scripts/bench_real_prompts.py` | TTFT / TPOT / throughput on MATH-500 chat prompts, plus MTP acceptance from `/metrics` |
| `scripts/check_mtp.py` | MTP vs plain greedy output (lossless check) and acceptance |
| `scripts/bench_matrix.sh` | `vllm bench serve` matrix, random dataset, a distinct seed per cell so cells never share prefix-cache entries |
| `scripts/bench_prefix_cache.sh` | prefix-cache benchmark: shared-prefix share sweep, hit rate read from the server's `/metrics` |
| `scripts/summarize_bench.py` | markdown table from benchmark JSONs, optionally against a baseline directory |
| `scripts/eval_math500.py` | MATH-500 through the OpenAI API, graded with math-verify |
| `scripts/make_mini.py` | reduced real-weight checkpoint (first N layers), FP8 + BF16 |
| `scripts/hf_reference.py` | HF CPU reference logits for the BF16 reduced checkpoint |
| `scripts/compare_hf_logits.py` | vLLM Neuron vs the HF reference, per-step logits |
| `scripts/validate_features.py` | segmented prefill, prefix-cache hits, batch-128 vs batch-4 consistency (optionally with DCP) |
