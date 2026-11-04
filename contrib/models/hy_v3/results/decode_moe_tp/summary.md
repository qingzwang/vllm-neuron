# Decode routed experts split over 4-rank groups (`HY3_DECODE_MOE_TP=4`, experimental, off)

At 1-2 requests a decode step routes only 32-64 (token, expert) pairs over 64 EP ranks:
the busiest rank runs 2-3 experts while most run none, and the MoE all-reduce waits for it
(9-10 ms of a 47.5 ms 1-request step, `docs/PROFILE_85K.md`). With the option every rank also
holds chunk `rank % G` (1536 / G intermediate columns) of the 3G experts of its group of G
consecutive ranks, and decode steps of at most `HY3_DECODE_MOE_TP_MAX_T` (8) tokens run the
hits kernel (`moe_tkg_hits_batch`, unchanged) over the group's hit experts on those chunks; the
partial outputs add up in the existing all-reduce. The idea is DeepSeek-V4.1's TP8EP8 decode MoE
(github.com/OswaldHe/vllm-neuron, contrib/models/deepseek_v41, kernels/moe_tp8.py).

Busiest rank per layer, from the routing captures (`scripts/decode_moe_tp_sim.py`, 19k decode
tokens) and the kernel's cost per hit on one core (`scripts/decode_moe_tp_bench.py`, T = 4:
whole expert 75 us, 384-column chunk 28 us, 192-column chunk 24 us):

| tokens per step | EP64 | TP4 groups | TP8 groups |
|---|---|---|---|
| 4 (1 request, MTP k=3) | 2.07 experts, 155 us | 3.76 chunks, 106 us | 5.49 chunks, 132 us |
| 8 (2 requests) | 2.53, 190 us | 5.44, 154 us | 8.50, 204 us |
| 16 (4 requests) | 2.97, 223 us | 7.57, 214 us | 12.6, 302 us |

Full model, `serve_85k.sh` FP8 KV + MTP k=3, `KV_CAP=0.41` (the decode copy is ~4.5 GB a rank:
KV 468,736 tokens instead of ~697k): continuations 19/19 identical to dense, needle 12/12.
`bench_pd_split.sh "1 4"` pass 2 (`results/bench/pd_split_85k_nodcp_fp8kv_mtp3_moetp4`) vs the
dense run of the same session (`..._rerun`):

| | dense | TP4 |
|---|---|---|
| ITL c1 | 47.4 ms | **44.5 ms (-6%)** |
| TPOT c1 | 23.4 ms | **21.7 ms (-7%)** |
| ITL c4 (16 tokens: EP64 path) | 66.6 ms | 66.7 ms |
| TPOT c4 | 39.5 ms | 55.2 ms |
| prefix-cache hit, pass 2 | 99.9% | 92.3% |

The c4 TPOT and the output throughput are worse only because the smaller KV cache evicts
prefixes between the two passes: the evicted ones are prefilled again and interrupt decode
(the c4 decode step itself is unchanged). So the option suits low-concurrency, latency-bound
serving with room in the KV cache; a decode-only (P/D) instance could keep just the chunked
copy and lose no KV.
