# MTP speculative decoding for Hy3 on Neuron: design

Status: implemented on branch `hy3-fp8-opt` (framework part on `mtp-spec-decode`); see Results. Goal: lower single-request TPOT,
which the 0.29 profile puts at 27.8 ms per step (`PROFILE_0.29.md`), by using
the MTP (next-n prediction) layer that tencent/Hy3-FP8 ships.

## What the checkpoint and vLLM provide

- `num_nextn_predict_layers = 1`: `model.layers.80.*` holds `enorm`,
  `hnorm`, `eh_proj` ([H, 2H]), a full decoder layer (attention + 192-expert
  MoE + shared expert, FP8) and `final_layernorm`. There is no embedding or
  lm_head of its own: both are shared with the target.
- MTP step (vLLM `hy_v3_mtp.py`, HF `HYV3MTPDecoderLayer`):
  `h = eh_proj(cat(enorm(embed(t_{i+1})), hnorm(h_i)))`, then the decoder
  layer (with its residual), then `final_layernorm`, then the target lm_head.
  `h_i` is the target's final (post-norm) hidden state at position i; for
  further draft steps the MTP output itself is fed back. (The target hands
  over the pre-norm state and the draft applies the target's final norm; see
  Results.)
- vLLM 0.29: `--speculative-config '{"method": "mtp", "num_speculative_tokens": k}'`
  turns the target `hy_v3` config into a draft config with
  `model_type=hy_v3_mtp`, `architectures=["HYV3MTPModel"]`, same checkpoint.

## What vllm-neuron 0.29 lacks

1. `NeuronModelRunner` accepts only `eagle3` and `dflash`
   (`neuron_model_runner.py`, "Unsupported speculative decoding method").
2. Draft classes are looked up in `vllm_neuron.model.registry.get_models()`
   only (`eagle.py`), so a contrib / plugin model cannot provide one.
3. `EagleProposer` hard-codes `method == "eagle3"`, a `3 * hidden_size`
   target-hidden width (synthetic warmup inputs) and draft layer count =
   draft `num_hidden_layers` (for MTP that is the target's 80).

The proposer / draft-model contract itself (fused draft NEFF with the KV
update plus an unrolled recurrent loop, on-device accepted-token extraction,
`(tokens [bs, 1+k], drafts [bs, k], logits)` output, async correction
kwargs) fits MTP unchanged.

## Plan

### A. Framework PR (vllm_neuron, model-agnostic)

1. `model/registry.py`: `register_neuron_model(arch, cls)`; registered
   classes are returned by `get_models()` (and so registered with vLLM in each
   worker). Plugins call it from their `vllm.general_plugins` hook.
2. `vllm/spec_decode/mtp.py`: `MTPProposer(EagleProposer)`:
   - accepts `method == "mtp"`;
   - draft layer names `layers.{L}..layers.{L+n_predict-1}` from
     `n_predict` / `num_nextn_predict_layers`;
   - `get_target_aux_hidden_state_layers()` returns `(L,)`, meaning "the
     target's final post-norm hidden state" (index == `num_hidden_layers`);
   - `alias_target_weights(target_model)` delegates to the draft model, which
     points its embedding / lm_head at the target's (no extra HBM);
   - target hidden width = `hidden_size` (EagleProposer gets a
     `target_hidden_dim` hook instead of the hard-coded `3 * hidden_size`).
3. `NeuronModelRunner`: `method == "mtp"` -> `MTPProposer`, `is_mtp_spec`
   included in `is_spec_decode`; the aux-layer selection and
   `alias_target_weights` calls go through the drafter when it provides them
   (today only DFlash does). Unit tests for the proposer plumbing in CPU mode.

### B. Hy3 contrib (contrib/models/hy_v3)

1. Target (`model.py`): `SupportsEagle3`-style `set_aux_hidden_state_layers`;
   when it contains `num_hidden_layers`, `forward` also returns the final
   normed hidden states for every token (gathered over SP in prefill).
   `@async_speculative_decoding` + rejection sampler when
   `spec_decode_metadata` is given (as in `llama3/model.py`). Decode already
   accepts `S_decode = 1 + k` tokens per request (attention megakernel,
   `moe_tkg`, shared expert switches to CTE above 96 tokens). DCP stays
   restricted to `S_decode == 1`.
2. Draft (`model_mtp.py`, registered as `HYV3MTPModel`): reuses
   `HYV3DecoderLayer(layer_idx=80)` and the FP8 loaders; `enorm`, `hnorm`,
   `eh_proj` (BF16 or FP8 per checkpoint), `final_layernorm`; forward
   mirrors `Eagle3LlamaForCausalLM.forward` (KV update pass, then k-1
   recurrent decode passes, greedy on-device sampling with the shared lm_head).
3. Tests: 4-layer checkpoint with the MTP layer kept (`make_mini.py` option)
   to compare the draft's step logits with HF `HYV3MTP`; acceptance rate and
   TPOT on the full model (MATH-500 / chat prompts, k = 1 and 2).

## Expected gain (estimate, to be measured)

- Draft cost per draft token: one decoder layer + lm_head + sampler,
  ~0.5 ms vs a 27.8 ms target step.
- Verify step at batch 1 with 1 + k tokens: weight-bound, so close to the
  single-token step (more experts get hit; the 12.6 ms routed-expert part grows
  the most).
- With k = 1 and acceptance ~0.7-0.8, ~1.7-1.8 tokens per target step:
  TPOT from 27.8 ms to roughly 16-18 ms. k = 2 reuses the same MTP layer for
  the second draft; its acceptance is typically much lower, so it is measured,
  not assumed.

## Risks

- Acceptance on Neuron numerics (FP8 MLP / attention) is unknown until
  measured.
- MoE in the verify step: 1 + k tokens select up to 8(1 + k) experts, which
  spreads the decode MoE load and may raise the verify-step cost more than
  for a dense model.
- Compile time: the draft adds one fused NEFF per decode bucket.

## Results (2026-10-22)

### Correctness

- 4-layer checkpoint with the MTP layer (`make_mini.py --keep-mtp`,
  `scripts/check_mtp.py`): greedy MTP output equals plain decoding token for
  token, k=1 and k=2, batch 4, 16 tokens.
- Bug found on the way: returning the target's post-norm hidden states as the
  auxiliary output, while the same tensor also feeds the lm_head, made the
  verify step's lm_head rows wrong for every request but the first at batch > 1
  (the norm output itself was correct in a tensor capture; computing the
  lm_head from it on the host gave the right tokens). Handing over the pre-norm
  last-layer output, as the Neuron capture convention does for EAGLE3, fixed it.
- Full model, 20 MATH-500 problems, greedy, 512 tokens: the MTP and plain
  outputs agree on 2/20 and split early on the others at equally plausible
  wording ("coordinates" vs "point", `\[` vs `$$`). Graded with math-verify:
  plain 11/20 correct (7 truncated), MTP 13/20 (5 truncated). Most likely the
  verify step (1 + k tokens per request) is numerically slightly different from
  a single-token decode step and flips near-ties; not yet confirmed with
  per-token logprobs, and a full MATH-500 run with MTP is still to do.

### Performance (full model, latency config, concurrency 1)

`scripts/serve_latency.sh` (`MTP_K=k` for MTP), 20 MATH-500 problems as chat
prompts (no_think, greedy, up to 512 tokens, `scripts/bench_real_prompts.py`),
acceptance from the server's spec-decode counters. "Step" is the median gap
between streamed chunks (one engine step each) for two 256-token completions.

| | plain | MTP k=1 | k=2 | k=3 | k=4 |
| --- | --- | --- | --- | --- | --- |
| **TPOT, real prompts (median)** | 26.3 ms | 30.9 ms | 23.4 ms | 21.0 ms | **20.6 ms (-22%)** |
| tokens per step (1 + accepted / drafts) | 1 | 1.87 | 2.48 | 2.78 | 2.93 |
| acceptance, draft position 1 | - | 87.0% | 86.5% | 85.4% | 85.4% |
| acceptance, position 2 | - | - | 61.1% | 59.4% | 58.9% |
| acceptance, position 3 | - | - | - | 33.4% | 32.3% |
| acceptance, position 4 | - | - | - | - | 16.0% |
| step time (streamed chunk gap, median) | 26.4 ms | 55-59 ms | 60 ms | 60-61 ms | 61-62 ms |
| 256-token completion, total (2 requests) | 6717 / 6719 ms | - | 6221 / 7407 ms | 5389 / 6068 ms | 5579 / 5790 ms |
| TPOT, random prompts 1k / 16k | 26.3 / 27.7 ms | 39.2 / 43.3 ms | 35.3 / 36.4 ms | 32.6 / 39.1 ms | 33.9 / 32.3 ms |
| TTFT, real prompts (median) | 0.16 s | 0.20 s | 0.20 s | 0.20 s | 0.20 s |

Acceptance per position counts the steps whose draft at that position was
accepted (so it is cumulative: position 2 can only be accepted after position 1).

- Each step costs ~55-62 ms for every k (vs 26.4 ms plain), so the gain comes
  only from tokens per step. An extra draft position costs ~1-2 ms per step, so
  k=3 / k=4 still pay off with the single MTP layer reused recurrently
  (positions 3-4 accept 32% / 16%).
- On real chat prompts k=4 lowers TPOT by 22% and k=3 by 20%; k=1 is slower than
  plain. Random-token prompts are hardly ever accepted, so every k is slower
  than plain there. TTFT grows by ~40 ms (the draft's prefill pass).

Device profile of a spec step (core 0): the draft NEFF takes 0.6-1.0 ms and
the target verify NEFF 28-30 ms of compute, i.e. the device work per step is
about one plain decode step. The rest of the ~55 ms is waiting: an idle gap
before the verify NEFF or a `BARRIER` at its start of up to 32 ms (this core is
ready early and waits for other ranks). The step is bounded by host-side work
on the slowest rank, which the async pipeline hides in plain decode but not in
speculative decoding.

The same holds for the officially supported Llama-3.1-8B + EAGLE3 (TP=8, k=1,
tutorial config): ~25 ms per spec step against ~7-15 ms per plain step, 68%
acceptance, total time -8% / +35% on two requests. So the overhead is in the
framework's speculative-decoding step, not in the Hy3 draft (reported to
vllm-neuron as an issue).

### Outlook

- With the per-step overhead removed (step ~= 29 ms device time), k=1 at 87%
  acceptance would give a TPOT of ~16-18 ms (~35-40% lower than plain).
- Until then, raising k amortizes the overhead: k=4 is the best measured
  setting (TPOT 20.6 ms on real prompts, -22%).
