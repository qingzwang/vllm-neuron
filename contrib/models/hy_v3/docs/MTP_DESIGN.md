# MTP speculative decoding for Hy3 on Neuron: design

Status: proposal (branch `hy3-fp8-opt`). Goal: lower single-request TPOT,
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
  further draft steps the MTP output itself is fed back.
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
