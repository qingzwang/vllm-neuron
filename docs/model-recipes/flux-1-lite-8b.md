# FLUX.1-lite-8B Model Recipe

<!-- meta: description: Model recipe for running FLUX.1-lite-8B text-to-image
generation on Neuron, including supported checkpoints, component placement,
measured latency on Trn2, and known limitations. -->
<!-- meta: keywords: Neuron, FLUX, FLUX.1-lite, flux.1-lite-8B, Freepik,
diffusion, text-to-image, DiT, diffusers, model recipe, Trn2, Trainium -->
<!-- meta: date_updated: 2026-09-01 -->
<!-- Content type: model-card -->

## Introduction

[FLUX.1-lite-8B](https://huggingface.co/Freepik/flux.1-lite-8B) is an 8B-parameter
text-to-image rectified-flow transformer, distilled by Freepik from
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) by pruning it
from 19 to 8 double-stream (MMDiT) blocks while keeping all 38 single-stream
blocks. It uses guidance distillation, so there is no negative pass: image cost
is independent of the guidance value.

**This model does not run through `vllm serve` or the vLLM offline API.** vLLM
0.24 has no text-to-image request path — its `DiffusionConfig` covers discrete
diffusion *language* models, not latent image diffusion. FLUX therefore runs
through a standalone pipeline, `vllm_neuron.model.flux.NeuronFluxPipeline`, which
reuses this plugin's compilation stack and NKI kernels but not its model runner.
See `vllm_neuron/model/flux/README.md` for the design.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| FLUX.1-lite-8B | [Freepik/flux.1-lite-8B](https://huggingface.co/Freepik/flux.1-lite-8B) | Trn2 | BF16 |
| FLUX.1-lite-8B-alpha | [Freepik/flux.1-lite-8B-alpha](https://huggingface.co/Freepik/flux.1-lite-8B-alpha) | Trn2 | BF16 |
| FLUX.1-dev | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Trn2 | BF16 |

> Any diffusers-format `FluxPipeline` with `guidance_embeds=True` loads through
> the same path. FLUX.1-schnell and other guidance-free variants are rejected at
> load time — they need a different guidance path, not just different weights.
> FLUX.1-dev is 19 double blocks instead of 8, so expect roughly 1.6x the step
> latency measured below.

## Features

| Category | Feature | Status |
|---|---|---|
| **Task** | Text-to-image | ✅ |
| | Image-to-image / inpainting | ❌ |
| | ControlNet / IP-Adapter / LoRA | ❌ |
| **Quantization** | BF16 | ✅ |
| | FP8 / MXFP8 | ❌ |
| **Parallelism** | Single logical NeuronCore | ✅ |
| | Tensor parallelism (TP) | ❌ |
| **Guidance** | Distilled guidance embedding | ✅ |
| | True classifier-free guidance | ❌ |
| **Batching** | Batch 1 | ✅ |
| | Batch > 1 | ❌ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| **Attention** | NKI flash attention | ✅ |

## Requirements

Install diffusers on top of the plugin's environment:

```bash
pip install -r requirements/flux.txt
```

## Quick start

```bash
python examples/vllm_neuron/models/flux/generate.py \
    --model-checkpoint Freepik/flux.1-lite-8B \
    --prompt "A close-up photo of a red panda wearing tiny round glasses, reading a leather-bound book in a cozy library" \
    --steps 28 \
    --output flux_output.png
```

The first run compiles every component (a few minutes, dominated by the VAE
decode stages); subsequent runs hit the compilation cache. Within a process, the
first request additionally pays for loading each NEFF onto the device — about
20 s extra at 1024x1024 — so warm latency only shows from the second request on.

```python
from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline

config = FluxNeuronConfig(height=1024, width=1024, max_sequence_length=512)
pipeline = NeuronFluxPipeline.from_pretrained("Freepik/flux.1-lite-8B", config)
pipeline.compile()

image, timing = pipeline.generate("a red panda reading a book", num_inference_steps=28, seed=42)
image.save("out.png")
print(timing.as_dict())
```

## Component placement

A process is bound to one logical NeuronCore, which addresses its own 24 GB slice
of trn2 HBM at `logical-neuroncore-config: 2`. BF16 FLUX.1-lite (16 GB) plus
T5-XXL (9.5 GB) does not fit, so the default placement keeps T5 on CPU:

| Component | Default | Size (BF16) |
|---|---|---|
| `transformer` (FluxTransformer2DModel) | Neuron | 16 GB |
| `vae` (AutoencoderKL decoder) | Neuron | 0.17 GB |
| `text_encoder` (CLIP-L) | Neuron | 0.25 GB |
| `text_encoder_2` (T5-XXL) | CPU | 9.5 GB |

The transformer keeps the seat because it runs once per denoising step against
T5's once per request. T5 on CPU costs ~1.5 s per request; it is the largest
remaining non-transformer cost. Splitting components across two cores inside one
process does not work — the runtime loads every NEFF onto the process's own core
and rejects a graph whose weights were uploaded elsewhere.

Override with `on_device` (or `--on-device`) to move a component to CPU for an
A/B comparison. A component that fails to load or compile falls back to CPU with
a warning rather than taking the pipeline down.

## Measured latency

Trn2 (`trn2.3xlarge`, one logical NeuronCore, `logical-neuroncore-config: 2`),
BF16, batch 1, `neuronx-cc -O1`, 512-token prompt budget, default placement.
Warm numbers: median over 2 requests after a discarded warmup request.

| Resolution | Steps | ms/step | Denoise | Prompt encode (CPU T5) | VAE decode | **End to end** |
|---|---|---|---|---|---|---|
| 512x512 | 28 | 278 | 7.80 s | 1.56 s | 0.12 s | **9.47 s** |
| 1024x1024 | 8 | 792 | 6.33 s | 1.56 s | 0.50 s | **8.41 s** |
| 1024x1024 | 28 | 791 | 22.14 s | 1.56 s | 0.50 s | **24.20 s** |

Step latency is flat across step counts and stable to well under 1% (p90 within
0.5 ms of p50) — the same graph runs every step, on static shapes, with nothing
left on the host but a one-element fence read.

Staging the VAE decode onto the device is worth 15x on that stage: 0.50 s versus
7.55 s for the same decode in CPU eager mode at 1024x1024.

Attention dominates: 46 attentions per step over a joint sequence of 4608 tokens
(4096 image + 512 text) at 1024x1024. Measured in isolation at those shapes, the
NKI flash-attention kernel runs 4.9 ms/call against 22.1 ms for materialized
SDPA, which is why the kernel is the default.

Compilation, first time on a cold cache:

| Component | Graphs | Compile time (1024x1024, `-O1`) |
|---|---|---|
| `transformer` | 1 | ~2 min |
| `vae` | 5 | ~2 min |
| `text_encoder` | 1 | ~1 s |

Warm restarts hit the local compilation cache and take ~35 s to reach the first
request, almost all of it loading weights from disk.

### Reducing latency

- **Fewer steps.** FLUX.1-lite holds up well at 8 steps and is recognizable at 4.
  Step count scales latency linearly.
- **Lower resolution.** 512x512 is 2.8x faster per step than 1024x1024: the
  joint sequence drops from 4608 to 1536 tokens.
- **Shorter prompt budget.** `--max-sequence-length 256` removes 256 tokens from
  every attention in every step. Prompts longer than the budget are truncated.
- **`-O2` / `-O3`.** `--optimization-level` maps straight to `neuronx-cc -O`.
  Compiles slower; whether it runs faster is workload-dependent.

Reproduce with:

```bash
python examples/vllm_neuron/models/flux/benchmark.py \
    --sizes 512,1024 --steps 8,28 --iterations 3 --json flux_latency.json
```

## Accuracy

Numerics are BF16 throughout, matching how FLUX is normally served, with two
deliberate promotions to fp32: RoPE application and the fused Euler update, so
latent error does not accumulate across ~30 steps of BF16 addition.

**Logic equivalence.** `NeuronFluxTransformer` was run against upstream
`FluxTransformer2DModel.forward` on identical inputs, both fp32 on CPU: max
relative difference **2.9e-7**, i.e. fp32 rounding. This is what covers the
rewritten pieces — hoisted RoPE, the reimplemented rotation, patched GELU,
latent packing, and the timestep/guidance convention.

**BF16 fidelity.** One denoising step at 512x512 with a 256-token prompt,
against that fp32 reference:

| Path | mean abs error | cosine similarity |
|---|---|---|
| Neuron BF16 (this pipeline) | 0.0074 | 0.999969 |
| CPU BF16 eager (diffusers) | 0.1039 | 0.993771 |

Reference velocity has std 1.16, so the Neuron path lands ~14x closer to fp32
than CPU BF16 does — Trainium accumulates matmuls in fp32 where CPU BF16 eager
accumulates in BF16. Separately, the NKI attention kernel matches an fp32 SDPA
reference to 7e-4 max absolute error at 1024x1024 joint-attention shapes.

**End to end.** Compared against the stock diffusers pipeline running BF16 on
CPU, same prompt, seed and schedule (512x512, 8 steps): latent cosine similarity
0.9993 after 1 step, 0.9916 after 8, and image PSNR 27.3 dB. The two images share
composition, subject, palette and lighting but differ in fine detail. That is
expected rather than a defect: a flow-matching ODE amplifies any per-step
numerical difference across steps, and per the table above the CPU BF16 side is
the less faithful of the two. Diffusion output is not reproducible bit-for-bit
across numerical backends — treat a seed as reproducible only within one
placement and dtype.

## Limitations

- One logical NeuronCore; no tensor parallelism. The step graph is where all the
  time goes, so sharding it is the highest-value next change.
- Batch 1 only.
- Resolution and prompt budget are fixed at load time. Changing either means
  recompiling — there is no bucketing, because a diffusion step has no analogue
  of a variable sequence length.
- T5-XXL runs on CPU by default (see above).
- A `neuronx-cc` failure terminates the process rather than raising, so the
  CPU fallback cannot catch it; HBM allocation failures are caught.
