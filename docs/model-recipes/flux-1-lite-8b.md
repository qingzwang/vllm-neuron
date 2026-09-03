# FLUX.1-lite-8B Model Recipe

<!-- meta: description: Model recipe for running FLUX.1-lite-8B text-to-image
generation on Neuron, including supported checkpoints, how the model is sharded
across NeuronCores, measured latency on Trn2 at tp_degree 2 and 4, and known
limitations. -->
<!-- meta: keywords: Neuron, FLUX, FLUX.1-lite, flux.1-lite-8B, Freepik,
diffusion, text-to-image, DiT, diffusers, model recipe, tensor parallelism,
tp_degree, Trn2, Trainium -->
<!-- meta: date_updated: 2026-09-03 -->
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
| | LoRA, loaded and switched at runtime | ✅ |
| | ControlNet / IP-Adapter | ❌ |
| **Quantization** | BF16 | ✅ |
| | FP8 / MXFP8 | ❌ |
| **Parallelism** | Tensor parallel over 2, 4 or 8 NeuronCores | ✅ |
| | Single core (`tp_degree=1`) | ❌ — 24.44 GiB of weights against ~22 GiB |
| | Data or context parallelism | ❌ |
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

A trn2.3xlarge is enough: it has four logical NeuronCores at the default
`logical-neuroncore-config: 2`, which covers `tp_degree` 2 and 4.

## Quick start

```bash
python examples/vllm_neuron/models/flux/generate.py \
    --model-checkpoint Freepik/flux.1-lite-8B \
    --tp 4 \
    --prompt "A close-up photo of a red panda wearing tiny round glasses, reading a leather-bound book in a cozy library" \
    --steps 28 \
    --output flux_output.png
```

![FLUX.1-lite-8B output on Trn2: a red panda in round glasses reading a book](images/flux-1-lite-8b-sample.png)

*1024x1024, 28 steps, guidance 3.5, seed 42 — exactly the command above.
Downscaled for this page.*

```python
from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline

config = FluxNeuronConfig(height=1024, width=1024, tp_degree=4)
# `with` releases the ranks' NeuronCores on the way out.
with NeuronFluxPipeline.from_pretrained("Freepik/flux.1-lite-8B", config) as pipeline:
    pipeline.compile()
    image, timing = pipeline.generate(
        "a red panda reading a book", num_inference_steps=28, seed=42
    )
image.save("out.png")
print(timing.as_dict())
```

No core pinning to arrange: this process never touches the device, and the ranks
pin themselves to `tp_core_ids` (`0..tp_degree-1` by default).

The first run compiles every component (a few minutes, dominated by the VAE decode
stages); later runs hit the compilation cache. Within a process the first request
that decodes an image additionally pays for loading the VAE NEFFs onto the device
— about 12 s at 1024x1024 — so warm latency shows from the second image on.

## How the model is divided

Every network runs on Neuron, tensor-parallel across `tp_degree` logical cores with
one process per core. That is not a choice: the compile backend loads every NEFF
onto the process's own core, and a core belongs to one process — a second process
asking for the same core fails to bring the runtime up at all. So `tp_degree`
cores means `tp_degree` processes. **This pipeline's own process holds no core**; it
tokenizes, drives the denoising loop, and turns the result into an image.

| Component | BF16 weights | Across ranks | Why |
|---|---|---|---|
| `transformer` (FluxTransformer2DModel) | 15.20 GiB | sharded | 28 invocations per request: the whole cost |
| `text_encoder_2` (T5-XXL) | 8.87 GiB | sharded | large, and 64 heads divide cleanly |
| `text_encoder` (CLIP-L) | 0.22 GiB | replicated | sharding would add collectives to save nothing |
| `vae` (AutoencoderKL decoder) | 0.15 GiB | replicated | convolutional, once per request |

This is the same division NxD Inference makes for the same model, which makes the
two directly comparable — see "Against NxD Inference" below.

### Why there is no tp_degree=1

The four components are **24.44 GiB** of BF16 weights, and one core's HBM
partition holds ~22 GiB. At `tp_degree=1` it compiles and loads, then dies asking
for activation space on top:

```
NRT:nrt_infodump  Failure: NRT_RESOURCE in nrt_tensor_allocate
RuntimeError: nrt_tensor_allocate status=4
```

`tp_degree=2` halves that to 12.22 GiB per core and is the floor for this
checkpoint. `FluxNeuronConfig` rejects 1 with that explanation rather than letting
it fail at load.

### What is sharded, exactly

Only attention heads and feed-forward widths. The residual stream stays full width
and identical on every rank, so norms, modulation projections, embedders, the
final `proj_out` and both LayerNorm stacks are replicated — and the attention
processor needs no changes at all, because it derives the head count from the
tensor it is handed rather than from a config value.

Two places need more than the standard pattern:

* **The single-stream block's `proj_out`** consumes
  `cat([attn_output, mlp_hidden_states])`. With both producers column-parallel each
  rank holds `[attn_dim/tp + mlp_dim/tp]`, which is not that rank's contiguous
  slice of the global input, so a plain row-parallel linear would multiply the
  wrong weights. Its weight is split at `attn_dim`, each half sharded along its own
  axis, and the two partial products summed before a single all-reduce.
* **T5's relative-attention bias** is an `Embedding(32 buckets, 64 heads)` that
  only block 0 owns, computed once there and threaded through every later block.
  It is sharded along the head axis so each block sees exactly its own heads'
  biases, and `T5Attention.n_heads` and `inner_dim` are reduced to match — T5
  reshapes from those attributes rather than from the tensor.

Row-parallel layers use a local implementation rather than `vllm_neuron.nn`'s,
which rejects a non-float32 bias; FLUX's are BF16. The bias is not a distributed
quantity, so it is held whole and added after the reduce.

## LoRA

Adapters are loaded into device slots at runtime and selected per request. Nothing
is recompiled, and switching between loaded adapters costs **under a millisecond**:

```python
config = FluxNeuronConfig(height=1024, width=1024, tp_degree=4,
                          lora_slots=2, lora_max_rank=64)
with NeuronFluxPipeline.from_pretrained(CKPT, config) as pipeline:
    pipeline.compile()
    pipeline.load_lora("realism", "/adapters/xlabs-realism")
    pipeline.load_lora("superreal", "/adapters/super-realism.safetensors")

    pipeline.set_lora("realism")
    image_a, _ = pipeline.generate(prompt, num_inference_steps=28)
    pipeline.set_lora("superreal")      # ~0.6 ms
    image_b, _ = pipeline.generate(prompt, num_inference_steps=28)
    pipeline.set_lora(None)             # back to the unmodified model
```

`lora_slots=0` (the default) leaves the graph exactly as it was, so a deployment
that does not use adapters pays nothing.

diffusers/PEFT, kohya and XLabs layouts all load: the file goes through
`FluxPipeline.lora_state_dict`, so diffusers' own converters do the format work.

| base model | [XLabs realism](https://huggingface.co/XLabs-AI/flux-RealismLora) (r=16) | [kohya super-realism](https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA) (r=64) |
|---|---|---|
| ![](images/flux-lora-base.png) | ![](images/flux-lora-xlabs.png) | ![](images/flux-lora-kohya.png) |

<sub>One compiled model, one prompt, one seed; both adapters loaded at runtime and
selected with a sub-millisecond switch. FLUX.1-dev, 512x512, 28 steps,
`tp_degree=4`.</sub>


### How it avoids recompiling

Two properties of the backend, both verified directly rather than assumed:

* An in-place write to a device tensor is visible to an already-compiled graph —
  after `copy_` into a parameter, a buffer or a plain tensor attribute, the next
  call returns the new value and Dynamo reports no additional graph. So adapter
  weights live in device tensors the NEFF reads.
* The *selection index* can be a device tensor too. Every adapted layer reads the
  same one-element tensor and does an `index_select` on its slot dimension, so
  switching adapters writes four bytes rather than moving weights.

That second point is what makes switching cheap. A full adapter is hundreds of MB
spread over ~1500 small tensors per rank; moving it takes hundreds of milliseconds
to seconds, and slots exist precisely so that cost is paid once per adapter rather
than once per switch.

### Cost

FLUX.1-dev at `tp_degree=4`, 512x512, two slots at rank 64:

| Operation | Cost |
|---|---|
| `set_lora(...)` between loaded adapters | **0.6–0.8 ms** |
| `load_lora(...)`, 22 MiB adapter (152 modules) | 0.14 s |
| `load_lora(...)`, 585 MiB adapter (494 modules) | 0.58 s |
| Per-step latency, adapter active | 116.6–118.7 ms against 117.3 ms for the base model |
| Slot memory | 385 MB per slot per rank |

The step cost of an active adapter is not measurable against run-to-run spread: the
extra work is two thin matmuls per adapted layer, against a 4608-token attention.
Slot memory scales with `lora_max_rank`, so set it to the widest adapter you
actually use.

### Correctness

Sharding an adapter has to match the sharding of the layer it adapts, and for
row-parallel layers the delta has to be added *before* the layer's all-reduce — `x`
is sharded there, so each rank can only compute a partial `A @ x`, and adding the
delta after the reduce leaves every rank with a different wrong answer. The four
cases (column-parallel, row-parallel, the single block's split `proj_out`, and plain
layers) are checked on CPU against a dense `W x + B (A x)`, exactly.

On device, with two adapters loaded: each changes the output (cos 0.979 and 0.984
against the base model, and they differ from each other), switching away and back
reproduces the earlier latents **bit for bit**, and so does returning to slot 0.

Against a reference -- the same adapter applied by diffusers on CPU, same prompt,
seed, schedule and initial latents, one denoising step at 512x512:

| | cos | mean abs diff |
|---|---|---|
| base vs CPU base (the backend's own floor) | 0.996447 | 0.0909 |
| **adapter vs CPU with the same adapter** | **0.996433** | 0.0924 |
| adapter vs CPU base | 0.995558 | 0.1107 |
| base vs CPU with the adapter | 0.986889 | 0.1555 |

The second row is the point: applying the adapter does not change how well this
pipeline agrees with CPU diffusers -- it lands on the same floor as the base model.
The cross rows are ordered the right way round in both directions.

Sharper still, subtracting the backends' own disagreement by comparing what the
adapter *contributes* on each side: the two deltas agree to **cos 0.9655** with
matching magnitude (mean 0.1014 against 0.1065). That is close to what agreement
can look like here, since the backend floor (0.0909) is itself the same size as the
contribution being measured.

### Adapters have to match the checkpoint

An adapter trained for FLUX.1-dev names 19 double-stream blocks; FLUX.1-lite has 8.
Loading one into the other logs how many of its targets do not exist here and
adapts only the rest, which is not what the adapter's author intended. The numbers
above are on FLUX.1-dev, because that is what the public adapters target.

## Cores

`tp_degree` logical cores, one rank each, and nothing else needs one:

| `tp_degree` | Cores | Per-core weights | Fits a trn2.3xlarge? |
|---|---|---|---|
| 2 | 2 | 12.22 GiB | yes |
| 4 | 4 | 6.11 GiB | yes, exactly |
| 8 | 8 | 3.06 GiB | no — needs a larger instance |

Use the default `logical-neuroncore-config: 2`. LNC=1 would give eight logical
cores on this instance, but each is one physical NeuronCore instead of a fused
pair, so a rank gets half the compute: measured at 1227 ms/step against 791 for
the same unsharded transformer. It is for models small enough that a fused core
would sit idle, and for this pipeline it is also currently broken — the transformer
graph compiled with `--lnc=1` returns NaN from the first denoising step at both
512x512 and 1024x1024, while the encoders and initial latents on the same run are
finite.

### How much HBM each core gets

trn2's 96 GB of HBM is divided into four ~22 GiB partitions, one per **physical
core pair** — not per logical core. Measured by allocating 1 GiB buffers until
failure:

| `logical-neuroncore-config` | Logical cores | Allocating on | Total before failure |
|---|---|---|---|
| 2 (default) | 4 | core 0 | 22 GiB |
| 2 | 4 | cores 0 + 1 | 44 GiB |
| 1 | 8 | core 0 | 22 GiB |
| 1 | 8 | cores 0 + 1 | 22 GiB — same pair, shared |
| 1 | 8 | cores 0 + 2 | 44 GiB |
| 1 | 8 | all eight | 88 GiB |

At LNC=2 each logical core is a pair and so owns a partition outright, which is why
the per-core weights column above is the number that matters.

### What stays on the host

Everything left is either not a neural network or cannot move; together it is
~25 ms of a request:

| Work | Cost | Why it stays |
|---|---|---|
| CLIP + T5 tokenization | ~1 ms | Text to token ids; no tensor math to place |
| Sampling initial latents | 6 ms | Seeded host RNG, so a seed reproduces an image at any `tp_degree` |
| Sigma / timestep schedule | ~1 ms | A few dozen scalars |
| RoPE table construction | one-off per rank at load | Needs float64, which Neuron does not have |
| Latent unpacking before decode | ~1 ms | A permute+reshape of 512 KiB; privateuse1 has no contiguity-fixing copy |
| Denormalize and PIL encode | 16 ms | Ends in a `PIL.Image`, which is host-only by definition |

## Measured latency

Trn2 (`trn2.3xlarge`, `logical-neuroncore-config: 2`), BF16, batch 1,
`neuronx-cc -O1`, 512-token prompt budget. Warm numbers: median over 2 requests
after a discarded warmup request.

| `tp_degree` | Resolution | Steps | ms/step | Denoise | Prompt encode | VAE decode | **End to end** |
|---|---|---|---|---|---|---|---|
| 2 | 512x512 | 4 | 134 | 0.54 s | 0.06 s | 0.14 s | **0.74 s** |
| 4 | 512x512 | 4 | 80 | 0.32 s | 0.03 s | 0.14 s | **0.51 s** |
| 2 | 1024x1024 | 28 | 392 | 11.0 s | 0.06 s | 0.56 s | **11.65 s** |
| 4 | 1024x1024 | 28 | 214 | 6.0 s | 0.03 s | 0.57 s | **6.62 s** |

Doubling the ranks takes 1.84x off the step and 1.76x off the request. Step latency
is flat across step counts and stable to well under 1% — the same graph runs every
step, on static shapes, with nothing left on the host but a one-element fence read.

Denoising is 91% of a 1024x1024 request at `tp_degree=4`. What running the rest on
Neuron buys, against the same weights in CPU eager mode:

| Stage | On Neuron (tp=4) | CPU eager | Speedup |
|---|---|---|---|
| Prompt encode (CLIP + T5, 512-token budget) | 0.03 s | 1.62 s | 54x |
| VAE decode (1024x1024, five staged graphs) | 0.57 s | 7.55 s | 13x |

Attention dominates the step: 46 attentions over a joint sequence of 4608 tokens
(4096 image + 512 text) at 1024x1024. Measured in isolation at those shapes, the
NKI flash-attention kernel runs 4.9 ms/call against 22.1 ms for materialized SDPA,
which is why the kernel is the default.

### Startup

| Phase | Cost |
|---|---|
| Compilation, cold cache | ~4 min, dominated by the five VAE decode stages |
| Rank startup, warm cache | 100-140 s |
| First image in a process | +12 s at 1024x1024, loading the VAE NEFFs |

Rank startup is where the cost sits, and it grows with `tp_degree`: every rank
materializes the whole checkpoint before keeping its shares of it, and they take a
lock to do that one at a time so host memory holds. Per-step overhead from the
split is about a millisecond: the embeddings and latents stay in the ranks for the
whole request, so a step sends three scalars and gets back a one-element fence.

### Against NxD Inference

The same checkpoint, resolution and step count under NxD Inference, which divides
the four components the same way:

| Cores | This pipeline | NxD Inference |
|---|---|---|
| 2 | 392 ms/step | 406 ms/step |
| 4 | 214 ms/step | 217 ms/step |

Two independent implementations landing within 3% is a useful check on both.

### Reducing latency

- **More ranks.** `tp_degree=4` instead of 2 is 1.84x per step, at twice the
  cores.
- **Fewer steps.** Step count scales latency linearly, and FLUX.1-lite degrades
  gracefully: 8 steps still resolves the subject, materials and lighting, losing
  mostly fine texture and background detail against 28; 4 steps is a usable
  preview.

  ![FLUX.1-lite-8B at 4, 8 and 28 denoising steps, same prompt and seed](images/flux-1-lite-8b-steps.png)

  *Same prompt, guidance and seed at each step count; 1024x1024, downscaled.*

- **Lower resolution.** 512x512 is 2.7x faster per step than 1024x1024: the joint
  sequence drops from 4608 to 1536 tokens.
- **Shorter prompt budget.** `--max-sequence-length 256` removes 256 tokens from
  every attention in every step. Prompts longer than the budget are truncated.
- **`-O2` / `-O3`.** `--optimization-level` maps straight to `neuronx-cc -O`.
  Compiles slower; whether it runs faster is workload-dependent.

Reproduce with:

```bash
python examples/vllm_neuron/models/flux/benchmark.py \
    --tp 4 --sizes 512,1024 --steps 8,28 --iterations 2 --json flux_latency.json
```

## Accuracy

Numerics are BF16 throughout, matching how FLUX is normally served, with two
deliberate promotions to fp32: RoPE application and the fused Euler update, so
latent error does not accumulate across ~30 steps of BF16 addition.

**Logic equivalence.** `NeuronFluxTransformer` was run against upstream
`FluxTransformer2DModel.forward` on identical inputs, both fp32 on CPU: max
relative difference **2.9e-7**, i.e. fp32 rounding. This is what covers the
rewritten pieces — hoisted RoPE, the reimplemented rotation, patched GELU, latent
packing, and the timestep/guidance convention.

**BF16 fidelity.** One denoising step at 512x512 with a 256-token prompt, against
that fp32 reference:

| Path | mean abs error | cosine similarity |
|---|---|---|
| Neuron BF16 (this pipeline) | 0.0074 | 0.999969 |
| CPU BF16 eager (diffusers) | 0.1039 | 0.993771 |

Reference velocity has std 1.16, so the Neuron path lands ~14x closer to fp32 than
CPU BF16 does — Trainium accumulates matmuls in fp32 where CPU BF16 eager
accumulates in BF16. Separately, the NKI attention kernel matches an fp32 SDPA
reference to 7e-4 max absolute error at 1024x1024 joint-attention shapes.

**Sharding.** The split is exact rather than an approximation — verified layer by
layer on CPU, where each rank's partial products sum to the dense layer's result —
but it does reorder summations, and BF16 shows that. Same seed and prompt:

| | Steps | cos | max abs diff | latent std |
|---|---|---|---|---|
| 512x512, tp=2 vs tp=4 | 4 | 0.999545 | 0.74 | 1.18 |
| 1024x1024, tp=2 vs tp=4 | 28 | 0.998092 | 2.27 | 1.29 |

The 28-step figure is lower because reassociation error accumulates along the
chain, not because more of the model is sharded. Decoded, the two 1024x1024 images
differ by a mean of 1.71/255 per channel:

![FLUX.1-lite-8B at tp_degree=2](images/flux-1-lite-8b-tp2.png)

*`tp_degree=2`, otherwise identical to the `tp_degree=4` sample at the top of this
page.*

A sharding mistake looks nothing like this — it lands below cos 0.9 and is obvious
in the image.

**End to end.** Compared against the stock diffusers pipeline running BF16 on CPU,
same prompt, seed and schedule (512x512, 8 steps): latent cosine similarity 0.9993
after 1 step, 0.9916 after 8, and image PSNR 27.3 dB. The two images share
composition, subject, palette and lighting but differ in fine detail. That is
expected rather than a defect: a flow-matching ODE amplifies any per-step numerical
difference across steps, and per the table above the CPU BF16 side is the less
faithful of the two. Diffusion output is not reproducible bit-for-bit across
numerical backends — treat a seed as reproducible only within one `tp_degree` and
dtype.

## Limitations

- Batch 1 only.
- `tp_degree` 2, 4 or 8; there is no single-core configuration, and 8 needs an
  instance larger than trn2.3xlarge.
- Resolution, prompt budget and `tp_degree` are fixed at load time. Changing any of
  them means recompiling — there is no bucketing, because a diffusion step has no
  analogue of a variable sequence length.
- CLIP and the VAE decoder are replicated rather than sharded, so every rank
  computes them redundantly. Together they are 0.37 GiB and ~9% of a request.
- One `tp_degree>1` configuration per process: the ranks are forked, and a child
  forked after this process has initialized the Neuron runtime inherits a dead NRT
  handle. Run one configuration per process.
- Rank startup is serialized by a file lock, so it grows with `tp_degree`.
- A `neuronx-cc` failure terminates the process rather than raising.
- `logical-neuroncore-config: 1` is unsupported: half the compute per rank, and
  the transformer graph returns NaN there.
