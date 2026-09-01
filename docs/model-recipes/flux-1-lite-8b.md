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
| **Parallelism** | Two logical NeuronCores (transformer + T5 encoder) | ✅ |
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

![FLUX.1-lite-8B output on Trn2: a red panda in round glasses reading a book in a library](images/flux-1-lite-8b-sample.png)

*1024x1024, 28 steps, guidance 3.5, seed 42 — exactly the command above.
Downscaled for this page.*

The first run compiles every component (a few minutes, dominated by the VAE
decode stages); subsequent runs hit the compilation cache. Within a process, the
first request additionally pays for loading each NEFF onto the device — about
20 s extra at 1024x1024 — so warm latency only shows from the second request on.
`generate.py` therefore issues a discarded single-step request before the one it
reports; `--no-warmup` skips it.

```python
import os

# Before the import: leaves core 1 free for the T5 encoder worker. See
# "Reserving a core for the worker" below.
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline

config = FluxNeuronConfig(height=1024, width=1024, max_sequence_length=512)
with NeuronFluxPipeline.from_pretrained("Freepik/flux.1-lite-8B", config) as pipeline:
    pipeline.compile()
    image, timing = pipeline.generate(
        "a red panda reading a book", num_inference_steps=28, seed=42
    )
image.save("out.png")
print(timing.as_dict())
```

## Component placement

Every neural network runs on Neuron. Two logical NeuronCores are used, because
one is not enough to hold both large models:

| Component | Runs on | Size (BF16) |
|---|---|---|
| `transformer` (FluxTransformer2DModel) | Neuron, core 0 | 15.2 GiB |
| `vae` (AutoencoderKL decoder) | Neuron, core 0 | 0.15 GiB |
| `text_encoder` (CLIP-L) | Neuron, core 0 | 0.22 GiB |
| `text_encoder_2` (T5-XXL) | Neuron, core 1 (child process) | 8.9 GiB |

An HBM partition holds ~22 GiB, and 15.2 + 8.9 GiB exceeds that before any
activation memory (see "Choosing cores" for how HBM is divided). Nor can one
process use two cores: the compile backend loads every NEFF onto the process's
own core, so weights uploaded to a second core get rejected at execution time. T5
therefore runs in a child process pinned with `NEURON_RT_VISIBLE_CORES`, which
the pipeline manages for you; see `text_encoder_worker.py`. Structurally this is
the same split as `vllm_neuron/vllm/disaggregated_encoder` in the LLM path.

That buys 16x on prompt encoding: **98 ms** on Neuron against **1617 ms** for
the same weights in CPU eager mode, measured over 10 requests at a 512-token
budget.

## Choosing cores

### Which core a process runs on

`NEURON_RT_VISIBLE_CORES`, set **before** `vllm_neuron` is imported — that import
brings the Neuron runtime up, and the choice is latched then. It both restricts
and renumbers: the cores it names become indices `0..n-1` inside the process, so
a process pinned to physical core 2 addresses it as `torch.device("neuron", 0)`
and `device_index=0`.

```python
import os
os.environ["NEURON_RT_VISIBLE_CORES"] = "2"   # must precede the import

from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline
```

Accepted forms are a single index, a list, and inclusive ranges: `"2"`, `"0,2"`,
`"0-3"`. `NEURON_RT_NUM_CORES` also exists but is coarse — the runtime only
accepts one core or a whole device (a multiple of 8), and it does not say which
core a request for one got — so prefer `NEURON_RT_VISIBLE_CORES`.

A process that sets neither claims **every** core when the runtime initializes,
even if it only ever runs graphs on one. That is what makes pinning mandatory
here rather than optional: without it there is no core left for the T5 worker,
and the pipeline says so and keeps T5 on CPU:

```
Not starting the T5 encoder worker: this process claimed every logical
NeuronCore when the Neuron runtime initialized, ... Keeping T5 in-process,
which means CPU and ~1.5 s per request.
```

`FluxNeuronConfig` then has two core knobs, in **different namespaces**:

| Knob | Namespace | Usual value |
|---|---|---|
| `device_index` | Index into this process's *visible* cores | `0` |
| `worker_device_index` | *Physical* core id, becomes the child's `NEURON_RT_VISIBLE_CORES` | `1` |

The worker holds its core for the pipeline's lifetime, so use the pipeline as a
context manager or call `close()`.

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

So the budget follows the pair, and the LNC setting only decides whether that
pair is one logical core or two.

### Small models on eight cores (LNC=1)

`NEURON_LOGICAL_NC_CONFIG=1` splits each pair into two logical cores, giving
eight. Each is one physical NeuronCore-v3 — half the compute of an LNC=2 core —
which suits small models where a whole fused core would sit idle. Note the
variable is `NEURON_LOGICAL_NC_CONFIG`; `NEURON_RT_LOGICAL_NC_CONFIG` looks
plausible and is silently ignored.

Because HBM follows the pair, spread independent models across pairs rather than
across cores:

```bash
# Eight logical cores; each of these four has a full ~22 GiB partition to itself.
export NEURON_LOGICAL_NC_CONFIG=1
NEURON_RT_VISIBLE_CORES=0 python serve.py &   # partition 0
NEURON_RT_VISIBLE_CORES=2 python serve.py &   # partition 1
NEURON_RT_VISIBLE_CORES=4 python serve.py &   # partition 2
NEURON_RT_VISIBLE_CORES=6 python serve.py &   # partition 3
```

Pack two models onto one pair (`0` and `1`) only when their combined footprint —
weights plus activations — fits inside that one ~22 GiB partition. For this
pipeline it does not: at LNC=1, cores 0 and 1 would put the transformer and T5
back in the same budget that could not hold them. Use different pairs:

```bash
NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=0 \
    python examples/vllm_neuron/models/flux/generate.py --size 512
# with FluxNeuronConfig(worker_device_index=2)
```

A NEFF records the logical-core config it was built for, and the runtime refuses
to load one that disagrees. `NEURON_LOGICAL_NC_CONFIG` is *not* part of the
compilation cache key, so switching to LNC=1 would otherwise replay the LNC=2
NEFFs and drop every component to CPU. `FluxNeuronConfig.neuronx_cc_args` avoids
that by mirroring the runtime setting into a `--lnc` compiler flag, which both
compiles the right NEFF and gives each config its own cache entry. Nothing to set
by hand — but expect a full recompile the first time you switch.

For FLUX, LNC=2 is the better setting. Measured at 512x512 / 8 steps, everything
on device:

| | ms/step | End to end |
|---|---|---|
| LNC=2 (one fused core) | 281 | 2.47 s |
| LNC=1 (one physical core) | 333 | 2.92 s |

Half the compute per core costs only 19% more per step, so LNC=1 is not a
disaster for this model — but it is a loss, and it exists to let *other*, smaller
models share the chip, not to speed this one up.

### Overrides

`on_device` (or `--on-device`) moves a component to CPU for an A/B comparison,
and `text_encoder_worker=False` keeps T5 in this process — which means CPU,
since it cannot fit next to the transformer. A component that fails to load,
compile or start falls back to CPU with a warning rather than taking the
pipeline down. One exception: a `neuronx-cc` failure terminates the process
outright, so that fallback cannot catch it.

### What stays on the host

Everything left is either not a neural network or cannot move; together it is
~25 ms of a 22.8 s request:

| Work | Cost | Why it stays |
|---|---|---|
| CLIP + T5 tokenization | ~1 ms | Text to token ids; no tensor math to place |
| Sampling initial latents | 6 ms | Seeded host RNG, so a seed reproduces an image regardless of placement |
| Sigma / timestep schedule | ~1 ms | A few dozen scalars |
| RoPE table construction | one-off at load | Needs float64, which Neuron does not have |
| Latent unpacking before decode | ~1 ms | A permute+reshape of 512 KiB; privateuse1 has no contiguity-fixing copy |
| Denormalize and PIL encode | 16 ms | Ends in a `PIL.Image`, which is host-only by definition |

## Measured latency

Trn2 (`trn2.3xlarge`, two logical NeuronCores, `logical-neuroncore-config: 2`),
BF16, batch 1, `neuronx-cc -O1`, 512-token prompt budget, default placement.
Warm numbers: median over 2 requests after a discarded warmup request.

| Resolution | Steps | ms/step | Denoise | Prompt encode | VAE decode | **End to end** |
|---|---|---|---|---|---|---|
| 512x512 | 8 | 281 | 2.24 s | 0.10 s | 0.12 s | **2.47 s** |
| 512x512 | 28 | 279 | 7.81 s | 0.10 s | 0.12 s | **8.04 s** |
| 1024x1024 | 8 | 792 | 6.34 s | 0.11 s | 0.50 s | **6.97 s** |
| 1024x1024 | 28 | 791 | 22.16 s | 0.10 s | 0.50 s | **22.78 s** |

Step latency is flat across step counts and stable to well under 1% (p90 within
0.7 ms of p50) — the same graph runs every step, on static shapes, with nothing
left on the host but a one-element fence read.

Denoising is 97% of a 1024x1024 request. What the other placements bought:

| Stage | On Neuron | CPU eager | Speedup |
|---|---|---|---|
| Prompt encode (CLIP + T5, 512-token budget) | 0.10 s | 1.62 s | 16x |
| VAE decode (1024x1024, five staged graphs) | 0.50 s | 7.55 s | 15x |

Attention dominates: 46 attentions per step over a joint sequence of 4608 tokens
(4096 image + 512 text) at 1024x1024. Measured in isolation at those shapes, the
NKI flash-attention kernel runs 4.9 ms/call against 22.1 ms for materialized
SDPA, which is why the kernel is the default.

Compilation, first time on a cold cache:

| Component | Graphs | Compile time (1024x1024, `-O1`) |
|---|---|---|
| `transformer` | 1 | ~2 min |
| `vae` | 5 | ~2 min |
| `text_encoder` (CLIP) | 1 | ~1 s |
| `text_encoder_2` (T5, in the worker) | 1 | ~20 s |

Warm restarts hit the local compilation cache and take ~55 s to reach the first
request, almost all of it loading weights from disk — the worker loads T5 in
parallel with the parent's own startup.

### Reducing latency

- **Fewer steps.** Step count scales latency linearly, and FLUX.1-lite degrades
  gracefully: 8 steps (7.0 s) still resolves the subject, materials and lighting,
  losing mostly fine texture and background detail against 28 steps (22.8 s);
  4 steps (3.8 s) is a usable preview.

  ![FLUX.1-lite-8B at 4, 8 and 28 denoising steps, same prompt and seed](images/flux-1-lite-8b-steps.png)

  *Same prompt, guidance and seed at each step count; 1024x1024, downscaled.*

- **Lower resolution.** 512x512 is 2.8x faster per step than 1024x1024: the
  joint sequence drops from 4608 to 1536 tokens. At 8 steps that is a 2.5 s
  request.
- **Shorter prompt budget.** `--max-sequence-length 256` removes 256 tokens from
  every attention in every step. Prompts longer than the budget are truncated.
- **`-O2` / `-O3`.** `--optimization-level` maps straight to `neuronx-cc -O`.
  Compiles slower; whether it runs faster is workload-dependent.

Reproduce with:

```bash
python examples/vllm_neuron/models/flux/benchmark.py \
    --sizes 512,1024 --steps 8,28 --iterations 2 --json flux_latency.json
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

The T5 encoder on its worker core agrees with the same weights in CPU eager mode
to cosine similarity 0.99945 (mean abs 0.0035) over a 512-token prompt — BF16
rounding again, and by the argument above the on-device side is the more faithful
one. The final image is unchanged by the move.

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

- No tensor parallelism. The pipeline uses two logical cores, but for placement
  rather than speed: the transformer still runs on one. On a trn2 chip's four
  cores, sharding the step graph is the highest-value next change -- it is 97% of
  a request.
- Batch 1 only.
- Resolution and prompt budget are fixed at load time. Changing either means
  recompiling — there is no bucketing, because a diffusion step has no analogue
  of a variable sequence length.
- The T5 encoder worker needs a free logical core, which means the main process
  must be pinned with `NEURON_RT_VISIBLE_CORES` before `vllm_neuron` is imported.
  Without that T5 falls back to CPU and costs ~1.5 s per request.
- A `neuronx-cc` failure terminates the process rather than raising, so the
  CPU fallback cannot catch it; HBM allocation failures are caught.
