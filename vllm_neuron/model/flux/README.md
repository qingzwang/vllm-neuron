# FLUX.1 on Neuron

Text-to-image inference for guidance-distilled FLUX.1 checkpoints, developed
against [`Freepik/flux.1-lite-8B`](https://huggingface.co/Freepik/flux.1-lite-8B).

## Why this is not a registered vLLM model

`vllm_neuron/model/registry.py` maps HuggingFace architecture names onto vLLM
model classes, and every entry there is a decoder-only LM or a pooling model:
`NeuronModelRunner` is built around tokens, a KV cache, and sampling. FLUX has
none of those. vLLM 0.24's `DiffusionConfig` does not help either — it describes
discrete diffusion *language* models, which denoise over a token canvas and reuse
the speculative-decoding data plane; FLUX denoises a latent image and emits
pixels.

So this module is a standalone pipeline rather than a registry entry. It does
reuse the parts of this package that are not LM-specific:

| Reused from `vllm_neuron` | Where |
| --- | --- |
| `torch.compile` Neuron backend + `neuronx-cc` flags | `config.neuronx_cc_args`, `pipeline._compile` |
| NKI flash-attention kernel (`functional.attention.attention_cte`) | `attention.neuron_joint_attention` |
| Warmup-driven compilation with static shapes | `pipeline.NeuronFluxPipeline.compile` |

## Layout

| File | Contents |
| --- | --- |
| `config.py` | `FluxNeuronConfig`: resolution, prompt budget, component placement, compiler flags |
| `attention.py` | `NeuronFluxAttnProcessor` — diffusers' joint attention over the NKI kernel |
| `transformer.py` | `NeuronFluxTransformer` — one denoising step as a single static graph |
| `vae.py` | Staged VAE decode and the nearest-upsample rewrite |
| `text_encoder_worker.py` | T5 on a second logical core, in a child process |
| `pipeline.py` | `NeuronFluxPipeline` — load, place, compile, generate, time |

## Placement

All four networks run on Neuron, across two logical cores:

| Core | Components |
| --- | --- |
| 0 (this process) | transformer, VAE decoder, CLIP |
| 1 (child process) | T5-XXL |

T5 needs its own core for two independent reasons: a core holds ~22 GiB of usable
HBM against 15.2 GiB of transformer plus 8.9 GiB of T5, and the compile backend
loads every NEFF onto the process's own core (`start_nc = 0` plus the distributed
rank), so one process cannot drive two. The caller has to leave a core free —
`NEURON_RT_VISIBLE_CORES` before importing `vllm_neuron` — or the pipeline says
why and keeps T5 on CPU. See the recipe.

## What had to change for Neuron

Upstream diffusers code runs unmodified except where it does not lower. Each
change is local and documented at the site:

1. **RoPE tables move to the host.** `FluxPosEmbed` builds them in float64.
   They depend only on resolution and prompt length, both static, so
   `build_rotary_embedding` computes them once and passes them in as tensors.
2. **`apply_rotary_emb` is reimplemented.** Upstream ends with
   `cos.to(x.device)`; an explicit device copy inside a traced graph lowers to an
   unimplemented `_copy_from xla:0 -> neuron:0`.
3. **GELU is recomputed from `tanh`.** `libtorch_neuronx_lite` replaces `F.gelu`
   with a C extension Dynamo cannot trace — same workaround as the Qwen3-VL
   vision encoder.
4. **Attention uses the NKI kernel, not SDPA.** At 1024x1024 with a 512-token
   prompt the joint sequence is 4608, and there are 46 attentions per step.
   Measured on trn2: 4.9 ms/call vs 22.1 ms for materialized SDPA.
5. **The scheduler update is folded into the step graph**, so latents stay on
   device across all ~30 steps instead of round-tripping to apply a two-term
   Euler update.
6. **VAE decode is compiled per resolution level.** End to end it lowers to
   ~10M instructions against `neuronx-cc`'s ~5M budget.
7. **Nearest-neighbour upsampling is rewritten as broadcast + reshape.**
   `F.interpolate(mode="nearest")` lowers to an indirect copy that the runtime
   rejects with an out-of-bounds access at these sizes.

## Usage

See `examples/vllm_neuron/models/flux/` and
`docs/model-recipes/flux-1-lite-8b.md`.

## Verified against upstream

`NeuronFluxTransformer` matches upstream `FluxTransformer2DModel.forward` to
2.9e-7 relative on identical fp32 inputs, which is what pins down the rewrites
above. In BF16 it lands 14x closer to that fp32 reference than CPU BF16 eager
does. Full numbers in the model recipe.

## Not implemented

- **Tensor parallelism.** The two cores are used for placement, not speed: the
  transformer runs on one. It is 97% of a request, so sharding the step graph
  across a chip's four cores is the highest-value next change.
- **Batching.** Static shapes are built for batch 1.
- **True classifier-free guidance**, img2img, inpainting, ControlNet, IP-Adapter,
  LoRA.
- **FLUX.1-schnell** and other `guidance_embeds=False` variants.
