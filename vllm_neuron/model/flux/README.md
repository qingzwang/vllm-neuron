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
| `config.py` | `FluxNeuronConfig`: resolution, prompt budget, TP degree, compiler flags |
| `attention.py` | `NeuronFluxAttnProcessor` — diffusers' joint attention over the NKI kernel |
| `transformer.py` | `NeuronFluxTransformer` — one denoising step as a single static graph |
| `parallel.py` | Tensor-parallel sharding for the transformer and the T5 encoder |
| `lora.py` | Dynamic LoRA: slot tensors on device, a shared device-side index, and adapter loading |
| `vae.py` | Staged VAE decode and the nearest-upsample rewrite |
| `worker.py` | What one rank runs: every network, on its core, plus the command protocol |
| `tp.py` | The ranks from the pipeline's side, over `utils.executor.MPExecutor` |
| `pipeline.py` | `NeuronFluxPipeline` — tokenize, drive the loop, postprocess |

## Parallelism

The model is tensor-parallel across `tp_degree` logical NeuronCores, one process
per core, because the compile backend loads every NEFF onto the process's own core
(`start_nc = 0` plus the distributed rank) and a core belongs to one process. Those
processes hold every network; the pipeline process holds none.

| Component | Across ranks | Why |
| --- | --- | --- |
| `transformer` | sharded | 15.2 GiB, 28 invocations per request: the whole cost |
| `text_encoder_2` (T5-XXL) | sharded | 8.9 GiB, divides cleanly on 64 heads |
| `text_encoder` (CLIP-L) | replicated | 0.22 GiB; sharding would add collectives to save nothing |
| `vae` (decoder) | replicated | 0.15 GiB, convolutional, once per request |

Same division as NxD Inference makes for this model. There is no `tp_degree=1`:
the four components are 24.44 GiB of BF16 weights against a ~22 GiB HBM partition.

Only attention heads and feed-forward widths are split; the residual stream stays
full width and identical on every rank. The one layer that does not fit that
pattern is the single-stream block's `proj_out`, whose input is the concatenation
of two column-parallel outputs — see `parallel.py`.

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

- **Batching.** Static shapes are built for batch 1.
- **Per-batch-item LoRA selection.** One adapter is live at a time, which is all
  batch 1 can use. The selection index is a device tensor, so making it per-item
  would be a shape change rather than a redesign.
- **LoRA on the text encoders.** Adapters that carry text-encoder weights have
  those tensors ignored with a warning.
- **Sharded CLIP and VAE.** Both are replicated, so every rank computes them
  redundantly. They are 0.37 GiB and ~4% of a request, so this costs little.
- **True classifier-free guidance**, img2img, inpainting, ControlNet, IP-Adapter,
  LoRA.
- **FLUX.1-schnell** and other `guidance_embeds=False` variants.
