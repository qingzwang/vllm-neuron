# SPDX-License-Identifier: Apache-2.0
"""FLUX.1 text-to-image support on Neuron.

Not registered in ``vllm_neuron.model.registry``: that registry maps HF
architecture names onto vLLM model classes, and vLLM 0.24 has no
text-to-image request path to hand a FLUX checkpoint to. This is a standalone
pipeline that reuses the package's compilation stack and NKI kernels. See
``pipeline.py`` for the rationale and ``docs/model-recipes/flux-1-dev.md``
for usage.
"""

from .attention import NeuronFluxAttnProcessor, neuron_joint_attention
from .config import SUPPORTED_TP_DEGREES, FluxNeuronConfig
from .lora import FLUX_LORA_TARGETS, load_lora_state_dict
from .parallel import shard_flux_transformer, shard_t5_encoder
from .pipeline import GenerationTiming, NeuronFluxPipeline
from .transformer import NeuronFluxTransformer, build_rotary_embedding

__all__ = [
    "FLUX_LORA_TARGETS",
    "SUPPORTED_TP_DEGREES",
    "FluxNeuronConfig",
    "GenerationTiming",
    "NeuronFluxAttnProcessor",
    "NeuronFluxPipeline",
    "NeuronFluxTransformer",
    "build_rotary_embedding",
    "load_lora_state_dict",
    "neuron_joint_attention",
    "shard_flux_transformer",
    "shard_t5_encoder",
]
