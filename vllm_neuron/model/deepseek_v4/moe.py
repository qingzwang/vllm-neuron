# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 mixture-of-experts feed-forward.

Each layer routes every token to ``num_experts_per_tok`` of
``n_routed_experts`` experts, plus one always-on shared expert. The first
``num_hash_layers`` layers select experts by a fixed token-id table
(``tid2eid``) instead of scoring, which makes their routing independent of the
hidden state.

Routed experts are stored FP4 in the checkpoint and dequantized to bf16 at load
time (Trainium has no FP4 datapath), then dispatched through the plugin's
blockwise MoE kernel (``NF.moe_cte``). The kernel only computes the experts a
token actually selected; evaluating every local expert densely instead makes the
43-layer graph exceed the compiler's instruction budget.
"""

import nki.language as nl
import torch
import torch.nn as nn
import torch.nn.functional as F

from nkilib.core.moe.moe_cte.moe_cte import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoECTEImplementation,
)
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF

from .config import DeepseekV4Config
from .layers import compute_hash_router_scores, compute_router_scores, swiglu
from .weight_loaders import (
    cast_weight_loader,
    fp4_expert_dequant_loader,
    fp8_dequant_weight_loader,
    fp8_expert_dequant_loader,
)

from vllm_neuron.utils.weight_loader import set_weight_loader

# Tokens per expert block in the blockwise kernel. Must be a multiple of 128.
MOE_BLOCK_SIZE = 256

# Smallest token count that may go through the blockwise kernel path.
# ``build_blockwise_mapping`` derives a flatten length of ``num_tokens // 16``
# and then evaluates ``num_tokens % flatten_len`` when deciding whether its NKI
# helpers apply — below 16 tokens that length is 0 and the check divides by zero.
# Decode (one token per sequence) therefore uses the dense path.
MIN_KERNEL_TOKENS = 16


class DeepseekV4Expert(nn.Module):
    """The always-on shared expert: a TP-sharded SwiGLU MLP.

    ``w1`` is the gate projection, ``w3`` the up projection and ``w2`` the down
    projection, following the checkpoint's naming. This one is dense (every token
    uses it), so it stays a plain matmul rather than going through the MoE kernel.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.swiglu_limit = config.swiglu_limit

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        intermediate = config.moe_intermediate_size
        if intermediate % self.world_size:
            raise ValueError(
                f"moe_intermediate_size ({intermediate}) must be divisible by "
                f"the TP degree ({self.world_size})"
            )
        self.intermediate_per_rank = intermediate // self.world_size

        self.w1 = nn.Parameter(
            torch.empty(self.intermediate_per_rank, self.hidden_size, dtype=self.dtype)
        )
        self.w3 = nn.Parameter(
            torch.empty(self.intermediate_per_rank, self.hidden_size, dtype=self.dtype)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_per_rank, dtype=self.dtype)
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        # w1 / w3 are column-parallel (shard the intermediate dim).
        for param in (self.w1, self.w3):
            set_weight_loader(
                param,
                fp8_dequant_weight_loader(
                    shard_dim=0,
                    shard_size=self.intermediate_per_rank,
                    num_shards=self.world_size,
                    out_dtype=self.dtype,
                ),
            )
        # w2 is row-parallel (shard the input dim); caller all-reduces.
        set_weight_loader(
            self.w2,
            fp8_dequant_weight_loader(
                shard_dim=1,
                shard_size=self.intermediate_per_rank,
                num_shards=self.world_size,
                out_dtype=self.dtype,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns a TP-partial ``[T, H]`` sum — the caller all-reduces."""
        gate = F.linear(hidden_states, self.w1)
        up = F.linear(hidden_states, self.w3)
        activated = swiglu(gate, up, self.swiglu_limit)
        return F.linear(activated.to(hidden_states.dtype), self.w2)


class DeepseekV4MoE(nn.Module):
    """Routed experts (blockwise kernel) + shared expert (dense).

    Parallelism follows the GPT-OSS layout:

    * **No EP** — every rank holds all experts, intermediate dim TP-sharded.
    * **EP** — rank *k* owns experts ``[k*L, (k+1)*L)`` with the intermediate dim
      sharded across ``tp_degree = world_size / ep_degree``.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.routed_scaling_factor = config.routed_scaling_factor
        self.normalize_weights = config.norm_topk_prob
        self.is_hash_layer = layer_idx < config.num_hash_layers
        self.expert_dtype = config.expert_dtype
        self.block_size = MOE_BLOCK_SIZE
        self.min_kernel_tokens = MIN_KERNEL_TOKENS

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self.ep_enabled = vllm_config.parallel_config.enable_expert_parallel

        # >>> PARALLELISM: EP / TP split for the expert weights <<<
        if self.ep_enabled:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )

            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
        else:
            self.ep_degree = 1
            self.ep_rank = 0
            self.tp_degree = self.world_size
            self.ep_tp_group = self.tp_group
        self.moe_group = self.tp_group

        self.total_num_experts = config.n_routed_experts
        if self.total_num_experts % self.ep_degree:
            raise ValueError(
                f"n_routed_experts ({self.total_num_experts}) must be divisible "
                f"by ep_degree ({self.ep_degree})"
            )
        self.num_local_experts = self.total_num_experts // self.ep_degree
        self.expert_start = self.ep_rank * self.num_local_experts

        intermediate = config.moe_intermediate_size
        if intermediate % self.tp_degree:
            raise ValueError(
                f"moe_intermediate_size ({intermediate}) must be divisible by "
                f"the per-EP TP degree ({self.tp_degree})"
            )
        self.intermediate_per_rank = intermediate // self.tp_degree

        # ── Router ────────────────────────────────────────────────────────
        self.router_weight = nn.Parameter(
            torch.empty(self.total_num_experts, self.hidden_size, dtype=self.dtype)
        )
        if self.is_hash_layer:
            # Fixed token-id -> expert table. The pipelined checkpoint loader
            # only enumerates named_parameters(), so this buffer is loaded
            # explicitly in DeepseekV4ForCausalLM.load_weights().
            self.register_buffer(
                "tid2eid",
                torch.empty(config.vocab_size, self.top_k, dtype=torch.int32),
                persistent=False,
            )
            self.router_bias = None
        else:
            self.tid2eid = None
            self.router_bias = nn.Parameter(
                torch.empty(self.total_num_experts, dtype=torch.float32)
            )

        # ── Routed experts, in the kernel's layout ────────────────────────
        # gate_up: [E, H, 2, I_TP] with gate at index 0 and up at index 1.
        # down:    [E, I_TP, H].
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                2,
                self.intermediate_per_rank,
                dtype=self.dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_per_rank,
                self.hidden_size,
                dtype=self.dtype,
            )
        )

        self.shared_expert = DeepseekV4Expert(config)

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        set_weight_loader(
            self.router_weight, cast_weight_loader(out_dtype=self.dtype)
        )
        if self.is_hash_layer:
            set_weight_loader(
                self.tid2eid, cast_weight_loader(out_dtype=torch.int32)
            )
        else:
            set_weight_loader(
                self.router_bias, cast_weight_loader(out_dtype=torch.float32)
            )

        expert_loader = (
            fp4_expert_dequant_loader
            if self.expert_dtype == "fp4"
            else fp8_expert_dequant_loader
        )
        # gate_up_proj_weight fuses w1 (gate) and w3 (up). Both are stored
        # [I, H] per expert and shard on the intermediate dim; the loader
        # transposes to [H, I] so the pair can be stacked into [E, H, 2, I].
        set_weight_loader(
            self.gate_up_proj_weight,
            _fused_gate_up_loader(
                inner=expert_loader(
                    num_local_experts=self.num_local_experts,
                    shard_dim=0,
                    shard_size=self.intermediate_per_rank,
                    num_shards=self.tp_degree,
                    transpose=True,
                    out_dtype=self.dtype,
                ),
                num_local_experts=self.num_local_experts,
            ),
        )
        # down_proj_weight: stored [H, I] per expert, sharded on I, transposed
        # to the kernel's [E, I, H].
        set_weight_loader(
            self.down_proj_weight,
            expert_loader(
                num_local_experts=self.num_local_experts,
                shard_dim=1,
                shard_size=self.intermediate_per_rank,
                num_shards=self.tp_degree,
                transpose=True,
                out_dtype=self.dtype,
            ),
        )

    def route(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> torch.Tensor:
        """Compute ``[T, total_num_experts]`` affinities (0 where unselected)."""
        if self.is_hash_layer:
            if input_ids is None:
                raise ValueError(
                    f"layer {self.layer_idx} uses hash routing and needs input_ids"
                )
            affinities, _ = compute_hash_router_scores(
                hidden_states,
                self.router_weight,
                self.tid2eid,
                input_ids,
                self.routed_scaling_factor,
                self.normalize_weights,
            )
        else:
            affinities, _ = compute_router_scores(
                hidden_states,
                self.router_weight,
                self.router_bias,
                self.top_k,
                self.routed_scaling_factor,
                self.normalize_weights,
            )
        return affinities

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run routed + shared experts over ``[T, H]`` and return ``[T, H]``."""
        affinities = self.route(hidden_states, input_ids)

        # The blockwise kernel's helpers need enough tokens to tile
        # (``build_blockwise_mapping`` derives a flatten length of
        # ``num_tokens // 16``), so it only covers prefill. Decode carries one
        # token per sequence and falls back to the dense path, where evaluating
        # every local expert for a single token is cheap.
        if hidden_states.shape[0] >= self.min_kernel_tokens:
            routed = self._run_routed_experts_kernel(hidden_states, affinities)
        else:
            routed = self._run_routed_experts_dense(hidden_states, affinities)

        # Both paths produce TP-partial sums, so a single all-reduce covers the
        # routed experts, the shared expert, and (under EP) the expert split.
        output = routed + self.shared_expert(hidden_states)
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output.to(hidden_states.dtype)

    def _run_routed_experts_dense(
        self, hidden_states: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate every local expert and scale by its routing affinity.

        Shape-static and free of the kernel's tiling constraints, at the cost of
        computing experts a token did not select. Used for decode, where a single
        token makes that waste negligible; prefill uses the kernel.
        """
        local_affinity = affinities[
            :, self.expert_start : self.expert_start + self.num_local_experts
        ]
        # gate_up is [E, H, 2, I]; split the fused axis for the dense matmuls.
        gate_w = self.gate_up_proj_weight[:, :, 0, :]
        up_w = self.gate_up_proj_weight[:, :, 1, :]

        gate = torch.einsum("th,ehi->eti", hidden_states, gate_w)
        up = torch.einsum("th,ehi->eti", hidden_states, up_w)
        activated = swiglu(gate, up, self.swiglu_limit)
        activated = activated * local_affinity.T.unsqueeze(-1)
        return torch.einsum(
            "eti,eih->th", activated.to(hidden_states.dtype), self.down_proj_weight
        )

    def _run_routed_experts_kernel(
        self, hidden_states: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch tokens to their selected experts via the blockwise kernel.

        Args:
            hidden_states: ``[T, H]``.
            affinities: ``[T, total_num_experts]`` router affinities.

        Returns:
            ``[T, H]`` partial sum over this rank's experts.
        """
        # >>> PARALLELISM: EP — keep only this rank's slice of the affinities <<<
        if self.ep_degree > 1:
            local_expert_indices = torch.arange(
                self.expert_start,
                self.expert_start + self.num_local_experts,
                device=affinities.device,
                dtype=torch.int32,
            )
            affinities = NF.get_local_expert_affinities(
                affinities, local_expert_indices
            )

        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
        )

        return NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj_weight,
            down_proj_weight=self.down_proj_weight,
            # <-- MODEL-SPECIFIC: SwiGLU where the gate is clamped from above
            # only and the up projection on both sides. DeepSeek-V4 has no
            # expert bias, so the clamp limits are used as-is.
            activation_function=ActFnType.SiLU,
            gate_clamp_upper_limit=self.swiglu_limit,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=self.swiglu_limit,
            up_clamp_lower_limit=-self.swiglu_limit,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )


def _fused_gate_up_loader(inner, num_local_experts: int):
    """Stack the gate (w1) and up (w3) expert weights into ``[E, H, 2, I]``.

    The mapping supplies w1's ``(weight, scale)`` pairs for every local expert
    followed by w3's, so ``inner`` runs once per half and the two ``[E, H, I]``
    results are interleaved on a new axis — the layout ``moe_cte`` expects, with
    gate at index 0 and up at index 1.
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    def transform(slices: list, rank: int) -> torch.Tensor:
        expected = 4 * num_local_experts
        if len(slices) != expected:
            raise ValueError(
                f"fused gate/up loader expects {expected} slices "
                f"([weight, scale] per expert for w1 then w3), got {len(slices)}"
            )
        half = 2 * num_local_experts
        gate = inner.transform(slices[:half], rank)  # [E, H, I]
        up = inner.transform(slices[half:], rank)  # [E, H, I]
        return torch.stack([gate, up], dim=2).contiguous()

    return SafetensorsWeightLoader(transform=transform)
