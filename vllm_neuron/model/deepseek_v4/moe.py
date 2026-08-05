# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 mixture-of-experts feed-forward.

Each layer routes every token to ``num_experts_per_tok`` of
``n_routed_experts`` experts, plus one always-on shared expert. The first
``num_hash_layers`` layers select experts by a fixed token-id table
(``tid2eid``) instead of scoring, which makes their routing independent of the
hidden state.

Routed experts are stored FP4 in the checkpoint and dequantized to bf16 at load
time (Trainium has no FP4 datapath), so the compute here is plain bf16 matmul.
Experts are sharded across the expert-parallel group; the intermediate dim of
the shared expert is sharded across TP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DeepseekV4Expert(nn.Module):
    """The always-on shared expert: a TP-sharded SwiGLU MLP.

    ``w1`` is the gate projection, ``w3`` the up projection and ``w2`` the down
    projection, following the checkpoint's naming.
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
    """Routed experts + shared expert.

    The routed experts are held as stacked ``[E_local, ...]`` parameters so the
    dispatch is a gather-matmul rather than a Python loop over modules, which
    keeps the traced graph a fixed size regardless of expert count.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.routed_scaling_factor = config.routed_scaling_factor
        self.normalize_weights = config.norm_topk_prob
        self.is_hash_layer = layer_idx < config.num_hash_layers
        self.expert_dtype = config.expert_dtype

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        neuron_config = config.neuron_config
        self.ep_degree = neuron_config.ep_degree if neuron_config else 1
        if self.num_experts % self.ep_degree:
            raise ValueError(
                f"n_routed_experts ({self.num_experts}) must be divisible by "
                f"ep_degree ({self.ep_degree})"
            )
        self.num_local_experts = self.num_experts // self.ep_degree
        self.ep_rank = self.rank % self.ep_degree if self.ep_degree > 1 else 0
        self.expert_start = self.ep_rank * self.num_local_experts

        # TP shards the expert intermediate dim; EP shards the expert count.
        tp_within_ep = self.world_size // self.ep_degree if self.ep_degree > 1 else self.world_size
        self.tp_within_ep = max(1, tp_within_ep)
        intermediate = config.moe_intermediate_size
        if intermediate % self.tp_within_ep:
            raise ValueError(
                f"moe_intermediate_size ({intermediate}) must be divisible by "
                f"the per-EP TP degree ({self.tp_within_ep})"
            )
        self.intermediate_per_rank = intermediate // self.tp_within_ep

        # ── Router ────────────────────────────────────────────────────────
        self.router_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype)
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
                torch.empty(self.num_experts, dtype=torch.float32)
            )

        # ── Routed experts (stacked) ──────────────────────────────────────
        self.expert_w1 = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_per_rank,
                self.hidden_size,
                dtype=self.dtype,
            )
        )
        self.expert_w3 = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_per_rank,
                self.hidden_size,
                dtype=self.dtype,
            )
        )
        self.expert_w2 = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_per_rank,
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
        # w1 / w3: shard the intermediate (output) dim across TP.
        for param in (self.expert_w1, self.expert_w3):
            set_weight_loader(
                param,
                expert_loader(
                    num_local_experts=self.num_local_experts,
                    shard_dim=0,
                    shard_size=self.intermediate_per_rank,
                    num_shards=self.tp_within_ep,
                    out_dtype=self.dtype,
                ),
            )
        # w2: shard the intermediate (input) dim; caller all-reduces.
        set_weight_loader(
            self.expert_w2,
            expert_loader(
                num_local_experts=self.num_local_experts,
                shard_dim=1,
                shard_size=self.intermediate_per_rank,
                num_shards=self.tp_within_ep,
                out_dtype=self.dtype,
            ),
        )

    def route(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(affinities, selected)``, both ``[T, num_experts]``."""
        if self.is_hash_layer:
            if input_ids is None:
                raise ValueError(
                    f"layer {self.layer_idx} uses hash routing and needs input_ids"
                )
            return compute_hash_router_scores(
                hidden_states,
                self.router_weight,
                self.tid2eid,
                input_ids,
                self.routed_scaling_factor,
                self.normalize_weights,
            )
        return compute_router_scores(
            hidden_states,
            self.router_weight,
            self.router_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.normalize_weights,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run routed + shared experts over ``[T, H]`` and return ``[T, H]``."""
        affinities, _ = self.route(hidden_states, input_ids)
        routed = self._run_routed_experts(hidden_states, affinities)

        # Both paths produce TP-partial sums, so a single all-reduce covers the
        # routed experts, the shared expert, and (under EP) the expert split.
        output = routed + self.shared_expert(hidden_states)
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output.to(hidden_states.dtype)

    def _run_routed_experts(
        self, hidden_states: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        """Dense-masked expert dispatch.

        Every local expert is evaluated over all tokens and scaled by its
        routing affinity, which is zero for the experts a token did not select.
        This keeps the traced graph shape-static — no data-dependent gather
        sizes — which is what the Neuron compiler needs; the cost is that a
        token pays for every local expert rather than just its top-k.

        Args:
            hidden_states: ``[T, H]``.
            affinities: ``[T, num_experts]`` global affinities; this rank slices
                out the experts it owns.

        TODO: Replace with the plugin's blockwise MoE kernels
        (``NF.moe_cte`` / ``NF.moe_block_tkg``) once the FP4-dequantized
        stacked-weight layout is wired up to them. The kernels want
        ``[E, H, 2, I]`` fused gate/up weights and an affinity scatter, so this
        is a layout change rather than a math change.
        """
        # Slice this rank's expert range out of the global affinity matrix.
        local_affinity = affinities[
            :, self.expert_start : self.expert_start + self.num_local_experts
        ]

        # Evaluate each local expert over all tokens: [E, T, I].
        gate = torch.einsum("th,eih->eti", hidden_states, self.expert_w1)
        up = torch.einsum("th,eih->eti", hidden_states, self.expert_w3)
        activated = swiglu(gate, up, self.swiglu_limit)
        # Scale by the per-(token, expert) affinity before the down projection,
        # so unselected experts contribute exactly zero.
        activated = activated * local_affinity.T.unsqueeze(-1)
        return torch.einsum(
            "eti,ehi->th", activated.to(hidden_states.dtype), self.expert_w2
        )
