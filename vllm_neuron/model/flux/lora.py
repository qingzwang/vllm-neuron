# SPDX-License-Identifier: Apache-2.0
"""Dynamic LoRA for the FLUX transformer.

Adapters are loaded, swapped and selected without recompiling anything. Two facts
make that possible:

* **An in-place write to a device tensor is visible to an already-compiled graph.**
  Verified directly: after `copy_` into a parameter, a buffer or a plain tensor
  attribute, the next call through the compiled module returns the new value, and
  Dynamo reports no additional graph. So adapter weights can live in device tensors
  the NEFF reads, and loading an adapter is a host-to-device copy.
* **The selection index can be a device tensor too.** Every wrapped module reads
  the *same* one-element `slot` tensor and does `index_select` on its slot
  dimension. Switching adapters is therefore a single 4-byte copy, not a copy of
  the weights -- which matters, because a full adapter is hundreds of MB spread over
  ~4300 small tensors and takes seconds to move.

That is the whole design: `max_loras` slots resident on device, slot 0 reserved for
"no adapter" (kept zero), and a shared index saying which slot is live.

## Where the delta goes under tensor parallelism

The base layers are already sharded (see `parallel.py`), so each adapter has to be
sharded to match -- and for row-parallel layers *where* the delta is added matters:

| Base layer | `lora_A` | `lora_B` | Delta added |
|---|---|---|---|
| column-parallel (`to_q`, `ff.net.0.proj`, `proj_mlp`, ...) | replicated | rows sliced | after the base; there is no reduce |
| row-parallel (`to_out.0`, `ff.net.2`, ...) | columns sliced | replicated | **before** the all-reduce |
| single-block `proj_out` | both halves' columns sliced | replicated | **before** the all-reduce |
| plain (`norm1.linear`, top-level `proj_out`, ...) | replicated | replicated | after the base |

The row-parallel case is the one that silently breaks if you get it wrong. There
`x` is sharded, so `A` must be sharded with it and each rank can only compute a
partial `A @ x`; `B` is replicated, so `B @ allreduce(A_local @ x_local)` equals
`allreduce(B @ (A_local @ x_local))` and the delta can ride the base layer's
existing all-reduce. Add it *after* that reduce instead and every rank ends up with
a different, wrong answer.

## Checkpoint formats

Community FLUX adapters come in at least three key conventions (diffusers/PEFT,
kohya, XLabs). The file is handed to `FluxPipeline.lora_state_dict`, which
dispatches to diffusers' own converters, so all three arrive in the diffusers key
space -- and from there the names match this module tree directly, because the
sharding in `parallel.py` kept diffusers' own names.

One exception needs arithmetic rather than renaming. The single-stream block's
`proj_out` consumes `cat([attn_output, mlp_hidden_states])`, and this
implementation shards those two halves separately, so the adapter's `lora_A` is cut
at `attn_dim` into two pieces while `lora_B` is used by both:

    B @ (A @ [x_a ; x_m]) == B @ (A[:, :d] @ x_a) + B @ (A[:, d:] @ x_m)

which is exact, not an approximation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .parallel import ColumnParallelLinear, RowParallelLinear, _SingleBlockProjOut, tp_world

logger = logging.getLogger(__name__)

_TRANSFORMER_PREFIX = "transformer."

# Module suffixes a FLUX adapter can target, as diffusers names them. The union of
# what real adapters touch: a kohya adapter covers all of these, an XLabs one only
# the double-block attention projections. Wrapping a module no adapter targets
# costs an unused slot row, so this is the full set rather than a guess per adapter.
FLUX_LORA_TARGETS = (
    # double-stream (MMDiT) blocks
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "norm1.linear",
    "norm1_context.linear",
    # single-stream blocks (their attention projections share the suffixes above)
    "proj_mlp",
    "proj_out",
    "norm.linear",
)


def load_lora_state_dict(
    path: str, weight_name: Optional[str] = None
) -> dict[str, torch.Tensor]:
    """Load a FLUX adapter from disk into the diffusers key space.

    Accepts anything ``FluxPipeline.lora_state_dict`` accepts -- diffusers/PEFT,
    kohya or XLabs.

    Args:
        path: Directory or file holding the adapter.
        weight_name: Specific file inside ``path``, when it holds more than one.

    Returns:
        ``{"transformer.<module path>.lora_{A,B}.weight": tensor}``.

    Raises:
        ValueError: If the adapter has nothing for the transformer, which means it
            is either not a FLUX adapter or adapts only the text encoders.
    """
    from diffusers import FluxPipeline

    state_dict = FluxPipeline.lora_state_dict(path, weight_name=weight_name)
    if isinstance(state_dict, tuple):  # newer diffusers also returns metadata
        state_dict = state_dict[0]

    keys = [k for k in state_dict if k.startswith(_TRANSFORMER_PREFIX)]
    if not keys:
        raise ValueError(
            f"No transformer weights in the adapter at {path}. Keys start with "
            f"{sorted({k.split('.')[0] for k in state_dict})}. Either this is not a "
            "FLUX adapter, or it only adapts the text encoders, which is not "
            "supported here."
        )
    dropped = len(state_dict) - len(keys)
    if dropped:
        logger.warning(
            "Ignoring %d non-transformer tensors in the adapter at %s "
            "(text-encoder adapters are not supported)",
            dropped,
            path,
        )
    return {k: state_dict[k] for k in keys}


def group_lora_pairs(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Group a diffusers-space adapter into ``{module path: (lora_A, lora_B)}``.

    Args:
        state_dict: Output of :func:`load_lora_state_dict`.

    Returns:
        One entry per adapted module, keyed by its path in this module tree (the
        ``transformer.`` prefix removed).

    Raises:
        ValueError: If a module has only one of the two matrices, which would
            otherwise be applied as a half-adapter.
    """
    halves: dict[str, dict[str, torch.Tensor]] = {}
    for key, weight in state_dict.items():
        name = key.removeprefix(_TRANSFORMER_PREFIX)
        module_path, _, tail = name.rpartition(".lora_")
        if not module_path:
            logger.warning("Skipping unrecognized LoRA key %r", key)
            continue
        halves.setdefault(module_path, {})["A" if tail.startswith("A") else "B"] = weight

    pairs = {}
    for module_path, half in halves.items():
        if set(half) != {"A", "B"}:
            raise ValueError(
                f"{module_path} has only lora_{sorted(half)[0]}; both matrices are "
                "needed to apply an adapter."
            )
        pairs[module_path] = (half["A"], half["B"])
    return pairs


def lora_rank(pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> int:
    """The adapter's rank, i.e. the widest ``lora_A`` it contains."""
    if not pairs:
        raise ValueError("adapter is empty")
    return max(a.shape[0] for a, _ in pairs.values())


class LoraSlots:
    """The slot table shared by every wrapped module in one transformer.

    Attributes:
        max_loras: Slots, including slot 0 which stays zero and means "no adapter".
        max_rank: Width every slot is allocated at. A narrower adapter is
            zero-padded into it, which is exact -- the padded rows of ``lora_A``
            pair with padded columns of ``lora_B``.
        slot: One-element int32 device tensor holding the live slot. Shared by
            every module, so switching adapters is one 4-byte copy.
    """

    def __init__(self, max_loras: int, max_rank: int, device: torch.device) -> None:
        self.max_loras = max_loras
        self.max_rank = max_rank
        self.device = device
        self.slot = torch.zeros(1, dtype=torch.int32, device=device)
        self.modules: dict[str, LoraLinear | RowParallelLinear | _SingleBlockProjOut] = {}

    def select(self, slot: int) -> None:
        """Point every wrapped module at ``slot``.

        Args:
            slot: Slot index; 0 is the unmodified model.

        Raises:
            ValueError: If the slot does not exist. The graph does not bounds-check
                an index_select, so an out-of-range slot would read garbage.
        """
        if not 0 <= slot < self.max_loras:
            raise ValueError(
                f"slot {slot} is out of range; this model was built with "
                f"max_loras={self.max_loras}, so slots are 0..{self.max_loras - 1}."
            )
        self.slot.copy_(torch.tensor([slot], dtype=torch.int32))

    def bytes_per_slot(self) -> int:
        """Device bytes one slot costs on this rank."""
        total = 0
        for module in self.modules.values():
            for weight in (module.lora_A, module.lora_B):
                total += weight[0].numel() * weight.element_size()
        return total


def _allocate(
    module: nn.Module,
    slots: LoraSlots,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
) -> None:
    """Give ``module`` its slot tensors and a reference to the shared index."""
    module.lora_slots = slots
    module.lora_A = torch.zeros(
        slots.max_loras, slots.max_rank, in_features, dtype=dtype, device=slots.device
    )
    module.lora_B = torch.zeros(
        slots.max_loras, out_features, slots.max_rank, dtype=dtype, device=slots.device
    )


def lora_delta(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """This rank's LoRA contribution for ``x``, for the live slot.

    Args:
        module: A module carrying ``lora_A``/``lora_B``/``lora_slots``.
        x: The same input the base layer sees.

    Returns:
        ``x @ A.T @ B.T``, shaped like the base layer's local output.
    """
    index = module.lora_slots.slot
    a = torch.index_select(module.lora_A, 0, index)[0]
    b = torch.index_select(module.lora_B, 0, index)[0]
    return torch.matmul(torch.matmul(x, a.transpose(0, 1)), b.transpose(0, 1))


def lora_delta_split(
    module: nn.Module, attn_part: torch.Tensor, mlp_part: torch.Tensor
) -> torch.Tensor:
    """The single block's LoRA contribution, summed over its two input halves.

    ``lora_B`` is shared by the halves, which is what makes the split exact::

        B @ (A @ [x_a ; x_m]) == B @ (A_a @ x_a + A_m @ x_m)

    so the two ``lora_A`` products are added before ``lora_B`` is applied, rather
    than each half being projected separately.

    Args:
        module: A ``_SingleBlockProjOut`` carrying ``lora_A``, ``lora_A_mlp`` and
            ``lora_B``.
        attn_part: This rank's slice of the attention half of the input.
        mlp_part: This rank's slice of the MLP half.

    Returns:
        This rank's partial delta, to be added before the layer's all-reduce.
    """
    index = module.lora_slots.slot
    a_attn = torch.index_select(module.lora_A, 0, index)[0]
    a_mlp = torch.index_select(module.lora_A_mlp, 0, index)[0]
    b = torch.index_select(module.lora_B, 0, index)[0]
    inner = torch.matmul(attn_part, a_attn.transpose(0, 1)) + torch.matmul(
        mlp_part, a_mlp.transpose(0, 1)
    )
    return torch.matmul(inner, b.transpose(0, 1))


class LoraLinear(nn.Module):
    """A base layer whose output gets a LoRA delta added to it.

    For layers with no all-reduce of their own: plain ``nn.Linear`` and
    column-parallel layers, whose local output is a slice of the columns and whose
    delta is the matching slice.

    Args:
        base: The layer to wrap. Kept as-is.
        slots: The shared slot table.
        in_features: Width of the input the adapter sees. Full width here -- these
            layers take the unsharded activation.
        out_features: Width of *this rank's* output, so the delta lines up.
    """

    def __init__(
        self,
        base: nn.Module,
        slots: LoraSlots,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base = base
        _allocate(self, slots, in_features, out_features, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + lora_delta(self, x)


def wrap_with_lora(
    transformer: nn.Module,
    max_loras: int,
    max_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    tp_group=None,
) -> LoraSlots:
    """Give every adaptable layer of a sharded FLUX transformer its slot tensors.

    Call after :func:`parallel.shard_flux_transformer` and after the module has been
    moved to the device: the slots are allocated there, and the compiled graph reads
    them in place.

    Args:
        transformer: The sharded ``FluxTransformer2DModel``.
        max_loras: Slots including slot 0, which stays zero.
        max_rank: Slot width; adapters may be narrower, not wider.
        device: Where the slots live.
        dtype: Slot dtype, matching the model's.
        tp_group: Process group the model is sharded over.

    Returns:
        The slot table, to keep for loading and selecting adapters.
    """
    tp_size, _, _ = tp_world(tp_group)
    slots = LoraSlots(max_loras, max_rank, device)

    for name, module in list(transformer.named_modules()):
        if not name.endswith(FLUX_LORA_TARGETS):
            continue
        parent_path, _, leaf = name.rpartition(".")
        parent = transformer.get_submodule(parent_path) if parent_path else transformer

        if isinstance(module, _SingleBlockProjOut):
            # Two halves, each sharded along its own axis, both feeding one reduce.
            _allocate(module, slots, module.attn_shard, module.out_features, dtype)
            module.lora_A_mlp = torch.zeros(
                max_loras, max_rank, module.mlp_shard, dtype=dtype, device=device
            )
            slots.modules[name] = module
        elif isinstance(module, RowParallelLinear):
            # x is sharded, so lora_A is too; lora_B is replicated and the delta
            # rides the layer's own all-reduce.
            _allocate(module, slots, module.weight.shape[1], module.out_features, dtype)
            slots.modules[name] = module
        elif isinstance(module, ColumnParallelLinear):
            wrapped = LoraLinear(
                module, slots, module.in_features, module.out_features_per_rank, dtype
            )
            _set_submodule(parent, leaf, wrapped)
            slots.modules[name] = wrapped
        elif isinstance(module, nn.Linear):
            wrapped = LoraLinear(
                module, slots, module.in_features, module.out_features, dtype
            )
            _set_submodule(parent, leaf, wrapped)
            slots.modules[name] = wrapped
        else:
            continue

    logger.info(
        "FLUX transformer wrapped for LoRA: %d modules, %d slots at rank %d, "
        "%.0f MB per slot on this rank (tp=%d)",
        len(slots.modules),
        max_loras,
        max_rank,
        slots.bytes_per_slot() / 2**20,
        tp_size,
    )
    return slots


def _set_submodule(parent: nn.Module, leaf: str, module: nn.Module) -> None:
    """Assign ``module`` at ``leaf`` on ``parent``, supporting numeric names."""
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def load_adapter_into_slot(
    slots: LoraSlots,
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    slot: int,
    attn_dim: int,
    tp_group=None,
) -> int:
    """Copy one adapter into a device slot, sharded to match this rank.

    Every wrapped module's slot is written, so a module this adapter does not touch
    contributes nothing rather than whatever the slot held before.

    Args:
        slots: The slot table from :func:`wrap_with_lora`.
        pairs: ``{module path: (lora_A, lora_B)}`` from :func:`group_lora_pairs`.
        slot: Which slot to write. Must not be 0, which is the unmodified model.
        attn_dim: The transformer's hidden size, for splitting the single block's
            ``proj_out``.
        tp_group: Process group the model is sharded over.

    Returns:
        Number of modules written.

    Raises:
        ValueError: If ``slot`` is 0 or out of range, or the adapter's rank exceeds
            the slot width, or a matrix does not match the module it names.
    """
    if slot == 0:
        raise ValueError("slot 0 is the unmodified model and cannot be written")
    if not 0 < slot < slots.max_loras:
        raise ValueError(f"slot {slot} is out of range 1..{slots.max_loras - 1}")
    rank = lora_rank(pairs)
    if rank > slots.max_rank:
        raise ValueError(
            f"adapter rank {rank} exceeds the slot width {slots.max_rank} this "
            f"model was built for; rebuild with lora_max_rank={rank} or higher."
        )

    tp_size, tp_rank, _ = tp_world(tp_group)

    # Every wrapped module is written, not just the ones the adapter names: a slot
    # has to be fully defined, or a module the adapter leaves alone would still
    # contribute whatever the slot's previous occupant put there.
    #
    # Each slot tensor is written with a single copy_ of a host-side padded matrix.
    # The obvious alternative -- zero the slot then write the corner the adapter
    # fills -- does not work: an in-place `zero_()` on a slot view is rejected by
    # the backend with "Can't call ReserveSpace on shared storage". One whole-slot
    # copy avoids that and halves the number of device writes.
    written = 0
    for path, module in slots.modules.items():
        pair = pairs.get(path)
        if pair is not None:
            written += 1
        a, b = (None, None) if pair is None else pair

        if isinstance(module, _SingleBlockProjOut):
            # lora_A is cut at attn_dim; lora_B is shared by both halves.
            a_attn = None if a is None else _shard_cols(a[:, :attn_dim], tp_size, tp_rank)
            a_mlp = None if a is None else _shard_cols(a[:, attn_dim:], tp_size, tp_rank)
            _write_slot(module.lora_A, slot, a_attn)
            _write_slot(module.lora_A_mlp, slot, a_mlp)
            _write_slot(module.lora_B, slot, b)
        elif isinstance(module, RowParallelLinear):
            _write_slot(module.lora_A, slot, None if a is None else _shard_cols(a, tp_size, tp_rank))
            _write_slot(module.lora_B, slot, b)
        else:  # LoraLinear over a column-parallel or plain layer
            _write_slot(module.lora_A, slot, a)
            _write_slot(module.lora_B, slot, None if b is None else _shard_rows(b, tp_size, tp_rank))

    missing = sorted(set(pairs) - set(slots.modules))
    if missing:
        logger.warning(
            "Adapter targets %d modules that are not adapted here, e.g. %s. An "
            "adapter trained against a different FLUX variant -- one with a "
            "different number of double-stream blocks -- looks like this.",
            len(missing),
            missing[:3],
        )
    return written


def _shard_cols(weight: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    """This rank's slice along the input dimension."""
    if tp_size == 1:
        return weight
    shard = weight.shape[1] // tp_size
    return weight[:, tp_rank * shard : (tp_rank + 1) * shard]


def _shard_rows(weight: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    """This rank's slice along the output dimension."""
    if tp_size == 1:
        return weight
    shard = weight.shape[0] // tp_size
    return weight[tp_rank * shard : (tp_rank + 1) * shard]


def _write_slot(
    slot_tensor: torch.Tensor, slot: int, weight: Optional[torch.Tensor]
) -> None:
    """Write one slot in full: ``weight`` padded with zeros, or all zeros.

    Args:
        slot_tensor: ``[max_loras, rows, cols]`` on device.
        slot: Which slot to overwrite.
        weight: The matrix to place at the top-left, or None for an untouched
            module.

    Raises:
        ValueError: If the weight does not fit, which means the adapter does not
            match the module it names.
    """
    target = slot_tensor[slot]
    padded = torch.zeros(target.shape, dtype=slot_tensor.dtype)
    if weight is not None:
        if weight.shape[0] > target.shape[0] or weight.shape[1] > target.shape[1]:
            raise ValueError(
                f"adapter matrix {tuple(weight.shape)} does not fit the slot "
                f"{tuple(target.shape)}"
            )
        padded[: weight.shape[0], : weight.shape[1]] = weight.to(slot_tensor.dtype)
    target.copy_(padded)
