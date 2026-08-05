# SPDX-License-Identifier: Apache-2.0
"""Model-owned storage for DeepSeek-V4's compressed KV streams.

vLLM's paged KV cache addresses one slot per token via ``slot_mapping``. The
CSA / HCA streams have a different time granularity — one slot per
``compress_ratio`` tokens (4 or 128) — so the runner's slot mapping cannot
address them and they cannot live in paged blocks without changes to the
allocator.

This module holds them in model-owned buffers instead, sized by
``max_num_seqs x max_model_len / compress_ratio``. The consequences, which the
model README also states:

* **No prefix-cache reuse of compressed state.** vLLM may hand back cached
  blocks for the sliding-window stream, but the compressed stream is rebuilt
  from the prompt on every prefill. Prefix caching therefore only helps the
  window, and a prefill whose blocks are entirely cache hits still pays the
  compressor cost.
* **No cross-request block sharing.** Each active request owns a private slice
  keyed by its slot in the runner's batch.
* **Decode appends must be driven by position.** A compressed slot is finalized
  only when a whole window of ``compress_ratio`` tokens has passed, so decode
  writes a new slot on every ``compress_ratio``-th step.

TODO: Fold these streams into the paged allocator as extra KV cache groups with
their own block size, so prefix caching and block sharing apply. That needs
``LayerSpec`` to express a per-group token-to-slot stride, which the current
runner contract does not have.
"""

import torch


class CompressedKVState:
    """Per-layer compressed KV buffers for every request slot in the batch.

    Two streams are held: the attention stream (``head_dim`` wide, the values
    that CSA / HCA attend over) and, on CSA layers, the indexer's scoring stream
    (``index_head_dim`` wide).

    Buffers are allocated once at warmup so the traced graph sees fixed shapes.
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_slots: int,
        head_dim: int,
        index_head_dim: int | None,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_slots = max_slots
        self.head_dim = head_dim
        self.index_head_dim = index_head_dim

        self.kv = torch.zeros(
            max_num_seqs, max_slots, head_dim, dtype=dtype, device=device
        )
        self.index_kv = (
            torch.zeros(
                max_num_seqs, max_slots, index_head_dim, dtype=dtype, device=device
            )
            if index_head_dim is not None
            else None
        )
        # Number of finalized slots per request slot.
        self.lengths = torch.zeros(
            max_num_seqs, dtype=torch.int32, device=device
        )
        # Which request slot the current single-request prefill/decode targets.
        self._active_slot = 0

    def set_active_slot(self, slot: int) -> None:
        self._active_slot = slot

    def store(
        self, compressed_kv: torch.Tensor, index_kv: torch.Tensor | None
    ) -> None:
        """Overwrite the active request's compressed streams (prefill)."""
        num_slots = compressed_kv.shape[0]
        if num_slots > self.max_slots:
            raise ValueError(
                f"compressed stream needs {num_slots} slots but only "
                f"{self.max_slots} are allocated; raise max_model_len budget"
            )
        slot = self._active_slot
        self.kv[slot, :num_slots] = compressed_kv.to(self.kv.dtype)
        if index_kv is not None and self.index_kv is not None:
            self.index_kv[slot, :num_slots] = index_kv.to(self.index_kv.dtype)
        self.lengths[slot] = num_slots

    def append(
        self, compressed_kv: torch.Tensor, index_kv: torch.Tensor | None
    ) -> None:
        """Append one finalized slot for the active request (decode)."""
        slot = self._active_slot
        position = self.lengths[slot].clamp(max=self.max_slots - 1).long()
        self.kv[slot, position] = compressed_kv.to(self.kv.dtype).squeeze(0)
        if index_kv is not None and self.index_kv is not None:
            self.index_kv[slot, position] = index_kv.to(
                self.index_kv.dtype
            ).squeeze(0)
        self.lengths[slot] = torch.clamp(
            self.lengths[slot] + 1, max=self.max_slots
        )

    def load(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(attention_stream, indexer_stream)`` for the active request.

        The full allocated extent is returned; slots past the request's length
        hold stale values and are excluded by the caller's causal index mask.
        """
        slot = self._active_slot
        index_stream = (
            self.index_kv[slot] if self.index_kv is not None else None
        )
        return self.kv[slot], index_stream

    def reset(self, slot: int | None = None) -> None:
        """Clear one request slot, or all of them."""
        if slot is None:
            self.lengths.zero_()
        else:
            self.lengths[slot] = 0
