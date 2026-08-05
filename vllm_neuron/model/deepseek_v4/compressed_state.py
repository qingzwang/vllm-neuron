# SPDX-License-Identifier: Apache-2.0
"""Why DeepSeek-V4's compressed KV streams are model-owned.

This module holds no code — it documents a design constraint that
:class:`~.attention.DeepseekV4Attention` and
:meth:`~.model_bf16.DeepseekV4ForCausalLM.bind_kv_cache` both refer to.

vLLM's paged KV cache addresses one slot per token via ``slot_mapping``. The
CSA / HCA streams have a different time granularity — one slot per
``compress_ratio`` tokens (4 or 128) — so the runner's slot mapping cannot
address them, and they cannot live in paged blocks without changes to the
allocator.

They are therefore held in buffers owned by each attention layer
(``compressed_kv``, ``compressed_index_kv``, ``compress_window_hidden``,
``compressed_length``), allocated in ``bind_kv_cache()`` and sized from
``max_model_len / compress_ratio``. Only the sliding-window stream is declared
to the runner in ``get_kv_spec()``.

Consequences:

* **No prefix-cache reuse of compressed state.** vLLM may hand back cached
  blocks for the sliding-window stream, but the compressed stream is rebuilt
  from the prompt on every prefill. Prefix caching therefore only helps the
  window, and a prefill whose blocks are entirely cache hits still pays the
  compressor cost.
* **No cross-request block sharing** of compressed state. The buffers hold one
  request's stream at a time, which is why ``max_num_seqs`` must be 1 for
  correctness today.
* **Decode appends are position-driven.** A compressed slot is finalized only
  once a whole window of ``compress_ratio`` tokens has passed, so decode writes
  a new slot every ``compress_ratio``-th step, fed by a rolling buffer of the
  most recent hidden states.

TODO: Fold these streams into the paged allocator as extra KV cache groups with
their own block size, so prefix caching, block sharing and batching apply. That
needs ``LayerSpec`` to express a per-group token-to-slot stride, which the
current runner contract does not have.
"""
