# SPDX-License-Identifier: Apache-2.0
"""The two changes HuggingFace's OneFormer needs before it will compile for Neuron.

Both are forced by ``probe_device_ops.py``, not by taste:

1. **Deformable attention.** ``F.grid_sample`` aborts the runtime here, so
   :mod:`bilinear` reimplements the op. The upstream call site passes
   ``spatial_shapes`` as a *tensor*; ours needs the level sizes as Python ints so the
   per-level ``split`` is a compile-time constant. Since the input resolution is
   pinned anyway, the sizes are known at install time and are passed in.

2. **The fully-masked-row guard.** Upstream writes

       attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

   which is a data-dependent index — Dynamo cannot trace it with ``fullgraph=True``,
   and the NaN it exists to avoid does *not* behave the same on device as on CPU
   (probe ``softmax_all_masked``: relative difference 6e+04). The arithmetic form
   ``mask & ~mask.all(-1, keepdim=True)`` is identical in effect, static in shape, and
   keeps the graph whole.

Both patches assert that the upstream code still looks the way they assume. A
transformers upgrade that moves either line turns into a loud failure at install time
rather than a silently unpatched model.
"""

from __future__ import annotations

import inspect

import torch

from .bilinear import multi_scale_deformable_attention as _msda


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(
            f"OneFormer patch does not fit this transformers version: {message}. "
            "Re-read modeling_oneformer.py and update src/patches.py."
        )


def install(level_shapes: list[tuple[int, int]]) -> None:
    """Patch ``transformers.models.oneformer.modeling_oneformer`` in place.

    Args:
        level_shapes: ``[(H_l, W_l), ...]`` for the pixel decoder's feature levels,
            **in the order the model itself uses**, which is smallest map first:
            stride 32, then 16, then 8. For a pinned 384x384 input that is
            ``[(12, 12), (24, 24), (48, 48)]``.

    The order matters and getting it wrong is silent: the reversed list sums to the
    same number of positions, so the shapes still "fit" while every level is sampled
    from the wrong feature map. That mistake cost a debugging round here, so the patch
    now cross-checks the caller's own ``value_spatial_shapes`` whenever it can do so
    without a device sync.
    """
    from transformers.models.oneformer import modeling_oneformer as m

    # ---------------------------------------------------------------- 1. deformable
    _check(
        hasattr(m, "multi_scale_deformable_attention"),
        "multi_scale_deformable_attention is gone",
    )
    _check(
        "grid_sample" in inspect.getsource(m.multi_scale_deformable_attention),
        "upstream deformable attention no longer uses grid_sample, so replacing it "
        "may no longer be necessary",
    )

    total = sum(h * w for h, w in level_shapes)

    def patched_msda(value, value_spatial_shapes, sampling_locations, attention_weights):
        # value: (B, sum(H*W), heads, dim). The shapes argument cannot be *used* — it
        # arrives as a device tensor and reading it would put a host sync in the middle
        # of the graph — but it can be checked when it is already on the host, which is
        # exactly the case during the CPU parity check.
        _check(
            value.shape[1] == total,
            f"value has {value.shape[1]} positions but level_shapes sums to {total}; "
            "the pinned input size and the installed level shapes disagree",
        )
        if not torch.is_tensor(value_spatial_shapes) or value_spatial_shapes.device.type == "cpu":
            passed = (
                value_spatial_shapes.tolist()
                if torch.is_tensor(value_spatial_shapes)
                else [list(s) for s in value_spatial_shapes]
            )
            _check(
                [list(s) for s in level_shapes] == [list(s) for s in passed],
                f"the model passes level shapes {passed} but this patch was installed "
                f"with {[list(s) for s in level_shapes]}. Same total, different order "
                "or sizes: every level would be sampled from the wrong feature map",
            )
        return _msda(value, level_shapes, sampling_locations, attention_weights)

    m.multi_scale_deformable_attention = patched_msda

    # ------------------------------------------------------------------- 2. the guard
    layer_cls = m.OneFormerTransformerDecoderLayer
    source = inspect.getsource(layer_cls.forward)
    _check(
        "attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False"
        in source,
        "the fully-masked-row guard is not where it used to be",
    )
    _check(
        "level_index = index % self.num_feature_levels" in source,
        "the decoder layer's forward no longer selects the level this way",
    )

    def patched_forward(
        self,
        index,
        output,
        multi_stage_features,
        multi_stage_positional_embeddings,
        attention_mask=None,
        query_embeddings=None,
        output_attentions=False,
    ):
        """A copy of upstream's forward with one line rewritten.

        Wrapping instead of copying does not work: upstream's line executes whether or
        not it selects anything, and ``torch.where(cond)`` has a data-dependent output
        shape, which Dynamo refuses under ``fullgraph=True``. So the body is
        reproduced verbatim except for the guard, which becomes arithmetic.
        """
        level_index = index % self.num_feature_levels
        if attention_mask is not None:
            # was: attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False
            attention_mask = attention_mask & ~attention_mask.all(dim=-1, keepdim=True)

        output, cross_attn_weights = self.cross_attn(
            output,
            multi_stage_features[level_index],
            memory_mask=attention_mask,
            memory_key_padding_mask=None,
            pos=multi_stage_positional_embeddings[level_index],
            query_pos=query_embeddings,
        )

        output, self_attn_weights = self.self_attn(
            output,
            output_mask=None,
            output_key_padding_mask=None,
            query_pos=query_embeddings,
        )

        output = self.ffn(output)

        outputs = (output,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

    layer_cls.forward = patched_forward
    layer_cls._oneformer_neuron_patched = True


def is_installed() -> bool:
    from transformers.models.oneformer import modeling_oneformer as m

    return getattr(m.OneFormerTransformerDecoderLayer, "_oneformer_neuron_patched", False)
