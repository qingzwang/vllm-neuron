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

3. **GELU.** ``F.gelu``, ``F.gelu(approximate="tanh")`` and ``nn.GELU`` all fail to
   compile here with ``apply() takes no keyword arguments`` — the backend's own
   lowering, not anything in the model. The exact GELU written out with ``erf``
   compiles and agrees with ``F.gelu`` to 2.8e-07, so the 24 ``GELUActivation``
   modules in Swin's MLPs are swapped for that. (The tanh approximation also compiles,
   at 9.6e-05 from exact; there is no reason to accept that error.)

4. **Sine position embeddings.** ``OneFormerSinePositionEmbedding`` builds its table
   from ``arange`` + strided (step 2) slices + ``stack`` + ``flatten``, and the compiler
   rejects the result: ``[NCC_IBIR243] Access pattern out of bounds``. It does not need
   to run on device at all — at a pinned input size the table is a *constant*, a
   function of the spatial shape only. So it is computed once on the host and cached,
   which removes the failing pattern and a few hundred device ops with it.

5. **Reference points.** ``OneFormerPixelDecoderEncoderOnly.get_reference_points``
   builds its grid with ``meshgrid`` and then ``reshape(-1)`` on the non-contiguous
   result, which the device rejects outright: ``Expected self.is_contiguous() to be
   true, but got false``. Like the position tables it is a constant — a function of the
   level shapes and of ``valid_ratios``, which is all ones whenever the input is a
   full, unpadded rectangle. So it too is computed on the host, once, after asserting
   that the ratios really are ones and that the constant matches what upstream
   computes for the same input.

6. **The deformable attention's shape assert.** Upstream calls
   ``torch_compilable_check((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() ==
   sequence_length, ...)``. Asserting on a *tensor* creates an unbacked symbolic int,
   and Dynamo then refuses the graph: ``PendingUnbackedSymbolNotFound: Pending unbacked
   symbols {u0} not in returned outputs``. The condition is a compile-time constant here
   — both sides come from the pinned input size — so it is checked for real on the host
   and skipped once the tensors are on device.

7. **The pixel decoder's split.** Its forward slices the encoder output back into
   levels with sizes computed from *tensors*
   (``level_start_index[i + 1] - level_start_index[i]``) and then views each piece with
   tensor dimensions. Both make sizes data-dependent, and Dynamo stops at
   ``Could not guard on data-dependent expression 256*u0 < 2``. At a pinned input the
   splits are constants, so upstream's own source is transformed -- three exact string
   substitutions, asserted to apply -- to use Python ints.

8. **A comparison against a Python float.** `forward_prediction_heads` ends with
   ``... < 0.5`` to turn mask logits into a boolean attention mask. Comparing a tensor
   with a Python float makes the lowering materialize that scalar as **f64**, and the
   compiler then refuses the whole graph with ``[NCC_ESPP004] f64 dtype is not
   supported``. Comparing against a same-dtype scalar *tensor* is identical in result
   (verified bit-for-bit) and lowers cleanly. This is the only such comparison in the
   model, and it is the one that blocked the full model for six compile attempts.

All patches assert that the upstream code still looks the way they assume. A
transformers upgrade that moves either line turns into a loud failure at install time
rather than a silently unpatched model.
"""

from __future__ import annotations

import inspect
import textwrap
import types

import torch

from .bilinear import multi_scale_deformable_attention as _msda


class ErfGELU(torch.nn.Module):
    """Exact GELU, written out.

    ``x * 0.5 * (1 + erf(x / sqrt(2)))`` is the definition; the point is only that it
    reaches the compiler as arithmetic instead of as the op whose lowering breaks.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 0.5 * (1.0 + torch.erf(hidden_states * _INV_SQRT2))


_INV_SQRT2 = 0.7071067811865476

# Set by install(), so the per-model helpers below do not need it passed twice.
_installed_level_shapes: list = []


def replace_gelu(module: torch.nn.Module) -> int:
    """Swap every GELU activation in ``module`` for :class:`ErfGELU`. Returns the count."""
    from transformers.activations import GELUActivation

    replaced = 0
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, (torch.nn.GELU, GELUActivation)):
                setattr(parent, name, ErfGELU())
                replaced += 1
    return replaced


def cache_position_embeddings(model: torch.nn.Module) -> int:
    """Make every sine position embedding a cached constant instead of a computation.

    ``forward(shape, device, dtype, mask=None)`` depends on nothing but its arguments,
    so the result can be memoized per (shape, dtype). Fill the cache with a CPU forward
    pass, then :func:`move_position_cache` puts the tables on the device — which keeps
    the *building* of them (the part the compiler rejects) off the device entirely.

    Returns the number of modules patched.
    """
    from transformers.models.oneformer import modeling_oneformer as m

    patched = 0
    for module in model.modules():
        if not isinstance(module, m.OneFormerSinePositionEmbedding):
            continue
        if getattr(module, "_position_cache", None) is not None:
            continue
        module._position_cache = {}
        module._uncached_forward = module.forward

        def cached_forward(self, shape, device, dtype, mask=None, _orig=module._uncached_forward):
            _check(mask is None, "a masked sine position embedding cannot be cached by shape")
            key = (tuple(shape), str(dtype))
            hit = self._position_cache.get(key)
            if hit is None:
                # Built on the host on purpose. Anything else defeats the point.
                hit = _orig(torch.Size(shape), torch.device("cpu"), dtype, None)
                self._position_cache[key] = hit
            return hit

        module.forward = types.MethodType(cached_forward, module)
        patched += 1
    return patched


def move_position_cache(model: torch.nn.Module, device, dtype=None) -> int:
    """Move every cached position table to ``device``. Returns how many were moved."""
    moved = 0
    for module in model.modules():
        cache = getattr(module, "_position_cache", None)
        if not cache:
            continue
        for key, value in list(cache.items()):
            cache[key] = value.to(device=device, dtype=dtype or value.dtype)
            moved += 1
    return moved


# The pixel decoder's reference grid, as one constant. Filled by the CPU pass and then
# moved once by move_reference_cache(); the traced graph only ever reads it. Doing the
# move *inside* the graph is what produced "unimplemented _copy_from xla:0neuron:0" --
# during tracing the device HuggingFace passes around is an XLA device, not neuron.
_reference_cache: dict = {}


def cache_reference_points(model: torch.nn.Module, level_shapes) -> None:
    """Replace the pixel decoder's reference-point grid with a host-computed constant.

    ``get_reference_points(spatial_shapes, valid_ratios, device)`` depends only on the
    level shapes and the ratios. The ratios are ones for a full rectangular input --
    asserted here rather than assumed -- so the grid is fixed, and the ``meshgrid`` plus
    non-contiguous ``reshape`` that the device refuses never runs.
    """
    from transformers.models.oneformer import modeling_oneformer as m

    cls = m.OneFormerPixelDecoderEncoderOnly
    # A staticmethod accessed off the class is already the plain function here; keep
    # the descriptor form working too, in case a future version changes that.
    raw = cls.__dict__.get("get_reference_points", cls.get_reference_points)
    original = raw.__func__ if isinstance(raw, staticmethod) else raw
    _check(
        "meshgrid" in inspect.getsource(original),
        "get_reference_points no longer builds a meshgrid; re-check this patch",
    )

    _reference_cache.clear()

    def cached_reference_points(spatial_shapes, valid_ratios, device):
        _check(
            valid_ratios.shape[1] == len(level_shapes),
            f"valid_ratios covers {valid_ratios.shape[1]} levels but this patch was "
            f"installed for {len(level_shapes)}",
        )
        moved = _reference_cache.get("device")
        if moved is not None:
            return moved

        host = _reference_cache.get("host")
        if host is None:
            ones = torch.ones(
                valid_ratios.shape[0], len(level_shapes), 2, dtype=valid_ratios.dtype
            )
            host = original(level_shapes, ones, torch.device("cpu"))
            # Eager only: bool(tensor) inside a traced region breaks the graph, and
            # the point of the check is the host pass, where it is free.
            if valid_ratios.device.type == "cpu" and not torch.compiler.is_compiling():
                # The one moment both are on the host: prove the constant is the same
                # thing upstream would have produced, ratios included.
                _check(
                    bool(torch.equal(valid_ratios, ones)),
                    "valid_ratios are not all ones, so the reference grid is not a "
                    "constant — is the input padded?",
                )
                expected = original(level_shapes, valid_ratios, torch.device("cpu"))
                _check(
                    bool(torch.equal(host, expected)),
                    "the cached reference grid differs from upstream's",
                )
            _reference_cache["host"] = host
        return host

    cls.get_reference_points = staticmethod(cached_reference_points)


def constant_fold_pixel_decoder_split(level_shapes) -> None:
    """Give the pixel decoder's per-level split and view Python-int sizes.

    Implemented as a source transform rather than a hand copy: upstream's own function
    text is read, three exact substitutions are applied (and asserted to have applied),
    and the result is compiled in the module's namespace. A copy would drift silently;
    this way an upstream edit that moves any of the three lines fails loudly instead.
    """
    from transformers.models.oneformer import modeling_oneformer as m

    cls = m.OneFormerPixelDecoder
    source = textwrap.dedent(inspect.getsource(cls.forward))

    substitutions = [
        (
            "split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]",
            "split_size_or_sections[i] = _NEURON_LEVEL_STARTS[i + 1] - _NEURON_LEVEL_STARTS[i]",
        ),
        (
            "split_size_or_sections[i] = y.shape[1] - level_start_index[i]",
            "split_size_or_sections[i] = y.shape[1] - _NEURON_LEVEL_STARTS[i]",
        ),
        (
            "z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1])",
            "z.transpose(1, 2).view(bs, -1, _NEURON_LEVEL_SHAPES[i][0], _NEURON_LEVEL_SHAPES[i][1])",
        ),
    ]
    for old, new in substitutions:
        _check(old in source, f"pixel decoder forward no longer contains: {old}")
        source = source.replace(old, new)

    starts = [0]
    for height, width in level_shapes[:-1]:
        starts.append(starts[-1] + height * width)

    # The constants go on the real module, and the transformed function is compiled in
    # the module's own namespace: Dynamo resolves a function's globals against the
    # module it claims to come from, so a copied namespace is not enough (it fails with
    # "module ... has no attribute '_NEURON_LEVEL_SHAPES'" the moment tracing starts).
    m._NEURON_LEVEL_SHAPES = [tuple(int(v) for v in s) for s in level_shapes]
    m._NEURON_LEVEL_STARTS = starts
    namespace = vars(m)
    previous = namespace.get("forward")
    exec(compile(source, "<oneformer pixel decoder forward, patched>", "exec"), namespace)
    cls.forward = namespace["forward"]
    # Do not leave a stray module-level `forward` behind.
    if previous is None:
        namespace.pop("forward", None)
    else:
        namespace["forward"] = previous


def detensorize_mask_threshold() -> None:
    """Replace ``mask_logits < 0.5`` with a comparison against a scalar *tensor*.

    Measured, in isolation: ``(x < 0.5)`` fails to compile with ``[NCC_ESPP004] f64
    dtype is not supported``, while ``(x < torch.tensor(0.5, dtype=x.dtype))`` compiles
    and is bit-identical. Every other op in the same chain -- interpolate, sigmoid,
    repeat, flatten -- is fine, and so is arithmetic with Python floats elsewhere; it is
    specifically the comparison that promotes the constant to f64.

    Applied as a source transform on upstream's own method, like patch 7.
    """
    from transformers.models.oneformer import modeling_oneformer as m

    cls = m.OneFormerTransformerDecoder
    source = textwrap.dedent(inspect.getsource(cls.forward_prediction_heads))
    old = (
        "attention_mask.sigmoid().flatten(2).unsqueeze(1)"
        ".repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5"
    )
    new = (
        "attention_mask.sigmoid().flatten(2).unsqueeze(1)"
        ".repeat(1, self.num_heads, 1, 1).flatten(0, 1) "
        "< torch.tensor(0.5, dtype=attention_mask.dtype, device=attention_mask.device)"
    )
    _check(old in source, "the mask threshold comparison is not where it used to be")
    source = source.replace(old, new)

    namespace = vars(m)
    previous = namespace.get("forward_prediction_heads")
    exec(compile(source, "<oneformer forward_prediction_heads, patched>", "exec"), namespace)
    cls.forward_prediction_heads = namespace["forward_prediction_heads"]
    if previous is None:
        namespace.pop("forward_prediction_heads", None)
    else:
        namespace["forward_prediction_heads"] = previous


def relax_shape_assert() -> None:
    """Keep upstream's shape check on the host, drop it on device.

    ``torch_compilable_check`` is exactly the wrong shape of assert for this compiler:
    a tensor-valued condition becomes an unbacked symbol that has to appear in the
    graph's outputs, and Dynamo gives up. The check itself is still worth running -- it
    catches the level-shape mistakes this port has already made once -- so it runs
    whenever the condition can be evaluated on the host, which includes the whole CPU
    parity pass.
    """
    from transformers.models.oneformer import modeling_oneformer as m

    original = m.torch_compilable_check

    def host_only_check(cond, msg="", *args, **kwargs):
        if isinstance(cond, torch.Tensor):
            if cond.device.type != "cpu":
                return  # tracing on device: the condition is a static constant here
            cond = bool(cond.all())
        if not cond:
            text = msg() if callable(msg) else msg
            raise ValueError(text or "OneFormer shape check failed")

    m.torch_compilable_check = host_only_check
    m._oneformer_neuron_original_check = original


def move_reference_cache(device, dtype=None) -> bool:
    """Put the reference grid on ``device``, before compiling. Returns whether it moved."""
    host = _reference_cache.get("host")
    if host is None:
        return False
    _reference_cache["device"] = host.to(device=device, dtype=dtype or host.dtype)
    return True


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

    global _installed_level_shapes
    _installed_level_shapes = list(level_shapes)


def is_installed() -> bool:
    from transformers.models.oneformer import modeling_oneformer as m

    return getattr(m.OneFormerTransformerDecoderLayer, "_oneformer_neuron_patched", False)
