# SPDX-License-Identifier: Apache-2.0
"""Equal-area input sizes: one graph per aspect ratio, all costing the same.

The compiler needs a fixed input shape, so *some* resize is unavoidable. A single square
size is the simplest choice and it is what this port was brought up on, but it distorts
everything that is not square -- and OneFormer's own processor does not do that; its
default is aspect-preserving (``shortest_edge=800``, ``longest_edge=1333``).

The alternative is a small table of shapes that all have the same *area*, pick the one
whose aspect ratio is closest to the image's, and let the resize be nearly isotropic. Area
is the right invariant because it is what the cost is a function of. The whole model is
convolution and attention over the feature grid, and the flattened length the pixel
decoder attends over is::

    L = sum(H*W/s^2 for s in (32, 16, 8)) = 21*H*W/1024

-- area only. So every bucket here is within 4% of 640x640's 409600 pixels, and the
measured latencies should land within a few percent of each other. ``640x640`` is in the
table unchanged, which keeps every number in the README comparable.

What the table cannot do is *avoid* recompiling: 640x640 and 512x800 flatten to the same
8400 positions but are not the same graph, so each bucket is its own NEFF. That is cheap
here -- a NEFF is 50-59 MB and does *not* embed the 839 MB of weights, so the buckets
share one device copy of the model -- but it is not free, which is why the table is six
entries and not sixty.

Two hard constraints on any entry:

* **Both sides a multiple of 32.** Swin-L downsamples by 32 and the pixel decoder's
  coarsest level is stride 32, so anything else does not divide. Worse, it is silent:
  :func:`resize.fixed_bilinear_resize` falls back to ``F.interpolate`` when the ratio is
  not an even integer, and that fallback is the 9216x144 dense-resample matmul that
  patch 9 exists to remove.
* **No padding.** ``valid_ratios`` comes from ``pixel_mask``, so a padded input makes the
  pixel decoder's reference grid a function of the padding rather than a constant, and
  patch 5 asserts on exactly that. Letterboxing is therefore not an option; the resize
  has to fill the frame, which is why aspect ratio is approximated rather than preserved.
"""

from __future__ import annotations

import math

#: ``(height, width)``, landscape only -- :func:`pick` transposes for portrait. Chosen by
#: search: among all multiple-of-32 pairs within 2% of the named aspect ratio, the one
#: closest in area to 640x640. Aspect error is at most 1.5% (a 4:3 photo comes out 1.5%
#: wide, which is not visible) and area error at most 4%, so latency should be flat.
BUCKETS = (
    (640, 640),  # 1:1     409600 px  L=8400  mask 160x160
    (544, 736),  # 4:3     400384     L=8211       136x184
    (512, 768),  # 3:2     393216     L=8064       128x192
    (480, 864),  # 16:9    414720     L=8505       120x216
    (448, 896),  # 2:1     401408     L=8232       112x224
    (416, 960),  # 21:9    399360     L=8190       104x240
)

STRIDES = (32, 16, 8)


def level_shapes(height: int, width: int) -> list[tuple[int, int]]:
    """The pixel decoder's three feature-map shapes, smallest first.

    The order is the model's own -- stride 32, then 16, then 8 -- and it matters: reversed,
    the list sums to the same number of positions, so nothing catches it while every level
    is sampled from the wrong feature map. :func:`patches.pin_input_size` puts this on the
    model and :func:`patches.install`'s deformable attention cross-checks it against the
    shapes the model passes at runtime.
    """
    check_size(height, width)
    return [(height // s, width // s) for s in STRIDES]


def check_size(height: int, width: int) -> None:
    """Reject anything the port would silently mis-handle. See the module docstring."""
    for name, value in (("height", height), ("width", width)):
        if value <= 0 or value % 32:
            raise ValueError(
                f"{name}={value} must be a positive multiple of 32: Swin-L downsamples by "
                "32, and a size that does not divide makes resize.fixed_bilinear_resize "
                "fall back to the dense-matmul interpolate that patch 9 removes"
            )


def parse_size(size) -> tuple[int, int]:
    """``"640"`` -> ``(640, 640)``, ``"480x864"`` -> ``(480, 864)``.

    Also takes a bare int and an ``(H, W)`` pair, so a ``--size 640`` from the README and
    the ``size`` field of a reference ``.pt`` written before this existed both still mean
    what they did.
    """
    if isinstance(size, int):
        height = width = size
    elif isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError(f"a size pair is (H, W), not {size!r}")
        height, width = (int(v) for v in size)
    else:
        parts = str(size).lower().split("x")
        if len(parts) == 1:
            height = width = int(parts[0])
        elif len(parts) == 2:
            height, width = int(parts[0]), int(parts[1])
        else:
            raise ValueError(f"a size is H, or HxW, not {size!r}")
    check_size(height, width)
    return height, width


def tag(height: int, width: int) -> str:
    """The name a size gets in a filename: ``640`` if square, else ``480x864``.

    Square sizes keep the bare form so the paths in the README, and every reference
    ``.pt`` already on disk, still resolve.
    """
    return str(height) if height == width else f"{height}x{width}"


def pick(width: int, height: int) -> tuple[int, int]:
    """The bucket for an image ``width x height``, as ``(height, width)``.

    Argument order is the image's -- ``Image.size`` is ``(width, height)`` -- and the
    return order is the tensor's, ``(H, W)``. Portrait images are served by the transpose
    of the landscape bucket, which is exactly as good: the model has no preferred
    orientation and the table would otherwise be twice as long for nothing.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image size {width}x{height} is not an image")
    portrait = height > width
    aspect = max(width, height) / min(width, height)
    # Compared in log space, so being 10% too wide and 10% too narrow cost the same.
    best_h, best_w = min(BUCKETS, key=lambda hw: abs(math.log((hw[1] / hw[0]) / aspect)))
    return (best_w, best_h) if portrait else (best_h, best_w)


def describe(height: int, width: int) -> str:
    """One line for a log: the shapes and lengths this size implies."""
    shapes = level_shapes(height, width)
    return (
        f"{height}x{width} ({height * width} px, {height * width / 409600:.3f}x 640x640), "
        f"deformable levels {shapes}, L={sum(h * w for h, w in shapes)}, "
        f"mask {height // 4}x{width // 4}"
    )
