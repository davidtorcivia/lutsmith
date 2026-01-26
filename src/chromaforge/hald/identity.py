"""Hald CLUT identity image generation.

A Hald CLUT identity image encodes a full 3D LUT as pixel colors.
When processed through a color grading pipeline and reconstructed,
it captures the exact transform without any regression.

Hald level L:
    - Image size: L^3 x L^3 pixels
    - LUT resolution: (L^2)^3 per axis
    - Level 8: 512x512 image -> 64^3 LUT
    - Level 12: 1728x1728 image -> 144^3 LUT
"""

from __future__ import annotations

import numpy as np


def generate_hald_identity(level: int = 8) -> np.ndarray:
    """Generate a Hald CLUT identity image (vectorized).

    Each pixel's color encodes its position in the 3D LUT grid.
    Processing this image through a color pipeline and then
    reconstructing gives you the exact LUT.

    Args:
        level: Hald level. Level L produces an L^3 x L^3 image
               encoding an (L^2)^3 LUT.

    Returns:
        (L^3, L^3, 3) float32 array with values in [0, 1].
    """
    if level < 2 or level > 16:
        raise ValueError(f"Hald level must be 2-16, got {level}")

    cube_size = level * level  # LUT resolution per axis
    img_size = level ** 3       # Image dimension

    # Vectorized coordinate generation
    x = np.arange(img_size, dtype=np.int32)
    y = np.arange(img_size, dtype=np.int32)
    xx, yy = np.meshgrid(x, y)  # xx varies along columns, yy along rows

    r = (xx % cube_size).astype(np.float32) / (cube_size - 1)
    g = ((xx // cube_size) + (yy % level) * level).astype(np.float32) / (cube_size - 1)
    b = (yy // level).astype(np.float32) / (cube_size - 1)

    img = np.stack([r, g, b], axis=-1)
    return img


def hald_image_size(level: int) -> int:
    """Return the image dimension for a given Hald level."""
    return level ** 3


def hald_lut_size(level: int) -> int:
    """Return the LUT grid size for a given Hald level."""
    return level * level
