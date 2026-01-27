"""LUT reconstruction from processed Hald CLUT images.

Each pixel in a Hald image maps directly to a LUT grid node.
No regression is needed -- this is a direct readout.
"""

from __future__ import annotations

import logging

import numpy as np

from lutsmith.errors import ImageDimensionError
from lutsmith.hald.identity import hald_image_size, hald_lut_size

logger = logging.getLogger(__name__)


def reconstruct_from_hald(
    processed_img: np.ndarray,
    level: int = 8,
) -> np.ndarray:
    """Reconstruct a 3D LUT from a processed Hald identity image.

    The pixel at (x, y) maps to LUT node (r_idx, g_idx, b_idx) via:
        r_idx = x % cube_size
        g_idx = (x // cube_size) + (y % level) * level
        b_idx = y // level

    Args:
        processed_img: (L^3, L^3, 3) float32 processed Hald image.
        level: Hald level used to generate the identity.

    Returns:
        (cube_size, cube_size, cube_size, 3) float32 LUT array
        indexed as [r, g, b, ch].
    """
    img_size = hald_image_size(level)
    cube_size = hald_lut_size(level)

    # Validate dimensions
    if processed_img.shape[0] != img_size or processed_img.shape[1] != img_size:
        raise ImageDimensionError(
            f"Expected {img_size}x{img_size} image for Hald level {level}, "
            f"got {processed_img.shape[0]}x{processed_img.shape[1]}"
        )

    if processed_img.ndim == 2:
        processed_img = np.repeat(processed_img[:, :, np.newaxis], 3, axis=2)
    elif processed_img.shape[2] > 3:
        processed_img = processed_img[:, :, :3]

    # Vectorized reconstruction
    x = np.arange(img_size, dtype=np.int32)
    y = np.arange(img_size, dtype=np.int32)
    xx, yy = np.meshgrid(x, y)

    r_idx = xx % cube_size
    g_idx = (xx // cube_size) + (yy % level) * level
    b_idx = yy // level

    # Direct mapping: pixel (x, y) -> LUT node (r_idx, g_idx, b_idx)
    lut = np.zeros((cube_size, cube_size, cube_size, 3), dtype=np.float32)
    lut[r_idx, g_idx, b_idx] = processed_img[yy, xx]

    logger.info(
        "Reconstructed %d^3 LUT from Hald level %d (%dx%d image)",
        cube_size, level, img_size, img_size,
    )

    return lut
