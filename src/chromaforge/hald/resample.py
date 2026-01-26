"""LUT resampling between different grid sizes.

Needed because Hald levels produce specific LUT sizes (e.g., 64^3)
but common export sizes are different (e.g., 33^3, 65^3).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def resample_lut(
    lut: np.ndarray,
    target_size: int,
    order: int = 1,
) -> np.ndarray:
    """Resample a LUT from its current size to a different grid size.

    Uses scipy's map_coordinates for smooth interpolation.

    Args:
        lut: (N, N, N, 3) source LUT indexed as [r, g, b, ch].
        target_size: Desired output grid size.
        order: Interpolation order (1 = trilinear, 3 = cubic).

    Returns:
        (target_size, target_size, target_size, 3) resampled LUT.
    """
    source_size = lut.shape[0]

    if source_size == target_size:
        return lut.copy()

    # Generate target grid coordinates in source grid space
    target_coords = np.linspace(0, source_size - 1, target_size)
    rr, gg, bb = np.meshgrid(target_coords, target_coords, target_coords, indexing="ij")

    result = np.empty((target_size, target_size, target_size, 3), dtype=np.float32)

    for ch in range(3):
        result[:, :, :, ch] = map_coordinates(
            lut[:, :, :, ch],
            [rr.ravel(), gg.ravel(), bb.ravel()],
            order=order,
            mode="nearest",
        ).reshape((target_size, target_size, target_size))

    return result
