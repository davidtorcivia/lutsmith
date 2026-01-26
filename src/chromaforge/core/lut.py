"""LUT data structure, identity generation, application, and utilities.

All LUTs use the convention: shape (N, N, N, 3), indexed as lut[r, g, b, ch].
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from chromaforge.config import LUT_CLAMP_MIN, LUT_CLAMP_MAX
from chromaforge.core.interpolation import apply_lut_to_colors
from chromaforge.core.types import LUTData, InterpolationKernel


def identity_lut(N: int) -> np.ndarray:
    """Generate an identity 3D LUT.

    Each node maps to its own normalized coordinate:
    lut[r, g, b] = (r/(N-1), g/(N-1), b/(N-1)).

    Args:
        N: Grid size per axis.

    Returns:
        (N, N, N, 3) float32 array.
    """
    coords = np.linspace(0.0, 1.0, N, dtype=np.float32)
    r, g, b = np.meshgrid(coords, coords, coords, indexing="ij")
    lut = np.stack([r, g, b], axis=-1)
    return lut


def identity_lut_flat(N: int) -> np.ndarray:
    """Generate identity LUT values as a flat (N^3, 3) array.

    Flat index convention: flat = b*N*N + g*N + r (R fastest).

    Returns:
        (N^3, 3) float32 array where row i contains the RGB identity
        value for flat index i.
    """
    lut = identity_lut(N)
    # Ravel in memory order: since lut is (N, N, N, 3) with [r, g, b, ch],
    # and our flat convention is b*N*N + g*N + r, we need to transpose
    # to (b, g, r, 3) before flattening.
    lut_bgr = np.transpose(lut, (2, 1, 0, 3))  # (N, N, N, 3) -> [b, g, r, ch]
    return lut_bgr.reshape(-1, 3).copy()


def apply_lut(
    image: np.ndarray,
    lut: np.ndarray,
    N: int,
    kernel: str = "tetrahedral",
) -> np.ndarray:
    """Apply a 3D LUT to an image.

    Args:
        image: (H, W, 3) or (M, 3) float32 array in [0, 1].
        lut: (N, N, N, 3) LUT array.
        N: Grid size.
        kernel: Interpolation kernel name.

    Returns:
        Transformed image with same shape as input.
    """
    original_shape = image.shape
    if image.ndim == 3 and len(original_shape) == 3 and original_shape[2] == 3:
        flat = image.reshape(-1, 3)
    elif image.ndim == 2 and original_shape[1] == 3:
        flat = image
    else:
        raise ValueError(f"Expected (H,W,3) or (M,3), got {original_shape}")

    result = apply_lut_to_colors(flat, lut, N, kernel)
    return result.reshape(original_shape)


def clip_lut(
    lut: np.ndarray,
    lo: float = LUT_CLAMP_MIN,
    hi: float = LUT_CLAMP_MAX,
) -> np.ndarray:
    """Clamp LUT values to a safe range.

    Args:
        lut: (N, N, N, 3) array.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clipped copy.
    """
    return np.clip(lut, lo, hi)


def lut_stats(lut: np.ndarray) -> dict:
    """Compute basic statistics of a LUT.

    Returns:
        Dict with min, max, mean per channel and overall.
    """
    return {
        "min_per_channel": lut.min(axis=(0, 1, 2)).tolist(),
        "max_per_channel": lut.max(axis=(0, 1, 2)).tolist(),
        "mean_per_channel": lut.mean(axis=(0, 1, 2)).tolist(),
        "global_min": float(lut.min()),
        "global_max": float(lut.max()),
    }


def lut_total_variation(lut: np.ndarray) -> float:
    """Compute normalized total variation of a LUT.

    Sum of absolute differences between adjacent nodes along each axis,
    normalized by the total number of node-channel pairs.
    """
    N = lut.shape[0]
    tv_r = np.sum(np.abs(np.diff(lut, axis=0)))
    tv_g = np.sum(np.abs(np.diff(lut, axis=1)))
    tv_b = np.sum(np.abs(np.diff(lut, axis=2)))
    return float((tv_r + tv_g + tv_b) / (3 * N ** 3))


def lut_neutral_monotonicity(lut: np.ndarray) -> tuple[bool, int]:
    """Check monotonicity along the neutral (gray) axis.

    The diagonal lut[i, i, i] should be monotonically increasing
    in all channels for a well-behaved LUT.

    Returns:
        (is_monotonic, num_violations)
    """
    N = lut.shape[0]
    neutral = np.array([lut[i, i, i] for i in range(N)])
    violations = 0
    for i in range(1, N):
        if np.any(neutral[i] < neutral[i - 1]):
            violations += 1
    return violations == 0, violations


def lut_oog_percentage(lut: np.ndarray) -> float:
    """Compute percentage of LUT values outside [0, 1]."""
    N = lut.shape[0]
    total_values = N ** 3 * 3
    oog = int(np.sum((lut < 0.0) | (lut > 1.0)))
    return float(oog / total_values * 100.0)


def create_lut_data(
    array: np.ndarray,
    kernel: InterpolationKernel = InterpolationKernel.TETRAHEDRAL,
    title: str = "ChromaForge LUT",
) -> LUTData:
    """Create a LUTData object from a raw array.

    Args:
        array: (N, N, N, 3) float32 array.
        kernel: Interpolation kernel used during fitting.
        title: Human-readable title.

    Returns:
        LUTData instance.
    """
    N = array.shape[0]
    return LUTData(
        array=array.astype(np.float32),
        size=N,
        title=title,
        kernel=kernel,
    )
