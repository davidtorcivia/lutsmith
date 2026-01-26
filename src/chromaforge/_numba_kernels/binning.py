"""Numba JIT-compiled pixel binning with Welford's online statistics.

Performance target: 8M pixels in < 2 seconds.

Uses serial Numba (nopython=True) to avoid race conditions on shared
accumulators. Still provides ~20-50x speedup over pure Python loops.

A pure NumPy fallback is provided for environments without Numba.
"""

from __future__ import annotations

import numpy as np

# Try to import Numba; provide fallback if unavailable
_HAS_NUMBA = False
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    pass


if _HAS_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _bin_pixels_numba(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        bin_res: int,
        pixel_x: np.ndarray,
        pixel_y: np.ndarray,
    ) -> tuple:
        """Numba-accelerated pixel binning with Welford statistics.

        Args:
            source_rgb: (M, 3) float32 source colors.
            target_rgb: (M, 3) float32 target colors.
            bin_res: Bin resolution per axis (e.g. 64 -> 64^3 bins).
            pixel_x: (M,) float32 normalized x coordinates.
            pixel_y: (M,) float32 normalized y coordinates.

        Returns:
            Tuple of (count, sum_input, mean_output, m2_output, sum_x, sum_y).
        """
        total_bins = bin_res * bin_res * bin_res
        count = np.zeros(total_bins, dtype=np.int64)
        sum_input = np.zeros((total_bins, 3), dtype=np.float64)
        mean_output = np.zeros((total_bins, 3), dtype=np.float64)
        m2_output = np.zeros((total_bins, 3), dtype=np.float64)
        sum_x = np.zeros(total_bins, dtype=np.float64)
        sum_y = np.zeros(total_bins, dtype=np.float64)

        M = source_rgb.shape[0]
        inv_res = 1.0 / bin_res  # not used, but bin_res is the scale factor

        for i in range(M):
            r = source_rgb[i, 0]
            g = source_rgb[i, 1]
            b = source_rgb[i, 2]

            # Clamp to [0, 1) to avoid out-of-bounds
            r = min(max(r, 0.0), 0.999999)
            g = min(max(g, 0.0), 0.999999)
            b = min(max(b, 0.0), 0.999999)

            ri = int(r * bin_res)
            gi = int(g * bin_res)
            bi = int(b * bin_res)

            # Flat index: b*res^2 + g*res + r (R fastest)
            bin_idx = bi * bin_res * bin_res + gi * bin_res + ri

            count[bin_idx] += 1
            n = count[bin_idx]

            # Accumulate input sums
            sum_input[bin_idx, 0] += source_rgb[i, 0]
            sum_input[bin_idx, 1] += source_rgb[i, 1]
            sum_input[bin_idx, 2] += source_rgb[i, 2]

            # Welford's online algorithm for mean and variance
            for ch in range(3):
                val = target_rgb[i, ch]
                delta = val - mean_output[bin_idx, ch]
                mean_output[bin_idx, ch] += delta / n
                delta2 = val - mean_output[bin_idx, ch]
                m2_output[bin_idx, ch] += delta * delta2

            # Spatial coordinates
            sum_x[bin_idx] += pixel_x[i]
            sum_y[bin_idx] += pixel_y[i]

        return count, sum_input, mean_output, m2_output, sum_x, sum_y


def bin_pixels_numpy(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    bin_res: int,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
) -> tuple:
    """Pure NumPy fallback for pixel binning.

    Uses np.bincount for aggregation. Does not compute Welford variance
    directly; instead computes sum and sum-of-squares for post-hoc variance.

    Returns same format as the Numba version.
    """
    M = source_rgb.shape[0]
    total_bins = bin_res ** 3

    # Compute bin indices
    indices = np.clip((source_rgb * bin_res).astype(np.int32), 0, bin_res - 1)
    flat = indices[:, 2] * bin_res * bin_res + indices[:, 1] * bin_res + indices[:, 0]

    # Counts
    count = np.bincount(flat, minlength=total_bins).astype(np.int64)

    # Input sums
    sum_input = np.zeros((total_bins, 3), dtype=np.float64)
    for ch in range(3):
        sum_input[:, ch] = np.bincount(
            flat, weights=source_rgb[:, ch].astype(np.float64), minlength=total_bins
        )

    # Output means and variance via sum and sum-of-squares
    sum_output = np.zeros((total_bins, 3), dtype=np.float64)
    sum_sq_output = np.zeros((total_bins, 3), dtype=np.float64)
    for ch in range(3):
        vals = target_rgb[:, ch].astype(np.float64)
        sum_output[:, ch] = np.bincount(flat, weights=vals, minlength=total_bins)
        sum_sq_output[:, ch] = np.bincount(flat, weights=vals * vals, minlength=total_bins)

    # Compute mean and M2 (for Welford-compatible output)
    mask = count > 0
    mean_output = np.zeros((total_bins, 3), dtype=np.float64)
    m2_output = np.zeros((total_bins, 3), dtype=np.float64)

    for ch in range(3):
        mean_output[mask, ch] = sum_output[mask, ch] / count[mask]
        # M2 = sum((x - mean)^2) = sum(x^2) - n * mean^2
        m2_output[mask, ch] = (
            sum_sq_output[mask, ch] - count[mask] * mean_output[mask, ch] ** 2
        )
        # Clamp to non-negative (numerical errors)
        m2_output[:, ch] = np.maximum(m2_output[:, ch], 0.0)

    # Spatial sums
    sum_x = np.bincount(flat, weights=pixel_x.astype(np.float64), minlength=total_bins)
    sum_y = np.bincount(flat, weights=pixel_y.astype(np.float64), minlength=total_bins)

    return count, sum_input, mean_output, m2_output, sum_x, sum_y


def bin_pixels(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    bin_res: int,
    pixel_x: np.ndarray | None = None,
    pixel_y: np.ndarray | None = None,
) -> tuple:
    """Bin pixel pairs into a 3D grid with running statistics.

    Automatically uses Numba if available, falls back to NumPy.

    Args:
        source_rgb: (M, 3) float32 source colors in [0, 1].
        target_rgb: (M, 3) float32 target colors.
        bin_res: Resolution per axis (e.g. 64).
        pixel_x: (M,) normalized x coordinates [0, 1]. Optional.
        pixel_y: (M,) normalized y coordinates [0, 1]. Optional.

    Returns:
        (count, sum_input, mean_output, m2_output, sum_x, sum_y)
    """
    M = source_rgb.shape[0]
    if pixel_x is None:
        pixel_x = np.zeros(M, dtype=np.float32)
    if pixel_y is None:
        pixel_y = np.zeros(M, dtype=np.float32)

    if _HAS_NUMBA:
        return _bin_pixels_numba(
            source_rgb.astype(np.float64),
            target_rgb.astype(np.float64),
            bin_res,
            pixel_x.astype(np.float64),
            pixel_y.astype(np.float64),
        )
    else:
        return bin_pixels_numpy(source_rgb, target_rgb, bin_res, pixel_x, pixel_y)
