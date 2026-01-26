"""LUT validation and quality metrics computation.

Validates the fitted LUT by applying it to source samples and comparing
against targets using perceptual metrics (DeltaE 2000).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from chromaforge.config import (
    COVERAGE_DENSE_THRESHOLD,
    COVERAGE_SPARSE_THRESHOLD,
    VALIDATION_SUBSAMPLE_FRACTION,
)
from chromaforge.core.distance import compute_distance_to_data
from chromaforge.core.interpolation import apply_lut_to_colors
from chromaforge.core.lut import (
    lut_neutral_monotonicity,
    lut_oog_percentage,
    lut_total_variation,
)
from chromaforge.core.types import QualityMetrics

logger = logging.getLogger(__name__)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to CIELAB for DeltaE computation.

    Uses colour-science if available, otherwise a simplified conversion.
    """
    try:
        import colour

        rgb_clipped = np.clip(rgb, 0.0, 1.0)
        xyz = colour.sRGB_to_XYZ(rgb_clipped)
        lab = colour.XYZ_to_Lab(xyz)
        return lab
    except ImportError:
        # Simplified conversion (less accurate but functional)
        return _simple_rgb_to_lab(rgb)


def _simple_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Simplified sRGB -> Lab conversion without colour-science."""
    # Linearize sRGB
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((np.clip(rgb, 0, None) + 0.055) / 1.055) ** 2.4,
    )

    # sRGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ M.T

    # XYZ -> Lab (D65 white point)
    wp = np.array([0.95047, 1.00000, 1.08883])
    xyz_n = xyz / wp

    def f(t):
        delta = 6.0 / 29.0
        return np.where(
            t > delta ** 3,
            np.cbrt(np.maximum(t, 1e-10)),
            t / (3 * delta ** 2) + 4.0 / 29.0,
        )

    fx = f(xyz_n[..., 0])
    fy = f(xyz_n[..., 1])
    fz = f(xyz_n[..., 2])

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def _delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute CIE Delta E 2000 between two Lab arrays.

    Uses colour-science if available for maximum accuracy.
    """
    try:
        import colour
        return colour.delta_E(lab1, lab2, method="CIE 2000")
    except ImportError:
        # Simplified: use CIE76 (Euclidean in Lab) as fallback
        diff = lab1 - lab2
        return np.sqrt(np.sum(diff ** 2, axis=-1))


def validate_lut(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    lut: np.ndarray,
    N: int,
    kernel: str = "tetrahedral",
    occupied_flat_indices: Optional[np.ndarray] = None,
    subsample: bool = True,
) -> QualityMetrics:
    """Validate a LUT against source/target sample pairs.

    Args:
        source_rgb: (M, 3) source colors.
        target_rgb: (M, 3) target colors.
        lut: (N, N, N, 3) LUT array.
        N: LUT grid size.
        kernel: Interpolation kernel (must match fitting).
        occupied_flat_indices: For coverage computation.
        subsample: Whether to subsample for faster DeltaE.

    Returns:
        QualityMetrics with all computed metrics.
    """
    M = len(source_rgb)

    # Subsample for DeltaE (it can be slow for millions of samples)
    if subsample and M > 10000:
        n_sub = max(int(M * VALIDATION_SUBSAMPLE_FRACTION), 1000)
        rng = np.random.default_rng(42)
        idx = rng.choice(M, size=n_sub, replace=False)
        src_sub = source_rgb[idx]
        tgt_sub = target_rgb[idx]
    else:
        src_sub = source_rgb
        tgt_sub = target_rgb

    # Apply LUT using the SAME kernel used during fitting
    predicted = apply_lut_to_colors(src_sub, lut, N, kernel)

    # Convert to Lab
    pred_lab = _rgb_to_lab(predicted)
    tgt_lab = _rgb_to_lab(tgt_sub)

    # Compute DeltaE 2000
    delta_e = _delta_e_2000(pred_lab, tgt_lab)

    # Find max error location
    max_idx = int(np.argmax(delta_e))
    max_location = predicted[max_idx].tolist()

    # LUT health metrics
    tv = lut_total_variation(lut)
    monotonic, mono_violations = lut_neutral_monotonicity(lut)
    oog = lut_oog_percentage(lut)

    # Coverage
    num_occupied = 0
    coverage_pct = 0.0
    if occupied_flat_indices is not None:
        num_occupied = len(occupied_flat_indices)
        coverage_pct = 100.0 * num_occupied / N ** 3

    metrics = QualityMetrics(
        mean_delta_e=float(np.mean(delta_e)),
        median_delta_e=float(np.median(delta_e)),
        p95_delta_e=float(np.percentile(delta_e, 95)),
        max_delta_e=float(np.max(delta_e)),
        max_delta_e_location=max_location,
        total_variation=tv,
        neutral_monotonic=monotonic,
        neutral_mono_violations=mono_violations,
        oog_percentage=oog,
        coverage_percentage=coverage_pct,
        num_occupied_bins=num_occupied,
        num_total_bins=N ** 3,
    )

    logger.info(
        "Validation: mean_dE=%.2f, p95_dE=%.2f, max_dE=%.2f, "
        "TV=%.4f, monotonic=%s, OOG=%.2f%%",
        metrics.mean_delta_e, metrics.p95_delta_e, metrics.max_delta_e,
        metrics.total_variation, metrics.neutral_monotonic, metrics.oog_percentage,
    )

    return metrics


def compute_coverage_map(
    occupied_flat_indices: np.ndarray,
    N: int,
) -> np.ndarray:
    """Compute a 3D coverage map showing distance to nearest data.

    Color interpretation:
        - distance < 2: Dense data (green)
        - distance 2-5: Sparse data (yellow)
        - distance > 5: Extrapolated (red)

    Args:
        occupied_flat_indices: Flat indices of bins with data.
        N: LUT grid size.

    Returns:
        (N, N, N) float array of distances.
    """
    distances = compute_distance_to_data(occupied_flat_indices, N)
    return distances.reshape((N, N, N))
