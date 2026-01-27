"""Pixel sampling, binning, and aggregation for LUT extraction.

Bins pixel pairs from source/target images into a 3D color-space grid,
computes statistics per bin, and produces weighted samples for regression.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from lutsmith.config import (
    CLIP_PENALTY_HIGH,
    CLIP_PENALTY_LOW,
    DEFAULT_MIN_SAMPLES_PER_BIN,
    EPSILON,
)
from lutsmith.core.types import (
    BinStatistics,
    PipelineConfig,
    SamplePoint,
    WeightingMode,
)
from lutsmith._numba_kernels.binning import bin_pixels

logger = logging.getLogger(__name__)


def bin_and_aggregate(
    source: np.ndarray,
    target: np.ndarray,
    config: PipelineConfig,
) -> tuple[list[BinStatistics], np.ndarray]:
    """Bin image pixel pairs and compute per-bin statistics.

    Args:
        source: (H, W, 3) source image.
        target: (H, W, 3) target image.
        config: Pipeline configuration.

    Returns:
        (bins, occupied_flat_indices): List of BinStatistics for occupied bins,
        and array of their flat indices.
    """
    H, W = source.shape[:2]
    bin_res = config.bin_resolution

    # Flatten to (M, 3)
    source_flat = source.reshape(-1, 3)
    target_flat = target.reshape(-1, 3)

    # Generate normalized pixel coordinates for spatial analysis
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    pixel_x = (x_coords.ravel() / max(W - 1, 1)).astype(np.float32)
    pixel_y = (y_coords.ravel() / max(H - 1, 1)).astype(np.float32)

    # Run binning
    logger.info(
        "Binning %d pixels into %d^3 grid...",
        len(source_flat), bin_res,
    )
    count, sum_input, mean_output, m2_output, sum_x, sum_y = bin_pixels(
        source_flat, target_flat, bin_res, pixel_x, pixel_y
    )

    # Extract occupied bins
    min_samples = max(config.min_samples_per_bin, 1)
    occupied_mask = count >= min_samples
    occupied_indices = np.where(occupied_mask)[0]

    logger.info(
        "Occupied bins: %d / %d (%.1f%%), min_samples=%d",
        len(occupied_indices),
        bin_res ** 3,
        100.0 * len(occupied_indices) / bin_res ** 3,
        min_samples,
    )

    bins = []
    for idx in occupied_indices:
        n = count[idx]
        # Variance from Welford's M2: var = M2 / (n - 1) for sample variance
        variance = m2_output[idx] / max(n - 1, 1)
        variance = np.maximum(variance, 0.0)  # Clamp numerical noise

        mean_in = (sum_input[idx] / n).astype(np.float32)
        mean_out = mean_output[idx].astype(np.float32)
        spatial_centroid = np.array([
            sum_x[idx] / n,
            sum_y[idx] / n,
        ], dtype=np.float32)

        bins.append(BinStatistics(
            mean_input=mean_in,
            mean_output=mean_out,
            count=int(n),
            variance=variance.astype(np.float32),
            spatial_centroid=spatial_centroid,
            bin_index=int(idx),
        ))

    return bins, occupied_indices


def compute_clip_penalty(rgb: np.ndarray) -> float:
    """Compute penalty for near-black/near-white inputs.

    Clipped pixels carry no useful information about the color transform.

    Args:
        rgb: (3,) input color.

    Returns:
        Penalty multiplier in [0, 1]. 1.0 = no penalty.
    """
    # Check if any channel is near black or white
    min_val = float(np.min(rgb))
    max_val = float(np.max(rgb))

    penalty = 1.0
    if min_val < CLIP_PENALTY_LOW:
        # Linear ramp from 0 at 0 to 1 at threshold
        penalty *= min(min_val / CLIP_PENALTY_LOW, 1.0)
    if max_val > CLIP_PENALTY_HIGH:
        penalty *= min((1.0 - max_val) / (1.0 - CLIP_PENALTY_HIGH), 1.0)

    return max(penalty, 0.0)


def compute_sample_weights(
    bins: list[BinStatistics],
    config: PipelineConfig | None = None,
    shadow_lum_thresh: float | None = None,
) -> np.ndarray:
    """Compute per-sample weights for the regression.

    Args:
        bins: List of bin statistics.
        config: Pipeline config.
        shadow_lum_thresh: Luminance threshold for shadow weighting.

    Returns:
        (M,) array of weights, one per bin/sample.
    """
    if config is None:
        config = PipelineConfig()

    M = len(bins)
    weights = np.ones(M, dtype=np.float64)

    # Variance sensitivity: aggressive in shadows where noisy bins
    # (e.g. from diffusion artifacts, JPEG compression) cause per-channel
    # divergence in the solved LUT.
    k_bright = 0.5
    k_shadow = 10.0
    if shadow_lum_thresh is None:
        shadow_lum_thresh = 0.25

    for i, b in enumerate(bins):
        # Luminance-aware variance sensitivity
        lum = float(
            0.2126 * b.mean_input[0]
            + 0.7152 * b.mean_input[1]
            + 0.0722 * b.mean_input[2]
        )
        t = min(max(lum / shadow_lum_thresh, 0.0), 1.0)
        k = k_bright * t + k_shadow * (1.0 - t)

        # Confidence from variance and count
        var_mag = float(np.mean(b.variance))
        count_factor = min(1.0, b.count / max(config.min_samples_per_bin, 1))
        confidence = (1.0 / (1.0 + k * var_mag)) * count_factor

        # Clip penalty
        clip_pen = compute_clip_penalty(b.mean_input)

        if config.weighting == WeightingMode.COVERAGE_FAIR:
            weights[i] = confidence * clip_pen
        else:  # FREQUENCY
            weights[i] = confidence * clip_pen * b.count

    # Normalize weights to have mean 1.0 for numerical stability
    mean_w = np.mean(weights)
    if mean_w > EPSILON:
        weights /= mean_w

    return weights


def detect_spatial_inconsistency(
    bins: list[BinStatistics],
    threshold: float = 0.3,
) -> list[int]:
    """Detect bins with suspected spatial inconsistency (vignettes).

    High variance that correlates with spatial position suggests
    the transform is position-dependent, not just color-dependent.

    Args:
        bins: List of bin statistics.
        threshold: Contradiction score threshold.

    Returns:
        List of indices of suspicious bins.
    """
    suspicious = []
    for i, b in enumerate(bins):
        if b.contradiction_score > threshold:
            suspicious.append(i)
    return suspicious


def bins_to_samples(
    bins: list[BinStatistics],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert bin statistics to sample arrays for the solver.

    Args:
        bins: List of bin statistics.
        weights: (M,) per-bin weights.

    Returns:
        (input_rgb, output_rgb, alpha): All (M, 3), (M, 3), (M,).
    """
    M = len(bins)
    input_rgb = np.empty((M, 3), dtype=np.float32)
    output_rgb = np.empty((M, 3), dtype=np.float32)

    for i, b in enumerate(bins):
        input_rgb[i] = b.mean_input
        output_rgb[i] = b.mean_output

    return input_rgb, output_rgb, weights


def occupied_lut_indices(
    input_rgb: np.ndarray,
    lut_size: int,
) -> np.ndarray:
    """Compute occupied LUT node indices from input samples.

    Uses the same flat indexing convention (R fastest).
    """
    if len(input_rgb) == 0:
        return np.array([], dtype=np.int64)

    scaled = np.clip(input_rgb, 0.0, 1.0) * (lut_size - 1)
    idx = np.minimum(scaled.astype(np.int64), lut_size - 1)
    r = idx[:, 0]
    g = idx[:, 1]
    b = idx[:, 2]
    flat = b * lut_size * lut_size + g * lut_size + r
    return np.unique(flat)
