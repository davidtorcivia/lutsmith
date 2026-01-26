"""Pipeline solving stage: orchestrates matrix construction and solver."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from chromaforge.config import LUT_CLAMP_MIN, LUT_CLAMP_MAX
from chromaforge.core.lut import clip_lut
from chromaforge.core.solver import solve_per_channel
from chromaforge.core.types import PipelineConfig

logger = logging.getLogger(__name__)


def solve_lut(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    sample_alpha: np.ndarray,
    config: PipelineConfig,
    occupied_flat_indices: Optional[np.ndarray] = None,
    shadow_threshold: float | None = None,
    deep_threshold: float | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Run the full LUT regression solve.

    Args:
        input_rgb: (M, 3) input sample colors.
        output_rgb: (M, 3) output sample colors.
        sample_alpha: (M,) per-sample weights.
        config: Pipeline configuration.
        occupied_flat_indices: Flat indices of occupied bins.

    Returns:
        (lut, infos): (N, N, N, 3) LUT and per-channel solver info.
    """
    N = config.lut_size
    logger.info(
        "Solving LUT: N=%d, kernel=%s, loss=%s, irls_iter=%d, "
        "lambda_s=%.4f, lambda_r=%.4f",
        N, config.kernel.value, config.robust_loss.value,
        config.irls_iterations, config.smoothness, config.prior_strength,
    )

    lut, infos = solve_per_channel(
        input_rgb=input_rgb,
        output_rgb=output_rgb,
        sample_alpha=sample_alpha,
        N=N,
        lambda_s=config.smoothness,
        lambda_r=config.prior_strength,
        kernel=config.kernel.value,
        loss=config.robust_loss.value,
        huber_delta=config.huber_delta,
        irls_iterations=config.irls_iterations,
        occupied_flat_indices=occupied_flat_indices,
        shadow_threshold=shadow_threshold,
        deep_threshold=deep_threshold,
    )

    # Post-solve safety clamp
    lut = clip_lut(lut, lo=LUT_CLAMP_MIN, hi=LUT_CLAMP_MAX)

    logger.info(
        "LUT solved: range [%.4f, %.4f]",
        float(np.min(lut)), float(np.max(lut)),
    )

    return lut, infos
