"""Optional iterative refinement loop for LUT quality improvement.

After an initial fit, applies the LUT to source samples, measures residuals,
identifies systematic errors, and refits with adjusted weights.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import numpy as np

from lutsmith.config import EPSILON
from lutsmith.core.interpolation import apply_lut_to_colors
from lutsmith.core.types import PipelineConfig
from lutsmith.pipeline.solving import solve_lut

logger = logging.getLogger(__name__)


def refine_lut(
    lut: np.ndarray,
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    sample_alpha: np.ndarray,
    config: PipelineConfig,
    occupied_flat_indices: Optional[np.ndarray] = None,
    max_iterations: int = 2,
    shadow_threshold: float | None = None,
    deep_threshold: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Iteratively refine the LUT by detecting and downweighting outliers.

    Steps per iteration:
        1. Apply current LUT to source samples
        2. Compute per-sample residuals vs target
        3. Identify high-residual samples
        4. Downweight them
        5. Refit

    Args:
        lut: (N, N, N, 3) initial LUT.
        input_rgb: (M, 3) input colors.
        output_rgb: (M, 3) target colors.
        sample_alpha: (M,) initial weights.
        config: Pipeline config.
        occupied_flat_indices: Occupied bin indices.
        max_iterations: Number of refinement passes.

    Returns:
        (refined_lut, diagnostics): Refined LUT and iteration info.
    """
    N = config.lut_size
    current_lut = lut.copy()
    current_alpha = sample_alpha.copy()
    diagnostics = {"iterations": [], "initial_error": None}

    for iteration in range(max_iterations):
        # Apply current LUT
        predicted = apply_lut_to_colors(
            input_rgb, current_lut, N, config.kernel.value
        )

        # Compute per-sample RGB residuals
        residuals = predicted - output_rgb
        residual_norms = np.linalg.norm(residuals, axis=1)

        # Statistics
        mean_err = float(np.mean(residual_norms))
        p95_err = float(np.percentile(residual_norms, 95))
        max_err = float(np.max(residual_norms))

        if iteration == 0:
            diagnostics["initial_error"] = mean_err

        iter_info = {
            "iteration": iteration,
            "mean_residual": mean_err,
            "p95_residual": p95_err,
            "max_residual": max_err,
        }

        logger.info(
            "Refinement iteration %d: mean=%.4f, p95=%.4f, max=%.4f",
            iteration, mean_err, p95_err, max_err,
        )

        # Identify high-residual samples (> 2x P95)
        outlier_threshold = 2.0 * p95_err
        outlier_mask = residual_norms > outlier_threshold
        n_outliers = int(np.sum(outlier_mask))

        if n_outliers == 0:
            logger.info("No outliers found, stopping refinement")
            iter_info["outliers_found"] = 0
            diagnostics["iterations"].append(iter_info)
            break

        iter_info["outliers_found"] = n_outliers
        logger.info("Downweighting %d outlier samples", n_outliers)

        # Downweight outliers: reduce weight by factor proportional to excess residual
        downweight = np.where(
            outlier_mask,
            outlier_threshold / (residual_norms + EPSILON),
            1.0,
        )
        current_alpha = current_alpha * downweight

        # Refit with slightly reduced smoothness (more data trust)
        reduced_config = replace(config, smoothness=config.smoothness * 0.8)

        current_lut, _ = solve_lut(
            input_rgb,
            output_rgb,
            current_alpha,
            reduced_config,
            occupied_flat_indices,
            shadow_threshold=shadow_threshold,
            deep_threshold=deep_threshold,
        )

        diagnostics["iterations"].append(iter_info)

    diagnostics["final_error"] = float(np.mean(
        np.linalg.norm(
            apply_lut_to_colors(input_rgb, current_lut, N, config.kernel.value)
            - output_rgb,
            axis=1,
        )
    ))

    return current_lut, diagnostics
