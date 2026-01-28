"""Pipeline solving stage: orchestrates matrix construction and solver.

Supports three prior models:
    identity                 - Standard identity prior (original behavior)
    baseline_residual        - Fit baseline T0, solve for residual deltaL
    baseline_multigrid_residual - Same but with coarse-to-fine multigrid
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from lutsmith.config import LUT_CLAMP_MIN, LUT_CLAMP_MAX
from lutsmith.core.lut import clip_lut
from lutsmith.core.solver import solve_per_channel
from lutsmith.core.types import PipelineConfig, PriorModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multigrid helpers
# ---------------------------------------------------------------------------

def _determine_multigrid_schedule(N: int, coarse_size: int) -> list[int]:
    """Determine coarse-to-fine grid sizes for multigrid solving.

    Doubles (minus one) at each step until reaching N.
    Examples: N=33, coarse=17 -> [17, 33]
              N=65, coarse=17 -> [17, 33, 65]
    """
    if coarse_size >= N:
        return [N]

    schedule = []
    size = coarse_size
    while size < N:
        schedule.append(size)
        size = min(size * 2 - 1, N)
    schedule.append(N)
    return schedule


def _solve_multigrid(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    sample_alpha: np.ndarray,
    config: PipelineConfig,
    baseline,
    occupied_flat_indices: Optional[np.ndarray],
    shadow_threshold: float | None,
    deep_threshold: float | None,
) -> tuple[np.ndarray, list[dict]]:
    """Solve using multigrid coarse-to-fine with baseline prior.

    At each level:
        1. Resample current LUT to level size
        2. Compute residual targets (output - prior_at_samples)
        3. Solve for delta with scaled smoothness
        4. Combine: current = prior + delta
        5. Upsample to next level

    Args:
        input_rgb: (M, 3) input colors.
        output_rgb: (M, 3) target colors.
        sample_alpha: (M,) weights.
        config: Pipeline configuration.
        baseline: Fitted BaselineTransform.
        occupied_flat_indices: Occupied bins (used only at final level).
        shadow_threshold: Shadow smoothness boost threshold.
        deep_threshold: Deep shadow threshold.

    Returns:
        (lut, infos): Final LUT and solver info from last level.
    """
    from lutsmith.core.baseline import evaluate_baseline_lut
    from lutsmith.core.interpolation import apply_lut_to_colors
    from lutsmith.hald.resample import resample_lut

    N = config.lut_size
    schedule = _determine_multigrid_schedule(N, config.multigrid_coarse_size)
    n_levels = len(schedule)

    logger.info("Multigrid schedule: %s (%d levels)", schedule, n_levels)

    # Start with baseline LUT at the coarsest level
    current_lut = evaluate_baseline_lut(baseline, schedule[0])
    infos: list[dict] = []

    for level_idx, level_size in enumerate(schedule):
        # Resample current LUT to this level's size if needed
        if current_lut.shape[0] != level_size:
            current_lut = resample_lut(current_lut, level_size)

        # Compute residual targets: output - LUT(input)
        prior_at_samples = apply_lut_to_colors(
            input_rgb, current_lut, level_size, config.kernel.value,
        )
        level_residual = output_rgb - prior_at_samples

        # Scale smoothness: higher at coarse levels, normal at finest
        level_smoothness = config.smoothness * (
            config.multigrid_smoothness_scale ** (n_levels - 1 - level_idx)
        )

        logger.info(
            "Multigrid level %d/%d: N=%d, smoothness=%.4f",
            level_idx + 1, n_levels, level_size, level_smoothness,
        )

        # Zero prior for residual solve
        level_prior_flat = np.zeros((level_size ** 3, 3), dtype=np.float64)

        # Use proper occupied indices only at final level (grid-size must match)
        level_occupied = occupied_flat_indices if level_size == N else None

        # Solve for delta at this level
        delta_lut, level_infos = solve_per_channel(
            input_rgb=input_rgb,
            output_rgb=level_residual,
            sample_alpha=sample_alpha,
            N=level_size,
            lambda_s=level_smoothness,
            lambda_r=config.prior_strength,
            kernel=config.kernel.value,
            loss=config.robust_loss.value,
            huber_delta=config.huber_delta,
            irls_iterations=config.irls_iterations,
            occupied_flat_indices=level_occupied,
            shadow_threshold=shadow_threshold,
            deep_threshold=deep_threshold,
            connectivity=config.laplacian_connectivity,
            prior_lut_flat=level_prior_flat,
            color_basis=config.color_basis.value,
            chroma_smoothness_ratio=config.chroma_smoothness_ratio,
            neutral_chroma_prior_k=config.neutral_chroma_prior_k,
            neutral_chroma_prior_sigma=config.neutral_chroma_prior_sigma,
        )

        # Combine: current = prior + delta
        current_lut = current_lut + delta_lut
        infos = level_infos

    return current_lut, infos


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

    When prior_model is BASELINE_RESIDUAL or BASELINE_MULTIGRID_RESIDUAL:
        1. Fit baseline transform T0(x) = h(Mx + b)
        2. Quality-gate the baseline against identity
        3. If accepted: solve for residual deltaL with zero prior
        4. Final LUT = T0_LUT + deltaL

    When prior_model is BASELINE_MULTIGRID_RESIDUAL, the residual solve
    uses a coarse-to-fine multigrid approach for better convergence.

    Args:
        input_rgb: (M, 3) input sample colors.
        output_rgb: (M, 3) output sample colors.
        sample_alpha: (M,) per-sample weights.
        config: Pipeline configuration.
        occupied_flat_indices: Flat indices of occupied bins.
        shadow_threshold: Upper luminance boundary for shadow smoothness boost.
        deep_threshold: Luminance boundary for maximum smoothness boost.

    Returns:
        (lut, infos): (N, N, N, 3) LUT and per-channel solver info.
    """
    N = config.lut_size
    logger.info(
        "Solving LUT: N=%d, kernel=%s, loss=%s, irls_iter=%d, "
        "lambda_s=%.4f, lambda_r=%.4f, prior=%s, basis=%s",
        N, config.kernel.value, config.robust_loss.value,
        config.irls_iterations, config.smoothness, config.prior_strength,
        config.prior_model.value, config.color_basis.value,
    )

    # Baseline prior handling
    prior_lut_flat = None
    solve_output = output_rgb
    t0_lut = None
    multigrid_done = False

    use_baseline = config.prior_model in (
        PriorModel.BASELINE_RESIDUAL,
        PriorModel.BASELINE_MULTIGRID_RESIDUAL,
    )

    if use_baseline:
        from lutsmith.core.baseline import (
            fit_baseline,
            evaluate_baseline_lut,
            baseline_quality_gate,
        )

        logger.info("Fitting baseline transform (prior_model=%s)...",
                     config.prior_model.value)
        baseline = fit_baseline(
            input_rgb, output_rgb, sample_alpha,
            huber_delta=config.huber_delta,
        )

        if baseline_quality_gate(baseline, input_rgb, output_rgb, sample_alpha):
            if config.prior_model == PriorModel.BASELINE_MULTIGRID_RESIDUAL:
                # Multigrid coarse-to-fine solve
                lut, infos = _solve_multigrid(
                    input_rgb, output_rgb, sample_alpha, config,
                    baseline, occupied_flat_indices,
                    shadow_threshold, deep_threshold,
                )
                multigrid_done = True
            else:
                # Single-level residual solve (BASELINE_RESIDUAL)
                t0_at_samples = baseline.evaluate(input_rgb)
                solve_output = output_rgb - t0_at_samples

                # Prior for residual: zeros (pull toward zero in unseen regions,
                # so final LUT = T0 + 0 = T0 where there's no data)
                prior_lut_flat = np.zeros((N ** 3, 3), dtype=np.float64)

                # T0 LUT for recombination after solving
                t0_lut = evaluate_baseline_lut(baseline, N)

                logger.info("Solving for residual delta on top of baseline...")
        else:
            logger.info("Baseline failed quality gate, falling back to identity prior")

    if not multigrid_done:
        lut, infos = solve_per_channel(
            input_rgb=input_rgb,
            output_rgb=solve_output,
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
            connectivity=config.laplacian_connectivity,
            prior_lut_flat=prior_lut_flat,
            color_basis=config.color_basis.value,
            chroma_smoothness_ratio=config.chroma_smoothness_ratio,
            neutral_chroma_prior_k=config.neutral_chroma_prior_k,
            neutral_chroma_prior_sigma=config.neutral_chroma_prior_sigma,
        )

        # Recombine: final LUT = T0 + deltaL
        if t0_lut is not None:
            lut = t0_lut + lut

    # Post-solve safety clamp
    lut = clip_lut(lut, lo=LUT_CLAMP_MIN, hi=LUT_CLAMP_MAX)

    logger.info(
        "LUT solved: range [%.4f, %.4f]",
        float(np.min(lut)), float(np.max(lut)),
    )

    return lut, infos
