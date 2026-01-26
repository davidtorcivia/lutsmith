"""Pipeline runner: orchestrates all stages from input to output.

This is the single entry point for both CLI and GUI.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from chromaforge.color.shaper import generate_1d_shaper
from chromaforge.core.lut import create_lut_data, estimate_shadow_thresholds
from chromaforge.core.types import (
    CancelCheck,
    PipelineConfig,
    PipelineResult,
    ProgressCallback,
)
from chromaforge.errors import PipelineCancelledError, PipelineError
from chromaforge.io.cube import write_cube
from chromaforge.pipeline.preprocess import preprocess_pair
from chromaforge.pipeline.refinement import refine_lut
from chromaforge.pipeline.sampling import (
    bin_and_aggregate,
    bins_to_samples,
    compute_sample_weights,
    detect_spatial_inconsistency,
    occupied_lut_indices,
)
from chromaforge.pipeline.solving import solve_lut
from chromaforge.pipeline.validation import compute_coverage_map, validate_lut

logger = logging.getLogger(__name__)


def _check_cancel(cancel_check: Optional[CancelCheck]) -> None:
    """Raise if cancellation requested."""
    if cancel_check is not None and cancel_check():
        raise PipelineCancelledError("Pipeline cancelled by user")


def _emit_progress(
    callback: Optional[ProgressCallback],
    stage: str,
    fraction: float,
    message: str = "",
) -> None:
    """Emit progress update if callback is provided."""
    if callback is not None:
        callback(stage, fraction, message)


def run_pipeline(
    config: PipelineConfig,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> PipelineResult:
    """Run the complete LUT extraction pipeline.

    Stages:
        1. Preprocess: load images, sanitize, detect transfer function
        2. Sample: bin pixels, compute weights
        3. Solve: build sparse system, run IRLS solver
        4. Refine (optional): iterative refit
        5. Validate: compute quality metrics
        6. Export: write LUT file

    Args:
        config: Full pipeline configuration.
        progress_callback: (stage_name, fraction, message) callback.
        cancel_check: Returns True if pipeline should be cancelled.

    Returns:
        PipelineResult with LUT, metrics, and diagnostics.
    """
    t_start = time.perf_counter()
    diagnostics = {}

    if config.source_path is None or config.target_path is None:
        raise PipelineError("Source and target paths are required")

    # ---------------------------------------------------------------
    # Stage 1: Preprocess
    # ---------------------------------------------------------------
    _emit_progress(progress_callback, "preprocess", 0.0, "Loading images...")
    _check_cancel(cancel_check)

    t0 = time.perf_counter()
    source, target, meta, shaper = preprocess_pair(
        config.source_path, config.target_path, config
    )
    diagnostics["preprocess_time"] = time.perf_counter() - t0
    diagnostics["image_size"] = f"{meta['width']}x{meta['height']}"
    diagnostics["total_pixels"] = meta["total_pixels"]
    diagnostics["transfer_function"] = meta["transfer_function"].value
    diagnostics["shaper_applied"] = meta.get("shaper_applied", False)
    diagnostics["shaper_mode"] = meta.get("shaper_mode", "disabled")

    _emit_progress(progress_callback, "preprocess", 1.0, "Images loaded")
    logger.info("Preprocess: %.2fs", diagnostics["preprocess_time"])

    # ---------------------------------------------------------------
    # Stage 2: Sample
    # ---------------------------------------------------------------
    _emit_progress(progress_callback, "sampling", 0.0, "Binning pixels...")
    _check_cancel(cancel_check)

    t0 = time.perf_counter()
    bins, occupied_indices = bin_and_aggregate(source, target, config)

    if len(bins) < 10:
        raise PipelineError(
            f"Only {len(bins)} occupied bins found. Need at least 10 "
            f"for meaningful LUT extraction. Check that images are different "
            f"and contain varied colors."
        )

    # Detect spatial issues
    suspicious = detect_spatial_inconsistency(bins)
    if suspicious:
        logger.warning(
            "%d bins flagged for spatial inconsistency (possible vignette)",
            len(suspicious),
        )
        diagnostics["spatial_warnings"] = len(suspicious)

    # Adaptive shadow thresholds (helps with low-contrast / log-like inputs)
    shadow_threshold = config.shadow_threshold
    deep_threshold = config.deep_shadow_threshold
    shadow_info = {"auto_applied": False, "reason": "manual_or_default"}

    if shadow_threshold is None or deep_threshold is None:
        if config.shadow_auto:
            input_means = np.stack([b.mean_input for b in bins], axis=0)
            est_shadow, est_deep, shadow_info = estimate_shadow_thresholds(input_means)
            if shadow_threshold is None:
                shadow_threshold = est_shadow
            if deep_threshold is None:
                deep_threshold = est_deep
        else:
            shadow_threshold = 0.25 if shadow_threshold is None else shadow_threshold
            deep_threshold = 0.08 if deep_threshold is None else deep_threshold

    diagnostics["shadow_threshold"] = shadow_threshold
    diagnostics["deep_shadow_threshold"] = deep_threshold
    diagnostics["shadow_auto"] = shadow_info.get("auto_applied", False)
    if shadow_info.get("auto_applied"):
        logger.info(
            "Adaptive shadow thresholds: deep=%.3f, shadow=%.3f (p99=%.3f, mean=%.3f)",
            deep_threshold,
            shadow_threshold,
            shadow_info.get("p99", 0.0),
            shadow_info.get("mean", 0.0),
        )

    # Compute weights
    weights = compute_sample_weights(bins, config, shadow_lum_thresh=shadow_threshold)
    input_rgb, output_rgb, alpha = bins_to_samples(bins, weights)
    lut_occupied = occupied_lut_indices(input_rgb, config.lut_size)

    diagnostics["sampling_time"] = time.perf_counter() - t0
    diagnostics["occupied_bins"] = len(bins)
    diagnostics["total_bins"] = config.bin_resolution ** 3

    _emit_progress(progress_callback, "sampling", 1.0,
                   f"{len(bins)} bins occupied")
    logger.info("Sampling: %.2fs, %d bins", diagnostics["sampling_time"], len(bins))

    # ---------------------------------------------------------------
    # Stage 3: Solve
    # ---------------------------------------------------------------
    _emit_progress(progress_callback, "solving", 0.0, "Building matrix...")
    _check_cancel(cancel_check)

    t0 = time.perf_counter()
    lut_array, solver_infos = solve_lut(
        input_rgb,
        output_rgb,
        alpha,
        config,
        lut_occupied,
        shadow_threshold=shadow_threshold,
        deep_threshold=deep_threshold,
    )
    diagnostics["solve_time"] = time.perf_counter() - t0
    diagnostics["solver_info"] = [
        {k: v for k, v in info.items() if not isinstance(v, np.ndarray)}
        for info in solver_infos if info is not None
    ]

    _emit_progress(progress_callback, "solving", 1.0, "LUT solved")
    logger.info("Solve: %.2fs", diagnostics["solve_time"])

    # ---------------------------------------------------------------
    # Stage 4: Refine (optional)
    # ---------------------------------------------------------------
    if config.enable_refinement:
        _emit_progress(progress_callback, "refinement", 0.0, "Refining...")
        _check_cancel(cancel_check)

        t0 = time.perf_counter()
        lut_array, refine_diag = refine_lut(
            lut_array,
            input_rgb,
            output_rgb,
            alpha,
            config,
            lut_occupied,
            max_iterations=config.refinement_iterations,
            shadow_threshold=shadow_threshold,
            deep_threshold=deep_threshold,
        )
        diagnostics["refinement_time"] = time.perf_counter() - t0
        diagnostics["refinement"] = refine_diag

        _emit_progress(progress_callback, "refinement", 1.0, "Refinement complete")
        logger.info("Refinement: %.2fs", diagnostics["refinement_time"])

    # ---------------------------------------------------------------
    # Stage 5: Validate
    # ---------------------------------------------------------------
    _emit_progress(progress_callback, "validation", 0.0, "Computing metrics...")
    _check_cancel(cancel_check)

    t0 = time.perf_counter()
    metrics = validate_lut(
        input_rgb, output_rgb, lut_array,
        config.lut_size, config.kernel.value,
        lut_occupied,
    )

    coverage_map = None
    if config.generate_coverage_report:
        coverage_map = compute_coverage_map(lut_occupied, config.lut_size)

    diagnostics["validation_time"] = time.perf_counter() - t0
    _emit_progress(progress_callback, "validation", 1.0,
                   f"mean dE={metrics.mean_delta_e:.2f}")
    logger.info("Validation: %.2fs", diagnostics["validation_time"])

    # ---------------------------------------------------------------
    # Stage 6: Export
    # ---------------------------------------------------------------
    output_path = None
    if config.output_path is not None:
        _emit_progress(progress_callback, "export", 0.0, "Exporting...")
        _check_cancel(cancel_check)

        t0 = time.perf_counter()
        output_path = Path(config.output_path)

        shaper_lut = None
        if shaper.get("applied") and config.include_shaper is not False:
            if config.format.value == "cube":
                shaper_lut = generate_1d_shaper(shaper["forward"])
            else:
                logger.warning(
                    "Shaper requested but not exported for format '%s'",
                    config.format.value,
                )
                diagnostics["shaper_warning"] = (
                    f"Shaper not exported for format '{config.format.value}'"
                )

        if config.format.value == "cube":
            write_cube(
                output_path,
                lut_array,
                title=config.title,
                kernel_hint=config.kernel.value,
                shaper=shaper_lut,
            )
        elif config.format.value in ("aml", "alf4"):
            from chromaforge.io.arri import export_arri
            export_arri(lut_array, output_path, format=config.format.value)

        diagnostics["export_time"] = time.perf_counter() - t0
        _emit_progress(progress_callback, "export", 1.0, f"Saved: {output_path}")
        logger.info("Export: %.2fs -> %s", diagnostics["export_time"], output_path)

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    total_time = time.perf_counter() - t_start
    diagnostics["total_time"] = total_time
    logger.info("Pipeline complete: %.2fs total", total_time)

    lut_data = create_lut_data(lut_array, config.kernel, config.title)

    return PipelineResult(
        lut=lut_data,
        metrics=metrics,
        coverage_map=coverage_map,
        diagnostics=diagnostics,
        output_path=output_path,
    )
