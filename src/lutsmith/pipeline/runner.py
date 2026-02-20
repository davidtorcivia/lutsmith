"""Pipeline runner: orchestrates all stages from input to output.

This is the single entry point for both CLI and GUI.
"""

from __future__ import annotations

from dataclasses import replace
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from lutsmith.color.shaper import generate_1d_shaper
from lutsmith.core.lut import create_lut_data, estimate_shadow_thresholds
from lutsmith.core.types import (
    CancelCheck,
    PipelineConfig,
    PipelineResult,
    ProgressCallback,
    TransferFunction,
)
from lutsmith.errors import PipelineCancelledError, PipelineError
from lutsmith.io.cube import write_cube
from lutsmith.pipeline.preprocess import preprocess_pair
from lutsmith.pipeline.refinement import refine_lut
from lutsmith.pipeline.normalization import apply_pair_normalization
from lutsmith.pipeline.sampling import (
    bin_and_aggregate,
    bins_to_samples,
    compute_sample_weights,
    detect_spatial_inconsistency,
    occupied_lut_indices,
)
from lutsmith.pipeline.solving import solve_lut
from lutsmith.pipeline.validation import compute_coverage_map, validate_lut

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
            from lutsmith.io.arri import export_arri
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


def run_multi_pipeline(
    pair_paths: Sequence[tuple[Path | str, Path | str]],
    config: PipelineConfig,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
    pair_weights: Optional[Sequence[float]] = None,
    pair_transfer_fns: Optional[Sequence[Optional[str]]] = None,
    pair_normalization_modes: Optional[Sequence[Optional[str]]] = None,
    pair_balance: str = "equal",
    outlier_sigma: float = 0.0,
    min_pairs_after_outlier: int = 3,
    allow_mixed_transfer: bool = False,
) -> PipelineResult:
    """Run LUT extraction from multiple matched source/target image pairs."""
    if len(pair_paths) == 0:
        raise PipelineError("At least one source/target pair is required")
    if pair_balance not in {"equal", "by_bins", "by_pixels"}:
        raise PipelineError(
            f"Invalid pair_balance='{pair_balance}'. Use one of: equal, by_bins, by_pixels"
        )
    if outlier_sigma < 0:
        raise PipelineError("outlier_sigma must be >= 0")
    if min_pairs_after_outlier < 1:
        raise PipelineError("min_pairs_after_outlier must be >= 1")

    t_start = time.perf_counter()
    diagnostics = {"num_pairs": len(pair_paths), "pairs": []}

    # ---------------------------------------------------------------
    # Stage 1: Preprocess + Sample per pair
    # ---------------------------------------------------------------
    _emit_progress(progress_callback, "preprocess", 0.0, "Loading image pairs...")
    _check_cancel(cancel_check)

    t0 = time.perf_counter()
    tf_values = []
    pair_shaper = None
    total_pixels = 0
    spatial_warnings = 0
    pair_records = []

    if pair_weights is None:
        pair_weights = [1.0] * len(pair_paths)
    if len(pair_weights) != len(pair_paths):
        raise PipelineError(
            f"pair_weights length ({len(pair_weights)}) must match pair_paths length ({len(pair_paths)})"
        )
    if pair_transfer_fns is None:
        pair_transfer_fns = [None] * len(pair_paths)
    if len(pair_transfer_fns) != len(pair_paths):
        raise PipelineError(
            f"pair_transfer_fns length ({len(pair_transfer_fns)}) must match pair_paths length ({len(pair_paths)})"
        )
    if pair_normalization_modes is None:
        pair_normalization_modes = [None] * len(pair_paths)
    if len(pair_normalization_modes) != len(pair_paths):
        raise PipelineError(
            f"pair_normalization_modes length ({len(pair_normalization_modes)}) must match pair_paths length ({len(pair_paths)})"
        )

    from lutsmith.pipeline.sampling import (
        bin_and_aggregate,
        detect_spatial_inconsistency,
    )

    for i, ((source_path, target_path), user_weight, tf_override, norm_mode) in enumerate(
        zip(pair_paths, pair_weights, pair_transfer_fns, pair_normalization_modes)
    ):
        _check_cancel(cancel_check)
        pair_idx = i + 1
        weight = float(user_weight)
        if weight < 0:
            raise PipelineError(f"Pair {pair_idx} has negative weight: {weight}")

        pair_config = config
        if tf_override is not None and str(tf_override).strip():
            tf_text = str(tf_override).strip().lower()
            try:
                pair_tf = TransferFunction(tf_text)
            except ValueError as exc:
                raise PipelineError(
                    f"Pair {pair_idx} has invalid transfer_fn override '{tf_override}'. "
                    "Valid values: auto, linear, log_c3, log_c4, slog3, vlog, unknown."
                ) from exc
            pair_config = replace(config, transfer_function=pair_tf)

        source, target, meta, shaper = preprocess_pair(source_path, target_path, pair_config)
        source, target, norm_diag = apply_pair_normalization(source, target, norm_mode)
        bins, _ = bin_and_aggregate(source, target, config)

        if len(bins) < 10:
            raise PipelineError(
                f"Pair {pair_idx} has only {len(bins)} occupied bins. "
                f"Need at least 10 per pair for robust multi-pair fitting. "
                f"Pair: {source_path} -> {target_path}"
            )

        suspicious = detect_spatial_inconsistency(bins)
        if suspicious:
            spatial_warnings += len(suspicious)
            logger.warning(
                "Pair %d: %d bins flagged for spatial inconsistency",
                pair_idx, len(suspicious),
            )

        tf_value = meta["transfer_function"].value
        tf_values.append(tf_value)
        total_pixels += int(meta["total_pixels"])

        if pair_shaper is None:
            pair_shaper = shaper
        elif shaper.get("applied") != pair_shaper.get("applied"):
            raise PipelineError(
                "Inconsistent shaper application across pairs. "
                "Use inputs with consistent transfer function/shaper settings."
            )

        pair_records.append({
            "index": pair_idx,
            "source": str(source_path),
            "target": str(target_path),
            "bins": bins,
            "weight": weight,
            "occupied_bins": len(bins),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "pixels": int(meta["total_pixels"]),
            "transfer_function": tf_value,
            "shaper_applied": bool(meta.get("shaper_applied", False)),
            "transfer_fn_override": str(tf_override).strip().lower() if tf_override else "",
            "normalization_mode": norm_diag.mode,
            "normalization_gains": norm_diag.gains,
            "normalization_biases": norm_diag.biases,
        })

        progress = pair_idx / len(pair_paths)
        _emit_progress(
            progress_callback,
            "preprocess",
            progress,
            f"Loaded pair {pair_idx}/{len(pair_paths)}",
        )

    preprocess_time = time.perf_counter() - t0
    unique_tfs = sorted(set(tf_values))
    if len(unique_tfs) > 1 and not allow_mixed_transfer:
        raise PipelineError(
            "Mixed transfer-function detections across pairs: "
            f"{unique_tfs}. Re-run with --allow-mixed-transfer if intentional."
        )

    diagnostics["preprocess_time"] = preprocess_time
    diagnostics["total_pixels"] = total_pixels
    diagnostics["total_bins"] = config.bin_resolution ** 3
    diagnostics["spatial_warnings"] = spatial_warnings
    diagnostics["transfer_function"] = unique_tfs[0] if len(unique_tfs) == 1 else "mixed"
    diagnostics["transfer_functions"] = unique_tfs
    diagnostics["shaper_applied"] = bool(pair_shaper and pair_shaper.get("applied"))
    diagnostics["shaper_mode"] = pair_shaper.get("mode", "disabled") if pair_shaper else "disabled"
    diagnostics["pair_balance"] = pair_balance
    diagnostics["normalization_modes"] = sorted(set(rec["normalization_mode"] for rec in pair_records))

    logger.info(
        "Preprocess multi-pair: %.2fs, pairs=%d",
        preprocess_time,
        len(pair_paths),
    )

    def _fit_once(records: list[dict], label: str) -> dict:
        """Fit one LUT from the provided pair records."""
        from lutsmith.core.interpolation import apply_lut_to_colors

        if len(records) == 0:
            raise PipelineError("No pair records available for fitting")

        # -----------------------------------------------------------
        # Sampling
        # -----------------------------------------------------------
        _emit_progress(progress_callback, "sampling", 0.0, f"Aggregating bins ({label})...")
        _check_cancel(cancel_check)
        t_sampling = time.perf_counter()

        all_bins = []
        pair_slices = []
        for rec in records:
            start = len(all_bins)
            all_bins.extend(rec["bins"])
            end = len(all_bins)
            pair_slices.append((start, end))

        shadow_threshold = config.shadow_threshold
        deep_threshold = config.deep_shadow_threshold
        shadow_info = {"auto_applied": False, "reason": "manual_or_default"}
        if shadow_threshold is None or deep_threshold is None:
            if config.shadow_auto:
                input_means = np.stack([b.mean_input for b in all_bins], axis=0)
                est_shadow, est_deep, shadow_info = estimate_shadow_thresholds(input_means)
                if shadow_threshold is None:
                    shadow_threshold = est_shadow
                if deep_threshold is None:
                    deep_threshold = est_deep
            else:
                shadow_threshold = 0.25 if shadow_threshold is None else shadow_threshold
                deep_threshold = 0.08 if deep_threshold is None else deep_threshold

        base_weights = compute_sample_weights(
            all_bins,
            config,
            shadow_lum_thresh=shadow_threshold,
        )

        scaled_weights = base_weights.copy().astype(np.float64)
        weight_scales = []
        weight_shares = []
        eps = 1e-12

        for rec, (start, end) in zip(records, pair_slices):
            pair_sum = float(np.sum(base_weights[start:end]))
            if pair_balance == "equal":
                target_total = rec["weight"]
            elif pair_balance == "by_bins":
                target_total = rec["weight"] * (end - start)
            else:  # by_pixels
                target_total = rec["weight"] * rec["pixels"]

            if target_total <= 0:
                scale = 0.0
            else:
                scale = target_total / max(pair_sum, eps)
            scaled_weights[start:end] *= scale
            weight_scales.append(scale)

        total_weight = float(np.sum(scaled_weights))
        if total_weight <= eps:
            raise PipelineError("All pair weights are zero after balancing")
        scaled_weights /= np.mean(scaled_weights)

        total_weight = float(np.sum(scaled_weights))
        for start, end in pair_slices:
            weight_shares.append(float(np.sum(scaled_weights[start:end]) / max(total_weight, eps)))

        input_rgb, output_rgb, alpha = bins_to_samples(all_bins, scaled_weights)
        lut_occupied = occupied_lut_indices(input_rgb, config.lut_size)
        sampling_time = time.perf_counter() - t_sampling

        _emit_progress(
            progress_callback,
            "sampling",
            1.0,
            f"{len(all_bins)} bins ({label})",
        )

        # -----------------------------------------------------------
        # Solve
        # -----------------------------------------------------------
        _emit_progress(progress_callback, "solving", 0.0, f"Building matrix ({label})...")
        _check_cancel(cancel_check)
        t_solve = time.perf_counter()

        lut_array, solver_infos = solve_lut(
            input_rgb,
            output_rgb,
            alpha,
            config,
            lut_occupied,
            shadow_threshold=shadow_threshold,
            deep_threshold=deep_threshold,
        )
        solve_time = time.perf_counter() - t_solve
        _emit_progress(progress_callback, "solving", 1.0, f"LUT solved ({label})")

        refinement_time = 0.0
        refinement_diag = None
        if config.enable_refinement:
            _emit_progress(progress_callback, "refinement", 0.0, f"Refining ({label})...")
            _check_cancel(cancel_check)
            t_refine = time.perf_counter()
            lut_array, refinement_diag = refine_lut(
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
            refinement_time = time.perf_counter() - t_refine
            _emit_progress(progress_callback, "refinement", 1.0, f"Refinement complete ({label})")

        # -----------------------------------------------------------
        # Validate
        # -----------------------------------------------------------
        _emit_progress(progress_callback, "validation", 0.0, f"Computing metrics ({label})...")
        _check_cancel(cancel_check)
        t_validate = time.perf_counter()

        metrics = validate_lut(
            input_rgb, output_rgb, lut_array,
            config.lut_size, config.kernel.value,
            lut_occupied,
        )
        validation_time = time.perf_counter() - t_validate
        _emit_progress(progress_callback, "validation", 1.0, f"mean dE={metrics.mean_delta_e:.2f}")

        coverage_map = None
        if config.generate_coverage_report:
            coverage_map = compute_coverage_map(lut_occupied, config.lut_size)

        # Per-pair diagnostics for weighting and outlier detection.
        predicted = apply_lut_to_colors(input_rgb, lut_array, config.lut_size, config.kernel.value)
        sq_err = np.sum((predicted - output_rgb) ** 2, axis=1)
        pair_stats = []
        for rec, (start, end), scale, share in zip(records, pair_slices, weight_scales, weight_shares):
            pair_w = alpha[start:end]
            pair_e = sq_err[start:end]
            if float(np.sum(pair_w)) <= eps:
                pair_mse = float("inf")
            else:
                pair_mse = float(np.average(pair_e, weights=pair_w))
            pair_stats.append({
                "index": rec["index"],
                "weight_scale": float(scale),
                "weight_share": float(share),
                "pair_mse": pair_mse,
            })

        return {
            "lut_array": lut_array,
            "solver_infos": solver_infos,
            "metrics": metrics,
            "coverage_map": coverage_map,
            "occupied_bins": len(all_bins),
            "lut_occupied_count": len(lut_occupied),
            "shadow_threshold": shadow_threshold,
            "deep_shadow_threshold": deep_threshold,
            "shadow_auto": shadow_info.get("auto_applied", False),
            "pair_stats": pair_stats,
            "times": {
                "sampling_time": sampling_time,
                "solve_time": solve_time,
                "refinement_time": refinement_time,
                "validation_time": validation_time,
            },
            "refinement_diag": refinement_diag,
        }

    # Initial fit
    fit_initial = _fit_once(pair_records, label="initial")
    fit_final = fit_initial
    used_records = list(pair_records)

    total_sampling_time = fit_initial["times"]["sampling_time"]
    total_solve_time = fit_initial["times"]["solve_time"]
    total_refinement_time = fit_initial["times"]["refinement_time"]
    total_validation_time = fit_initial["times"]["validation_time"]

    outlier_diag = {
        "enabled": outlier_sigma > 0,
        "sigma": outlier_sigma,
        "min_pairs_after_outlier": min_pairs_after_outlier,
        "dropped_pair_indices": [],
        "applied": False,
    }

    if outlier_sigma > 0 and len(pair_records) >= max(min_pairs_after_outlier, 3):
        pair_mse = np.array([p["pair_mse"] for p in fit_initial["pair_stats"]], dtype=np.float64)
        finite_mask = np.isfinite(pair_mse)
        if np.any(finite_mask):
            mse_vals = pair_mse[finite_mask]
            median = float(np.median(mse_vals))
            mad = float(np.median(np.abs(mse_vals - median)))
            robust_sigma = 1.4826 * mad
            threshold = median + outlier_sigma * robust_sigma

            outlier_diag["median_pair_mse"] = median
            outlier_diag["mad_pair_mse"] = mad
            outlier_diag["threshold_pair_mse"] = threshold

            dropped = [
                stat["index"]
                for stat in fit_initial["pair_stats"]
                if np.isfinite(stat["pair_mse"]) and stat["pair_mse"] > threshold
            ]

            if dropped and (len(pair_records) - len(dropped)) >= min_pairs_after_outlier:
                outlier_diag["dropped_pair_indices"] = dropped
                outlier_diag["applied"] = True
                used_records = [r for r in pair_records if r["index"] not in set(dropped)]

                logger.info(
                    "Outlier rejection: dropped %d/%d pairs (sigma=%.2f)",
                    len(dropped), len(pair_records), outlier_sigma,
                )

                fit_refit = _fit_once(used_records, label="refit")
                fit_final = fit_refit
                total_sampling_time += fit_refit["times"]["sampling_time"]
                total_solve_time += fit_refit["times"]["solve_time"]
                total_refinement_time += fit_refit["times"]["refinement_time"]
                total_validation_time += fit_refit["times"]["validation_time"]

    diagnostics["outlier_rejection"] = outlier_diag

    # Pair diagnostics (final fit stats + dropped flags)
    final_stats_by_idx = {p["index"]: p for p in fit_final["pair_stats"]}
    dropped_set = set(outlier_diag.get("dropped_pair_indices", []))
    for rec in pair_records:
        stats = final_stats_by_idx.get(rec["index"], {})
        diagnostics["pairs"].append({
            "index": rec["index"],
            "source": rec["source"],
            "target": rec["target"],
            "weight": rec["weight"],
            "occupied_bins": rec["occupied_bins"],
            "width": rec["width"],
            "height": rec["height"],
            "pixels": rec["pixels"],
            "transfer_function": rec["transfer_function"],
            "transfer_fn_override": rec["transfer_fn_override"],
            "shaper_applied": rec["shaper_applied"],
            "normalization_mode": rec["normalization_mode"],
            "normalization_gains": rec["normalization_gains"],
            "normalization_biases": rec["normalization_biases"],
            "dropped_as_outlier": rec["index"] in dropped_set,
            "weight_scale": stats.get("weight_scale"),
            "weight_share": stats.get("weight_share"),
            "pair_mse": stats.get("pair_mse"),
        })

    diagnostics["num_pairs_used"] = len(used_records)
    diagnostics["occupied_bins"] = fit_final["occupied_bins"]
    diagnostics["global_occupied_lut_nodes"] = fit_final["lut_occupied_count"]
    diagnostics["shadow_threshold"] = fit_final["shadow_threshold"]
    diagnostics["deep_shadow_threshold"] = fit_final["deep_shadow_threshold"]
    diagnostics["shadow_auto"] = fit_final["shadow_auto"]
    diagnostics["sampling_time"] = total_sampling_time
    diagnostics["solve_time"] = total_solve_time
    diagnostics["validation_time"] = total_validation_time
    if config.enable_refinement:
        diagnostics["refinement_time"] = total_refinement_time
        diagnostics["refinement"] = fit_final["refinement_diag"]
    diagnostics["solver_info"] = [
        {k: v for k, v in info.items() if not isinstance(v, np.ndarray)}
        for info in fit_final["solver_infos"] if info is not None
    ]

    # ---------------------------------------------------------------
    # Stage 6: Export
    # ---------------------------------------------------------------
    output_path = None
    if config.output_path is not None:
        _emit_progress(progress_callback, "export", 0.0, "Exporting...")
        _check_cancel(cancel_check)

        t0 = time.perf_counter()
        output_path = Path(config.output_path)
        lut_array = fit_final["lut_array"]

        shaper_lut = None
        if pair_shaper and pair_shaper.get("applied") and config.include_shaper is not False:
            if config.format.value == "cube":
                shaper_lut = generate_1d_shaper(pair_shaper["forward"])
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
            from lutsmith.io.arri import export_arri
            export_arri(lut_array, output_path, format=config.format.value)

        diagnostics["export_time"] = time.perf_counter() - t0
        _emit_progress(progress_callback, "export", 1.0, f"Saved: {output_path}")
        logger.info("Export multi-pair: %.2fs -> %s", diagnostics["export_time"], output_path)

    total_time = time.perf_counter() - t_start
    diagnostics["total_time"] = total_time
    logger.info("Multi-pair pipeline complete: %.2fs total", total_time)

    lut_data = create_lut_data(fit_final["lut_array"], config.kernel, config.title)
    return PipelineResult(
        lut=lut_data,
        metrics=fit_final["metrics"],
        coverage_map=fit_final["coverage_map"],
        diagnostics=diagnostics,
        output_path=output_path,
    )
