"""Baseline transform fitting for residual LUT regression.

Fits a parametric baseline T0(x) = h(Mx + b) where:
    M  : 3x3 affine matrix (cross-channel coupling)
    b  : 3-vector bias
    h  : per-channel monotone 1D curves (piecewise-linear, PAVA)

The solver then fits the LUT as a residual delta on top of this baseline,
yielding significantly better extrapolation in sparse / unseen regions.

Alternating optimization:
    1. Fix h  -> fit M, b  via weighted IRLS lstsq on h^{-1}(y) = Mx + b
    2. Fix M, b -> fit h   via binning + PAVA isotonic regression
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lutsmith.config import (
    BASELINE_ALTERNATING_ITERATIONS,
    BASELINE_PAVA_KNOTS,
    BASELINE_QUALITY_THRESHOLD,
    EPSILON,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PiecewiseLinearCurve:
    """Monotone non-decreasing piecewise-linear curve."""

    knots_x: np.ndarray  # sorted, strictly increasing
    knots_y: np.ndarray  # monotone non-decreasing

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the curve at arbitrary x values via linear interpolation."""
        return np.interp(x, self.knots_x, self.knots_y)

    def invert(self) -> PiecewiseLinearCurve:
        """Return the inverse curve (swap x <-> y roles).

        Flat segments (duplicate y values) are deduplicated to ensure
        the inverse knots_x is strictly increasing.
        """
        unique_mask = np.concatenate([[True], np.diff(self.knots_y) > 1e-12])
        inv_x = self.knots_y[unique_mask]
        inv_y = self.knots_x[unique_mask]

        if len(inv_x) < 2:
            # Degenerate: nearly constant curve
            val = float(np.mean(self.knots_x))
            inv_x = np.array([self.knots_y[0], self.knots_y[0] + 1e-8])
            inv_y = np.array([val, val])

        return PiecewiseLinearCurve(knots_x=inv_x, knots_y=inv_y)


@dataclass
class BaselineTransform:
    """Parametric baseline T0(x) = h(Mx + b)."""

    M: np.ndarray                        # (3, 3) affine matrix
    b: np.ndarray                        # (3,) bias vector
    curves: list[PiecewiseLinearCurve]   # per-channel monotone curves

    def evaluate(self, rgb: np.ndarray) -> np.ndarray:
        """Apply the baseline transform to (K, 3) RGB values.

        Returns:
            (K, 3) transformed values.
        """
        z = rgb @ self.M.T + self.b  # (K, 3)
        result = np.column_stack([
            self.curves[c].evaluate(z[:, c]) for c in range(3)
        ])
        return result


# ---------------------------------------------------------------------------
# Pool Adjacent Violators Algorithm (isotonic regression)
# ---------------------------------------------------------------------------

def _pava_isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators for weighted isotonic regression (non-decreasing).

    Args:
        y: (n,) target values.
        w: (n,) positive weights.

    Returns:
        (n,) isotonic (non-decreasing) regression values.
    """
    n = len(y)
    result = np.zeros(n, dtype=np.float64)

    # Block representation: each block is [weighted_sum, weight_sum, start, end]
    blocks: list[list] = []

    for i in range(n):
        blocks.append([float(y[i]) * float(w[i]), float(w[i]), i, i])

        # Pool adjacent violators
        while len(blocks) > 1:
            last = blocks[-1]
            prev = blocks[-2]

            mean_last = last[0] / max(last[1], EPSILON)
            mean_prev = prev[0] / max(prev[1], EPSILON)

            if mean_prev > mean_last:
                # Violation: pool the two blocks
                prev[0] += last[0]
                prev[1] += last[1]
                prev[3] = last[3]
                blocks.pop()
            else:
                break

    # Reconstruct result from blocks
    for ws, wt, start, end in blocks:
        mean = ws / max(wt, EPSILON)
        result[start:end + 1] = mean

    return result


# ---------------------------------------------------------------------------
# Affine fitting with IRLS
# ---------------------------------------------------------------------------

def _fit_affine_irls(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    weights: np.ndarray,
    h_inv_curves: list[PiecewiseLinearCurve] | None = None,
    huber_delta: float = 1.0,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit M, b per-channel via weighted lstsq with Huber IRLS.

    Solves: h^{-1}(y) = Mx + b  (or y = Mx + b if h_inv is None).

    Args:
        input_rgb: (M, 3) input colors.
        output_rgb: (M, 3) output colors.
        weights: (M,) per-sample weights.
        h_inv_curves: Inverse curves for transforming targets.
        huber_delta: Huber delta for IRLS.
        n_iter: Number of IRLS iterations.

    Returns:
        (M_mat, b_vec): (3, 3) matrix and (3,) bias.
    """
    n_samples = len(input_rgb)

    # Transform outputs through h^{-1} if available
    if h_inv_curves is not None:
        target = np.column_stack([
            h_inv_curves[c].evaluate(output_rgb[:, c]) for c in range(3)
        ])
    else:
        target = output_rgb

    # Augmented input: [x1, x2, x3, 1]
    A = np.column_stack([input_rgb, np.ones(n_samples)])  # (M, 4)

    M_mat = np.zeros((3, 3), dtype=np.float64)
    b_vec = np.zeros(3, dtype=np.float64)

    for c in range(3):
        w = weights.copy()
        y_c = target[:, c]

        # Initial weighted solve
        W_sqrt = np.sqrt(np.maximum(w, 0.0))
        Aw = A * W_sqrt[:, np.newaxis]
        bw = y_c * W_sqrt
        theta, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)

        # IRLS iterations for Huber robustness
        for _ in range(n_iter):
            residuals = A @ theta - y_c
            abs_r = np.abs(residuals)
            huber_w = np.where(abs_r < huber_delta, 1.0, huber_delta / (abs_r + EPSILON))
            combined_w = w * huber_w
            W_sqrt = np.sqrt(np.maximum(combined_w, 0.0))
            Aw = A * W_sqrt[:, np.newaxis]
            bw = y_c * W_sqrt
            theta, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)

        M_mat[c, :] = theta[:3]
        b_vec[c] = theta[3]

    return M_mat, b_vec


# ---------------------------------------------------------------------------
# Monotone curve fitting via binning + PAVA
# ---------------------------------------------------------------------------

def _fit_monotone_curves(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    n_knots: int = BASELINE_PAVA_KNOTS,
) -> list[PiecewiseLinearCurve]:
    """Fit per-channel monotone curves from (Mx+b) -> y via PAVA.

    Args:
        input_rgb: (M, 3) input colors.
        output_rgb: (M, 3) output colors.
        M: (3, 3) affine matrix.
        b: (3,) bias vector.
        weights: (M,) per-sample weights.
        n_knots: Number of knots for piecewise-linear curve.

    Returns:
        List of 3 PiecewiseLinearCurve instances.
    """
    # Compute z = Mx + b for all samples
    z = input_rgb @ M.T + b  # (M, 3)

    curves = []
    for c in range(3):
        z_c = z[:, c]
        y_c = output_rgb[:, c]

        z_min, z_max = float(np.min(z_c)), float(np.max(z_c))
        if z_max - z_min < EPSILON:
            # Degenerate: constant z -> constant output
            mean_y = float(np.average(y_c, weights=weights))
            curves.append(PiecewiseLinearCurve(
                knots_x=np.array([z_min - EPSILON, z_max + EPSILON]),
                knots_y=np.array([mean_y, mean_y]),
            ))
            continue

        bin_edges = np.linspace(z_min, z_max, n_knots + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Assign samples to bins
        bin_idx = np.clip(
            np.searchsorted(bin_edges, z_c, side="right") - 1,
            0, n_knots - 1,
        )

        # Compute weighted averages per bin
        bin_y_sum = np.zeros(n_knots, dtype=np.float64)
        bin_w_sum = np.zeros(n_knots, dtype=np.float64)
        np.add.at(bin_y_sum, bin_idx, weights * y_c)
        np.add.at(bin_w_sum, bin_idx, weights)

        filled = bin_w_sum > 0
        if not np.any(filled):
            mean_y = float(np.average(y_c, weights=weights))
            curves.append(PiecewiseLinearCurve(
                knots_x=bin_centers,
                knots_y=np.full(n_knots, mean_y),
            ))
            continue

        bin_y_sum[filled] /= bin_w_sum[filled]

        # Interpolate empty bins from filled neighbors
        if not np.all(filled):
            filled_idx = np.where(filled)[0]
            bin_y_sum[~filled] = np.interp(
                bin_centers[~filled],
                bin_centers[filled_idx],
                bin_y_sum[filled_idx],
            )
            bin_w_sum[~filled] = EPSILON  # tiny weight for interpolated knots

        # Apply PAVA for monotonicity
        bin_y_mono = _pava_isotonic(bin_y_sum, bin_w_sum)

        curves.append(PiecewiseLinearCurve(
            knots_x=bin_centers,
            knots_y=bin_y_mono,
        ))

    return curves


# ---------------------------------------------------------------------------
# Main fitting entry point
# ---------------------------------------------------------------------------

def fit_baseline(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    sample_alpha: np.ndarray,
    n_iter: int = BASELINE_ALTERNATING_ITERATIONS,
    huber_delta: float = 1.0,
    n_knots: int = BASELINE_PAVA_KNOTS,
) -> BaselineTransform:
    """Fit a baseline transform T0(x) = h(Mx + b) via alternating optimization.

    Iteration pattern:
        1. Fix h  -> fit M, b  (weighted IRLS lstsq)
        2. Fix M, b -> fit h   (binning + PAVA isotonic regression)

    Args:
        input_rgb: (M, 3) input colors.
        output_rgb: (M, 3) output colors.
        sample_alpha: (M,) per-sample weights.
        n_iter: Number of alternating optimization iterations.
        huber_delta: Huber delta for affine IRLS.
        n_knots: Number of knots for monotone curves.

    Returns:
        Fitted BaselineTransform.
    """
    logger.info("Fitting baseline transform (%d alternating iterations)...", n_iter)

    # Initialize: M = I, b = 0, h = identity (no curves yet)
    M = np.eye(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    h_inv_curves = None  # identity for first iteration

    for iteration in range(n_iter):
        # Step 1: Fix h -> fit M, b
        M, b = _fit_affine_irls(
            input_rgb, output_rgb, sample_alpha,
            h_inv_curves=h_inv_curves,
            huber_delta=huber_delta,
        )

        # Step 2: Fix M, b -> fit h
        curves = _fit_monotone_curves(
            input_rgb, output_rgb, M, b, sample_alpha, n_knots=n_knots,
        )

        # Compute inverse curves for next iteration's affine fit
        h_inv_curves = [c.invert() for c in curves]

        # Diagnostic: compute current residual
        baseline = BaselineTransform(M=M, b=b, curves=curves)
        pred = baseline.evaluate(input_rgb)
        residual = np.mean(np.sum((output_rgb - pred) ** 2, axis=1))
        logger.info(
            "  Baseline iter %d: MSE=%.6f, M diag=[%.3f, %.3f, %.3f]",
            iteration, residual, M[0, 0], M[1, 1], M[2, 2],
        )

    return BaselineTransform(M=M, b=b, curves=curves)


# ---------------------------------------------------------------------------
# Evaluation and quality gate
# ---------------------------------------------------------------------------

def evaluate_baseline_lut(baseline: BaselineTransform, N: int) -> np.ndarray:
    """Evaluate baseline transform at all N^3 grid nodes.

    Args:
        baseline: Fitted baseline transform.
        N: LUT grid size.

    Returns:
        (N, N, N, 3) LUT array in [r, g, b, ch] indexing.
    """
    from lutsmith.core.lut import identity_lut_flat

    id_flat = identity_lut_flat(N)  # (N^3, 3) grid RGB values
    out_flat = baseline.evaluate(id_flat)

    # Reshape: flat convention b*N*N + g*N + r -> [b, g, r] axes
    lut_bgr = out_flat.reshape(N, N, N, 3)
    lut = np.transpose(lut_bgr, (2, 1, 0, 3))  # -> [r, g, b, ch]
    return lut.astype(np.float32)


def baseline_quality_gate(
    baseline: BaselineTransform,
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    sample_alpha: np.ndarray,
    threshold: float = BASELINE_QUALITY_THRESHOLD,
) -> bool:
    """Check whether the baseline is significantly better than identity.

    Returns True if the baseline should be used (MSE ratio < threshold,
    meaning at least (1-threshold)*100% improvement over identity).

    Args:
        baseline: Fitted baseline transform.
        input_rgb: (M, 3) input colors.
        output_rgb: (M, 3) output colors.
        sample_alpha: (M,) per-sample weights.
        threshold: Maximum MSE ratio (baseline/identity) to accept.

    Returns:
        True if baseline passes the quality gate.
    """
    # Baseline predictions
    pred_baseline = baseline.evaluate(input_rgb)
    residual_baseline = output_rgb - pred_baseline
    mse_baseline = float(np.average(
        np.sum(residual_baseline ** 2, axis=1),
        weights=sample_alpha,
    ))

    # Identity predictions (output = input)
    residual_identity = output_rgb - input_rgb
    mse_identity = float(np.average(
        np.sum(residual_identity ** 2, axis=1),
        weights=sample_alpha,
    ))

    ratio = mse_baseline / max(mse_identity, EPSILON)

    logger.info(
        "Baseline quality gate: MSE_baseline=%.6f, MSE_identity=%.6f, "
        "ratio=%.4f (threshold=%.4f) -> %s",
        mse_baseline, mse_identity, ratio, threshold,
        "PASS" if ratio < threshold else "FAIL",
    )

    return ratio < threshold
