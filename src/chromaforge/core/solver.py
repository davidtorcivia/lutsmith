"""Sparse least-squares solver with IRLS for robust loss functions.

Uses scipy.sparse.linalg.lsmr for the core solve, with an optional
outer IRLS (Iteratively Reweighted Least Squares) loop for Huber loss.

Per-channel solving: the system is solved independently for R, G, B
using the same interpolation weights and Laplacian.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lsmr

from chromaforge.config import (
    EPSILON,
    IRLS_CONVERGENCE_TOL,
    LSMR_ATOL,
    LSMR_BTOL,
    LSMR_DEFAULT_MAXITER,
    MAX_IRLS_ITERATIONS,
    MAX_LSMR_ITERATIONS,
)
from chromaforge.core.lut import identity_lut_flat
from chromaforge.core.matrix import (
    build_data_matrix,
    build_data_rhs,
    build_full_system,
    build_prior_matrix,
    build_smoothness_matrix,
    compute_interpolation_weights,
)
from chromaforge.core.distance import compute_distance_to_data
from chromaforge.errors import SolverDivergenceError

logger = logging.getLogger(__name__)


def solve_lsmr(
    A,
    b: np.ndarray,
    atol: float = LSMR_ATOL,
    btol: float = LSMR_BTOL,
    maxiter: int = LSMR_DEFAULT_MAXITER,
) -> tuple[np.ndarray, dict]:
    """Solve a sparse least-squares system using LSMR.

    Args:
        A: Sparse matrix.
        b: RHS vector.
        atol, btol: Convergence tolerances.
        maxiter: Maximum iterations.

    Returns:
        (x, info): Solution vector and convergence info.
    """
    maxiter = min(maxiter, MAX_LSMR_ITERATIONS)
    result = lsmr(A, b, damp=0.0, atol=atol, btol=btol, maxiter=maxiter)
    x = result[0]
    istop = result[1]
    itn = result[2]

    info = {
        "istop": istop,
        "iterations": itn,
        "converged": istop in (1, 2, 3, 4),
    }

    if istop >= 5:
        logger.warning(
            "LSMR solver warning: istop=%d after %d iterations "
            "(5=max iter, 6=cond too large, 7=atol/btol too small)",
            istop, itn,
        )

    return x, info


def solve_irls(
    A,
    b: np.ndarray,
    loss: str = "huber",
    delta: float = 1.0,
    max_iter: int = 3,
    tol: float = IRLS_CONVERGENCE_TOL,
    lsmr_atol: float = LSMR_ATOL,
    lsmr_btol: float = LSMR_BTOL,
    lsmr_maxiter: int = LSMR_DEFAULT_MAXITER,
) -> tuple[np.ndarray, dict]:
    """Solve with IRLS outer loop for robust loss.

    Args:
        A: Sparse matrix.
        b: RHS vector.
        loss: Loss function ("l2" or "huber").
        delta: Huber delta parameter.
        max_iter: Maximum IRLS iterations (capped at MAX_IRLS_ITERATIONS).
        tol: Convergence tolerance (relative change in x).
        lsmr_atol, lsmr_btol, lsmr_maxiter: Inner solver parameters.

    Returns:
        (x, info): Solution and convergence info.
    """
    max_iter = min(max_iter, MAX_IRLS_ITERATIONS)

    # Initial L2 solve
    x, info = solve_lsmr(A, b, atol=lsmr_atol, btol=lsmr_btol, maxiter=lsmr_maxiter)

    if loss == "l2":
        info["irls_iterations"] = 0
        info["irls_converged"] = True
        return x, info

    if loss != "huber":
        raise ValueError(f"Unknown loss function: {loss!r}")

    irls_info = {"irls_iterations": 0, "irls_converged": False}

    for iteration in range(max_iter):
        residuals = A @ x - b
        abs_r = np.abs(residuals)

        # Huber weights
        weights = np.where(abs_r < delta, 1.0, delta / (abs_r + EPSILON))

        # Reweight the system: sqrt(weights) for proper LSQ
        W_sqrt = diags(np.sqrt(weights))
        Aw = W_sqrt @ A
        bw = W_sqrt @ b

        x_new, step_info = solve_lsmr(
            Aw, bw, atol=lsmr_atol, btol=lsmr_btol, maxiter=lsmr_maxiter
        )

        # Check convergence
        x_norm = np.linalg.norm(x)
        relative_change = np.linalg.norm(x_new - x) / (x_norm + EPSILON)
        irls_info["irls_iterations"] = iteration + 1

        logger.debug(
            "IRLS iteration %d: relative_change=%.6e",
            iteration + 1, relative_change,
        )

        if relative_change < tol:
            irls_info["irls_converged"] = True
            x = x_new
            break
        x = x_new

    irls_info.update(info)
    return x, irls_info


def solve_per_channel(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    sample_alpha: np.ndarray,
    N: int,
    lambda_s: float,
    lambda_r: float,
    kernel: str = "tetrahedral",
    loss: str = "huber",
    huber_delta: float = 1.0,
    irls_iterations: int = 3,
    occupied_flat_indices: Optional[np.ndarray] = None,
    distance_scale: float = 3.0,
    parallel: bool = True,
    shadow_threshold: float | None = None,
    deep_threshold: float | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Solve the LUT regression independently for each RGB channel.

    Pre-computes shared components (interpolation weights, Laplacian,
    distances) once and reuses them for all three channels.

    Args:
        input_rgb: (M, 3) input sample colors.
        output_rgb: (M, 3) output sample colors.
        sample_alpha: (M,) per-sample weights.
        N: LUT grid size.
        lambda_s: Smoothness regularization.
        lambda_r: Prior regularization.
        kernel: Interpolation kernel.
        loss: Robust loss function.
        huber_delta: Huber delta.
        irls_iterations: Number of IRLS iterations.
        occupied_flat_indices: Flat indices of occupied bins.
        distance_scale: Prior strength distance scale.
        parallel: Whether to solve channels in parallel threads.
        shadow_threshold: Upper luminance boundary for shadow smoothness boost.
        deep_threshold: Luminance boundary for maximum smoothness boost.

    Returns:
        (lut, infos): (N, N, N, 3) LUT array and list of 3 solver info dicts.
    """
    total = N ** 3

    # Pre-compute shared components
    logger.info("Computing interpolation weights for %d samples...", len(input_rgb))
    corner_indices, corner_weights = compute_interpolation_weights(input_rgb, N, kernel)

    logger.info("Building smoothness matrix (N=%d)...", N)
    laplacian_cached = build_smoothness_matrix(
        N, lambda_s,
        shadow_threshold=shadow_threshold,
        deep_threshold=deep_threshold,
    )

    logger.info("Computing distance-to-data...")
    if occupied_flat_indices is not None:
        distances = compute_distance_to_data(occupied_flat_indices, N)
    else:
        distances = np.ones(total)

    # Identity LUT values for prior
    id_flat = identity_lut_flat(N)

    def solve_channel(ch: int) -> tuple[np.ndarray, dict]:
        """Solve for one output channel."""
        A, b = build_full_system(
            input_rgb=input_rgb,
            output_channel=output_rgb[:, ch],
            sample_alpha=sample_alpha,
            N=N,
            lambda_s=lambda_s,
            lambda_r=lambda_r,
            kernel=kernel,
            precomputed_weights=(corner_indices, corner_weights),
            precomputed_laplacian=laplacian_cached,
            precomputed_distances=distances,
            prior_channel=id_flat[:, ch],
            distance_scale=distance_scale,
        )

        x, info = solve_irls(
            A, b,
            loss=loss,
            delta=huber_delta,
            max_iter=irls_iterations,
        )

        if not info.get("converged", True):
            logger.warning("Channel %d: LSMR did not fully converge (istop=%d)",
                           ch, info.get("istop", -1))

        return x, info

    # Solve all 3 channels
    channel_names = ["R", "G", "B"]
    lut = np.zeros((N, N, N, 3), dtype=np.float32)
    infos = [None, None, None]

    if parallel and N >= 17:
        # scipy releases the GIL during BLAS operations
        logger.info("Solving 3 channels in parallel...")
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(solve_channel, ch): ch for ch in range(3)}
            for future in futures:
                ch = futures[future]
                x, info = future.result()
                # Reshape: flat convention is b*N*N + g*N + r
                # So reshape to (N, N, N) gives [b, g, r] ordering
                # Then transpose to [r, g, b]
                lut_bgr = x.reshape((N, N, N))
                lut[:, :, :, ch] = np.transpose(lut_bgr, (2, 1, 0))
                infos[ch] = info
                logger.info("Channel %s solved: %s", channel_names[ch],
                            "converged" if info.get("irls_converged", info.get("converged"))
                            else "not converged")
    else:
        for ch in range(3):
            logger.info("Solving channel %s...", channel_names[ch])
            x, info = solve_channel(ch)
            lut_bgr = x.reshape((N, N, N))
            lut[:, :, :, ch] = np.transpose(lut_bgr, (2, 1, 0))
            infos[ch] = info
            logger.info("Channel %s solved: %s", channel_names[ch],
                        "converged" if info.get("irls_converged", info.get("converged"))
                        else "not converged")

    return lut, infos
