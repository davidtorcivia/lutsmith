"""Sparse linear system construction for lattice regression.

Builds the stacked system:
    [ sqrt(W_d) * A_d ]       [ sqrt(W_d) * b_d ]
    [ sqrt(ls) * A_s  ] * L = [ 0                ]
    [ sqrt(lr*B) * I  ]       [ sqrt(lr*B) * L0  ]

Where:
    A_d = data fidelity (interpolation weights)
    A_s = smoothness (3D Laplacian)
    I   = identity (prior toward L0)
    W_d = per-sample weights alpha_i
    B   = per-node prior strengths beta_j
    L0  = prior LUT (identity by default)

CRITICAL: Rows are scaled by sqrt(weight), not weight, for proper LSQ.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack

from chromaforge.core.interpolation import vectorized_trilinear, vectorized_tetrahedral
from chromaforge.core.laplacian import build_laplacian_vectorized
from chromaforge.core.distance import compute_distance_to_data, prior_strength_from_distance
from chromaforge.core.lut import identity_lut_flat


def compute_interpolation_weights(
    input_rgb: np.ndarray,
    N: int,
    kernel: str = "tetrahedral",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute interpolation weights for all samples.

    This is called ONCE and reused for all three RGB channels.

    Args:
        input_rgb: (M, 3) input colors in [0, 1].
        N: LUT grid size.
        kernel: "trilinear" or "tetrahedral".

    Returns:
        (indices, weights): (M, K) arrays where K=8 (trilinear) or K=4 (tetrahedral).
    """
    if kernel == "tetrahedral":
        return vectorized_tetrahedral(input_rgb, N)
    elif kernel == "trilinear":
        return vectorized_trilinear(input_rgb, N)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")


def build_data_matrix(
    corner_indices: np.ndarray,
    corner_weights: np.ndarray,
    sample_alpha: np.ndarray,
    N: int,
) -> tuple[csr_matrix, None]:
    """Build the data fidelity part of the sparse system.

    The RHS is not built here because it depends on the channel;
    only the matrix (which is shared across channels) is constructed.

    Args:
        corner_indices: (M, K) flat LUT indices per sample.
        corner_weights: (M, K) interpolation weights per sample.
        sample_alpha: (M,) per-sample weights.
        N: LUT grid size.

    Returns:
        CSR matrix of shape (M, N^3).
    """
    M, K = corner_indices.shape
    total_cols = N ** 3

    # Scale weights by sqrt(alpha) -- CRITICAL for proper LSQ
    sqrt_alpha = np.sqrt(np.maximum(sample_alpha, 0.0))  # (M,)
    scaled_weights = corner_weights * sqrt_alpha[:, np.newaxis]  # (M, K)

    # Build COO arrays
    row_idx = np.repeat(np.arange(M, dtype=np.int64), K)
    col_idx = corner_indices.ravel().astype(np.int64)
    vals = scaled_weights.ravel()

    A_data = coo_matrix(
        (vals, (row_idx, col_idx)),
        shape=(M, total_cols),
    ).tocsr()

    return A_data


def build_data_rhs(
    output_channel: np.ndarray,
    sample_alpha: np.ndarray,
) -> np.ndarray:
    """Build the RHS for one output channel.

    Args:
        output_channel: (M,) output values for one channel.
        sample_alpha: (M,) per-sample weights.

    Returns:
        (M,) scaled RHS vector.
    """
    sqrt_alpha = np.sqrt(np.maximum(sample_alpha, 0.0))
    return sqrt_alpha * output_channel


def build_smoothness_matrix(
    N: int,
    lambda_s: float,
) -> tuple[csr_matrix, np.ndarray]:
    """Build the smoothness (Laplacian) part of the system.

    Args:
        N: LUT grid size.
        lambda_s: Smoothness regularization strength.

    Returns:
        (matrix, rhs): CSR matrix (N^3, N^3) and zero RHS (N^3,).
    """
    L = build_laplacian_vectorized(N)
    sqrt_ls = np.sqrt(max(lambda_s, 0.0))
    A_smooth = sqrt_ls * L
    b_smooth = np.zeros(N ** 3, dtype=np.float64)
    return A_smooth, b_smooth


def build_prior_matrix(
    N: int,
    lambda_r: float,
    distances: np.ndarray,
    prior_channel: np.ndarray,
    distance_scale: float = 3.0,
) -> tuple[csr_matrix, np.ndarray]:
    """Build the identity prior part of the system.

    Args:
        N: LUT grid size.
        lambda_r: Prior regularization strength.
        distances: (N^3,) distance-to-data per node.
        prior_channel: (N^3,) prior values for this channel.
        distance_scale: Distance scaling for prior strength.

    Returns:
        (matrix, rhs): Diagonal CSR matrix (N^3, N^3) and prior RHS (N^3,).
    """
    total = N ** 3
    beta = prior_strength_from_distance(distances, scale=distance_scale)
    sqrt_lr_beta = np.sqrt(max(lambda_r, 0.0)) * np.sqrt(beta)

    # Diagonal matrix
    A_prior = coo_matrix(
        (sqrt_lr_beta, (np.arange(total), np.arange(total))),
        shape=(total, total),
    ).tocsr()

    b_prior = sqrt_lr_beta * prior_channel
    return A_prior, b_prior


def build_full_system(
    input_rgb: np.ndarray,
    output_channel: np.ndarray,
    sample_alpha: np.ndarray,
    N: int,
    lambda_s: float,
    lambda_r: float,
    kernel: str = "tetrahedral",
    precomputed_weights: tuple[np.ndarray, np.ndarray] | None = None,
    precomputed_laplacian: tuple[csr_matrix, np.ndarray] | None = None,
    precomputed_distances: np.ndarray | None = None,
    occupied_flat_indices: np.ndarray | None = None,
    prior_channel: np.ndarray | None = None,
    distance_scale: float = 3.0,
) -> tuple[csr_matrix, np.ndarray]:
    """Build the complete stacked sparse system for one output channel.

    Accepts pre-computed components to avoid redundant work across channels.

    Args:
        input_rgb: (M, 3) input colors.
        output_channel: (M,) output values for one channel.
        sample_alpha: (M,) per-sample weights.
        N: LUT grid size.
        lambda_s: Smoothness strength.
        lambda_r: Prior strength.
        kernel: Interpolation kernel.
        precomputed_weights: (indices, weights) from compute_interpolation_weights.
        precomputed_laplacian: (A_smooth, b_smooth) from build_smoothness_matrix.
        precomputed_distances: (N^3,) distances from compute_distance_to_data.
        occupied_flat_indices: Flat indices of occupied bins (for distance computation).
        prior_channel: (N^3,) prior values for this channel.
        distance_scale: Scale for prior strength falloff.

    Returns:
        (A, b): Stacked CSR matrix and RHS vector.
    """
    total = N ** 3

    # 1. Interpolation weights (compute once, reuse)
    if precomputed_weights is not None:
        corner_indices, corner_weights = precomputed_weights
    else:
        corner_indices, corner_weights = compute_interpolation_weights(input_rgb, N, kernel)

    # 2. Data fidelity
    A_data = build_data_matrix(corner_indices, corner_weights, sample_alpha, N)
    b_data = build_data_rhs(output_channel, sample_alpha)

    # 3. Smoothness
    if precomputed_laplacian is not None:
        A_smooth, b_smooth = precomputed_laplacian
    else:
        A_smooth, b_smooth = build_smoothness_matrix(N, lambda_s)

    # 4. Prior
    if precomputed_distances is None:
        if occupied_flat_indices is not None:
            precomputed_distances = compute_distance_to_data(occupied_flat_indices, N)
        else:
            # No distance info: use uniform prior
            precomputed_distances = np.ones(total)

    if prior_channel is None:
        # Default: identity LUT values for this channel
        id_flat = identity_lut_flat(N)
        # We need to figure out which channel this is --
        # caller should provide prior_channel explicitly
        prior_channel = id_flat[:, 0]  # fallback to R channel

    A_prior, b_prior = build_prior_matrix(
        N, lambda_r, precomputed_distances, prior_channel, distance_scale
    )

    # 5. Stack vertically
    A = vstack([A_data, A_smooth, A_prior], format="csr")
    b = np.concatenate([b_data, b_smooth, b_prior])

    return A, b
