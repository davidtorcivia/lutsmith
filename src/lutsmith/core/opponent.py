"""Opponent color space utilities for anisotropic regularization.

Provides an orthonormal opponent-space basis separating luminance (Y)
from chrominance (C1, C2).  When the solver operates in opponent space,
smoothness and prior regularization can be applied independently to
luminance and chroma channels.

Basis rows (orthonormal):
    u0 = (1, 1, 1) / sqrt(3)      -- luminance (equal-energy achromatic)
    u1 = (1, -1, 0) / sqrt(2)     -- red-green chroma
    u2 = (1, 1, -2) / sqrt(6)     -- yellow-blue chroma

Since the matrix is orthonormal, its inverse is its transpose.
"""

from __future__ import annotations

import numpy as np

# Orthonormal opponent basis: rows are basis vectors
OPPONENT_BASIS = np.array([
    [1.0, 1.0, 1.0],     # u0: luminance
    [1.0, -1.0, 0.0],    # u1: red-green
    [1.0, 1.0, -2.0],    # u2: yellow-blue
], dtype=np.float64)

# Normalize each row to unit length
OPPONENT_BASIS[0] /= np.sqrt(3.0)
OPPONENT_BASIS[1] /= np.sqrt(2.0)
OPPONENT_BASIS[2] /= np.sqrt(6.0)

# Inverse is transpose (orthonormal property)
OPPONENT_BASIS_INV = OPPONENT_BASIS.T.copy()


def rgb_to_opponent(rgb: np.ndarray) -> np.ndarray:
    """Convert (K, 3) RGB values to opponent space [Y, C1, C2].

    Args:
        rgb: (..., 3) array of RGB values.

    Returns:
        Same-shape array in opponent space.
    """
    return rgb @ OPPONENT_BASIS.T


def opponent_to_rgb(opp: np.ndarray) -> np.ndarray:
    """Convert (K, 3) opponent [Y, C1, C2] values back to RGB.

    Args:
        opp: (..., 3) array of opponent-space values.

    Returns:
        Same-shape array in RGB.
    """
    return opp @ OPPONENT_BASIS_INV.T


def compute_neutral_chroma_boost(
    N: int,
    k: float = 3.0,
    sigma: float = 0.12,
) -> np.ndarray:
    """Compute per-node prior boost for nodes near the neutral axis.

    Nodes close to the achromatic axis (r == g == b) get a stronger
    prior pull, discouraging spurious chroma in neutral regions.

    The boost is:  1 + k * exp(-d^2 / (2 * sigma^2))
    where d is the Euclidean distance from the neutral axis in RGB space.

    Args:
        N: LUT grid size.
        k: Maximum boost factor above 1.0 at the neutral axis.
        sigma: Gaussian width in RGB-unit distance.

    Returns:
        (N^3,) array of boost factors >= 1.0.
    """
    total = N ** 3

    # Compute (r, g, b) coordinates for each flat-index node
    flat = np.arange(total, dtype=np.float64)
    node_r = (flat % N) / max(N - 1, 1)
    node_g = ((flat // N) % N) / max(N - 1, 1)
    node_b = (flat // (N * N)) / max(N - 1, 1)

    # Distance from neutral axis: the neutral axis is the line r == g == b.
    # Project each point onto the axis and compute perpendicular distance.
    mean_rgb = (node_r + node_g + node_b) / 3.0
    dist_sq = (
        (node_r - mean_rgb) ** 2
        + (node_g - mean_rgb) ** 2
        + (node_b - mean_rgb) ** 2
    )

    boost = 1.0 + k * np.exp(-dist_sq / (2.0 * sigma ** 2))
    return boost
