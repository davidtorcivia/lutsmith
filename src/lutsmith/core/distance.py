"""Distance-to-data computation for adaptive prior strength.

Nodes far from observed data should be pulled more strongly toward the
identity prior, since there is no evidence to constrain them.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from lutsmith.config import PRIOR_DISTANCE_SCALE


def compute_distance_to_data(
    occupied_flat_indices: np.ndarray,
    N: int,
) -> np.ndarray:
    """Compute per-node distance to nearest occupied bin.

    Uses the Euclidean distance transform on a 3D boolean grid.

    Args:
        occupied_flat_indices: 1D array of flat indices of bins with data.
        N: LUT grid size per axis.

    Returns:
        (N^3,) array of distances in grid-step units.
    """
    # Create binary grid: True where there is NO data
    grid = np.ones((N, N, N), dtype=bool)

    if len(occupied_flat_indices) > 0:
        # Convert flat indices to (r, g, b) using our convention
        r = occupied_flat_indices % N
        g = (occupied_flat_indices // N) % N
        b = occupied_flat_indices // (N * N)
        grid[r, g, b] = False

    distances = distance_transform_edt(grid)
    return distances.ravel()  # Flatten in C order: [r, g, b] -> r varies fastest


def prior_strength_from_distance(
    distances: np.ndarray,
    scale: float = PRIOR_DISTANCE_SCALE,
) -> np.ndarray:
    """Convert distance-to-data into per-node prior strength (beta_j).

    Nodes at distance 0 (data present) get beta ~ 0 (no prior needed).
    Nodes far from data get beta ~ 1 (full prior strength).

    Uses an exponential ramp: beta = 1 - exp(-distance / scale).

    Args:
        distances: (N^3,) array of distances.
        scale: Distance at which prior reaches ~63% of maximum.

    Returns:
        (N^3,) array of prior strengths in [0, 1].
    """
    return 1.0 - np.exp(-distances / max(scale, 1e-8))
