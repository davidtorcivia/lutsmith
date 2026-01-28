"""3D discrete Laplacian construction for LUT smoothness regularization.

The Laplacian operates on the N^3 grid of LUT nodes. Each row enforces:
center_value * weighted_degree - sum(weighted_neighbor_values) = 0,
which penalizes deviation from a weighted local average.

Supported connectivity:
    6  - face-adjacent neighbors only (default)
    18 - faces + edge-diagonal neighbors
    26 - faces + edge-diagonals + corner-diagonal neighbors

Properties:
    - Symmetric
    - Row sums equal zero
    - Constant vectors in the null space
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from lutsmith.core.types import flat_index, grid_indices


def build_laplacian(N: int) -> csr_matrix:
    """Build the 3D discrete Laplacian as a sparse matrix.

    Uses 6-connectivity: each interior node has 6 face-adjacent neighbors.
    Boundary nodes have fewer neighbors (3 for corners, 4 for edges, 5 for faces).

    Args:
        N: LUT grid size per axis.

    Returns:
        CSR sparse matrix of shape (N^3, N^3).
    """
    total = N ** 3

    # Pre-allocate: each node has at most 7 entries (1 center + 6 neighbors)
    # Total non-zeros <= 7 * N^3
    rows = []
    cols = []
    vals = []

    for b in range(N):
        for g in range(N):
            for r in range(N):
                node = b * N * N + g * N + r
                num_neighbors = 0

                # Check all 6 face-adjacent neighbors
                if r > 0:
                    rows.append(node)
                    cols.append(b * N * N + g * N + (r - 1))
                    vals.append(-1.0)
                    num_neighbors += 1
                if r < N - 1:
                    rows.append(node)
                    cols.append(b * N * N + g * N + (r + 1))
                    vals.append(-1.0)
                    num_neighbors += 1
                if g > 0:
                    rows.append(node)
                    cols.append(b * N * N + (g - 1) * N + r)
                    vals.append(-1.0)
                    num_neighbors += 1
                if g < N - 1:
                    rows.append(node)
                    cols.append(b * N * N + (g + 1) * N + r)
                    vals.append(-1.0)
                    num_neighbors += 1
                if b > 0:
                    rows.append(node)
                    cols.append((b - 1) * N * N + g * N + r)
                    vals.append(-1.0)
                    num_neighbors += 1
                if b < N - 1:
                    rows.append(node)
                    cols.append((b + 1) * N * N + g * N + r)
                    vals.append(-1.0)
                    num_neighbors += 1

                # Center (diagonal) coefficient
                rows.append(node)
                cols.append(node)
                vals.append(float(num_neighbors))

    return coo_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(total, total),
    ).tocsr()


def build_laplacian_vectorized(N: int) -> csr_matrix:
    """Build the 3D Laplacian using vectorized NumPy operations.

    Significantly faster than the loop-based version for large N (>= 33).

    Args:
        N: LUT grid size per axis.

    Returns:
        CSR sparse matrix of shape (N^3, N^3).
    """
    total = N ** 3

    # Generate all node indices
    b_idx, g_idx, r_idx = np.mgrid[0:N, 0:N, 0:N]
    b_flat = b_idx.ravel()
    g_flat = g_idx.ravel()
    r_flat = r_idx.ravel()
    nodes = np.arange(total, dtype=np.int64)

    all_rows = []
    all_cols = []
    all_vals = []
    neighbor_count = np.zeros(total, dtype=np.float64)

    # For each direction, find valid neighbors and create entries
    # R-1
    mask = r_flat > 0
    src = nodes[mask]
    dst = b_flat[mask] * N * N + g_flat[mask] * N + (r_flat[mask] - 1)
    all_rows.append(src)
    all_cols.append(dst)
    all_vals.append(np.full(len(src), -1.0))
    neighbor_count[mask] += 1

    # R+1
    mask = r_flat < N - 1
    src = nodes[mask]
    dst = b_flat[mask] * N * N + g_flat[mask] * N + (r_flat[mask] + 1)
    all_rows.append(src)
    all_cols.append(dst)
    all_vals.append(np.full(len(src), -1.0))
    neighbor_count[mask] += 1

    # G-1
    mask = g_flat > 0
    src = nodes[mask]
    dst = b_flat[mask] * N * N + (g_flat[mask] - 1) * N + r_flat[mask]
    all_rows.append(src)
    all_cols.append(dst)
    all_vals.append(np.full(len(src), -1.0))
    neighbor_count[mask] += 1

    # G+1
    mask = g_flat < N - 1
    src = nodes[mask]
    dst = b_flat[mask] * N * N + (g_flat[mask] + 1) * N + r_flat[mask]
    all_rows.append(src)
    all_cols.append(dst)
    all_vals.append(np.full(len(src), -1.0))
    neighbor_count[mask] += 1

    # B-1
    mask = b_flat > 0
    src = nodes[mask]
    dst = (b_flat[mask] - 1) * N * N + g_flat[mask] * N + r_flat[mask]
    all_rows.append(src)
    all_cols.append(dst)
    all_vals.append(np.full(len(src), -1.0))
    neighbor_count[mask] += 1

    # B+1
    mask = b_flat < N - 1
    src = nodes[mask]
    dst = (b_flat[mask] + 1) * N * N + g_flat[mask] * N + r_flat[mask]
    all_rows.append(src)
    all_cols.append(dst)
    all_vals.append(np.full(len(src), -1.0))
    neighbor_count[mask] += 1

    # Diagonal: center coefficients
    all_rows.append(nodes)
    all_cols.append(nodes)
    all_vals.append(neighbor_count)

    rows_arr = np.concatenate(all_rows)
    cols_arr = np.concatenate(all_cols)
    vals_arr = np.concatenate(all_vals)

    return coo_matrix(
        (vals_arr, (rows_arr, cols_arr)),
        shape=(total, total),
    ).tocsr()


def build_laplacian_extended(N: int, connectivity: int = 6) -> csr_matrix:
    """Build a 3D graph Laplacian with configurable connectivity.

    Supports 6 (face), 18 (face+edge), and 26 (face+edge+corner) neighbors.
    Edge-diagonal neighbors are weighted by 1/sqrt(2) and corner-diagonals
    by 1/sqrt(3) to account for their greater Euclidean distance.

    The diagonal is normalized so that interior nodes have the same
    weighted degree as the 6-connected case (sum = 6.0), which keeps
    the meaning of lambda_s consistent across connectivity modes.

    Args:
        N: LUT grid size per axis.
        connectivity: 6, 18, or 26.

    Returns:
        CSR sparse matrix of shape (N^3, N^3).
    """
    if connectivity == 6:
        return build_laplacian_vectorized(N)

    if connectivity not in (18, 26):
        raise ValueError(f"Unsupported connectivity: {connectivity}. Use 6, 18, or 26.")

    total = N ** 3

    # Generate all node indices
    b_idx, g_idx, r_idx = np.mgrid[0:N, 0:N, 0:N]
    b_flat = b_idx.ravel()
    g_flat = g_idx.ravel()
    r_flat = r_idx.ravel()
    nodes = np.arange(total, dtype=np.int64)

    # Build neighbor offsets with weights
    # Face neighbors: 6 directions, weight 1.0
    offsets = []
    for axis_delta in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        offsets.append((axis_delta, 1.0))

    # Edge-diagonal neighbors: 12 directions, weight 1/sqrt(2)
    if connectivity >= 18:
        w_edge = 1.0 / np.sqrt(2.0)
        for dr, dg in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            offsets.append(((dr, dg, 0), w_edge))
        for dr, db in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            offsets.append(((dr, 0, db), w_edge))
        for dg, db in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            offsets.append(((0, dg, db), w_edge))

    # Corner-diagonal neighbors: 8 directions, weight 1/sqrt(3)
    if connectivity == 26:
        w_corner = 1.0 / np.sqrt(3.0)
        for dr in (-1, 1):
            for dg in (-1, 1):
                for db in (-1, 1):
                    offsets.append(((dr, dg, db), w_corner))

    all_rows = []
    all_cols = []
    all_vals = []
    weighted_degree = np.zeros(total, dtype=np.float64)

    for (dr, dg, db), w in offsets:
        # Build validity mask
        mask = np.ones(total, dtype=bool)
        nr = r_flat + dr
        ng = g_flat + dg
        nb = b_flat + db

        mask &= (nr >= 0) & (nr < N)
        mask &= (ng >= 0) & (ng < N)
        mask &= (nb >= 0) & (nb < N)

        src = nodes[mask]
        dst = nb[mask] * N * N + ng[mask] * N + nr[mask]

        all_rows.append(src)
        all_cols.append(dst)
        all_vals.append(np.full(len(src), -w))
        weighted_degree[mask] += w

    # Normalize: scale so interior weighted degree = 6.0
    # Interior node of 6-connected has degree 6.0
    # Interior node of 18-connected has raw degree 6*1 + 12/sqrt(2) ≈ 14.485
    # Interior node of 26-connected has raw degree 6*1 + 12/sqrt(2) + 8/sqrt(3) ≈ 19.104
    interior_mask = (
        (r_flat > 0) & (r_flat < N - 1) &
        (g_flat > 0) & (g_flat < N - 1) &
        (b_flat > 0) & (b_flat < N - 1)
    )
    if np.any(interior_mask):
        interior_degree = weighted_degree[interior_mask][0]
        norm_factor = 6.0 / max(interior_degree, 1e-8)
    else:
        norm_factor = 1.0

    # Scale all off-diagonal values and weighted degrees
    weighted_degree *= norm_factor
    scaled_vals = [v * norm_factor for v in all_vals]

    # Add diagonal
    all_rows.append(nodes)
    all_cols.append(nodes)
    scaled_vals.append(weighted_degree)

    rows_arr = np.concatenate(all_rows)
    cols_arr = np.concatenate(all_cols)
    vals_arr = np.concatenate(scaled_vals)

    return coo_matrix(
        (vals_arr, (rows_arr, cols_arr)),
        shape=(total, total),
    ).tocsr()
