"""3D discrete Laplacian construction for LUT smoothness regularization.

The Laplacian operates on the N^3 grid of LUT nodes using 6-connectivity
(face-adjacent neighbors). Each row enforces: center_value * num_neighbors
- sum(neighbor_values) = 0, which penalizes deviation from local average.

Properties:
    - Symmetric
    - Row sums equal zero
    - Constant vectors in the null space
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from chromaforge.core.types import flat_index, grid_indices


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
