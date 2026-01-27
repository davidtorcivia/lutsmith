"""Trilinear and tetrahedral interpolation for 3D LUTs.

Both scalar (single point) and vectorized (batch) variants are provided.
All functions use the flat indexing convention: flat = b*N*N + g*N + r
(R varies fastest, matching the .cube spec).
"""

from __future__ import annotations

import numpy as np

from lutsmith.core.types import flat_index_array


# ---------------------------------------------------------------------------
# Scalar interpolation (single point)
# ---------------------------------------------------------------------------

def trilinear_weights(rgb: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute trilinear interpolation weights for a single point.

    Args:
        rgb: (3,) array with values in [0, 1].
        N: LUT grid size per axis.

    Returns:
        (corner_indices, weights): flat indices (8,) and weights (8,) summing to 1.0.
    """
    scaled = np.clip(rgb, 0.0, 1.0) * (N - 1)
    r0 = min(int(np.floor(scaled[0])), N - 2)
    g0 = min(int(np.floor(scaled[1])), N - 2)
    b0 = min(int(np.floor(scaled[2])), N - 2)
    fr = scaled[0] - r0
    fg = scaled[1] - g0
    fb = scaled[2] - b0

    # 8 corners of the enclosing cube cell
    r_corners = np.array([r0, r0 + 1, r0, r0 + 1, r0, r0 + 1, r0, r0 + 1])
    g_corners = np.array([g0, g0, g0 + 1, g0 + 1, g0, g0, g0 + 1, g0 + 1])
    b_corners = np.array([b0, b0, b0, b0, b0 + 1, b0 + 1, b0 + 1, b0 + 1])

    weights = np.array([
        (1 - fr) * (1 - fg) * (1 - fb),
        fr * (1 - fg) * (1 - fb),
        (1 - fr) * fg * (1 - fb),
        fr * fg * (1 - fb),
        (1 - fr) * (1 - fg) * fb,
        fr * (1 - fg) * fb,
        (1 - fr) * fg * fb,
        fr * fg * fb,
    ])

    flat = flat_index_array(r_corners, g_corners, b_corners, N)
    return flat, weights


def tetrahedral_weights(rgb: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute tetrahedral interpolation weights for a single point.

    The unit cube cell is divided into 6 tetrahedra based on the relative
    ordering of the fractional parts (fr, fg, fb).  Only 4 vertices
    contribute, making the interpolation sparser and often smoother.

    Args:
        rgb: (3,) array with values in [0, 1].
        N: LUT grid size per axis.

    Returns:
        (corner_indices, weights): flat indices (4,) and weights (4,) summing to 1.0.
    """
    scaled = np.clip(rgb, 0.0, 1.0) * (N - 1)
    r0 = min(int(np.floor(scaled[0])), N - 2)
    g0 = min(int(np.floor(scaled[1])), N - 2)
    b0 = min(int(np.floor(scaled[2])), N - 2)
    fr = scaled[0] - r0
    fg = scaled[1] - g0
    fb = scaled[2] - b0

    # Base (0,0,0) and opposite (1,1,1) always included
    base_r, base_g, base_b = r0, g0, b0
    opp_r, opp_g, opp_b = r0 + 1, g0 + 1, b0 + 1

    # Determine which tetrahedron by sorting fractional parts
    if fr >= fg >= fb:
        c1 = (r0 + 1, g0, b0)
        c2 = (r0 + 1, g0 + 1, b0)
        w = np.array([1 - fr, fr - fg, fg - fb, fb])
    elif fr >= fb >= fg:
        c1 = (r0 + 1, g0, b0)
        c2 = (r0 + 1, g0, b0 + 1)
        w = np.array([1 - fr, fr - fb, fb - fg, fg])
    elif fg >= fr >= fb:
        c1 = (r0, g0 + 1, b0)
        c2 = (r0 + 1, g0 + 1, b0)
        w = np.array([1 - fg, fg - fr, fr - fb, fb])
    elif fg >= fb >= fr:
        c1 = (r0, g0 + 1, b0)
        c2 = (r0, g0 + 1, b0 + 1)
        w = np.array([1 - fg, fg - fb, fb - fr, fr])
    elif fb >= fr >= fg:
        c1 = (r0, g0, b0 + 1)
        c2 = (r0 + 1, g0, b0 + 1)
        w = np.array([1 - fb, fb - fr, fr - fg, fg])
    else:  # fb >= fg >= fr
        c1 = (r0, g0, b0 + 1)
        c2 = (r0, g0 + 1, b0 + 1)
        w = np.array([1 - fb, fb - fg, fg - fr, fr])

    r_arr = np.array([base_r, c1[0], c2[0], opp_r])
    g_arr = np.array([base_g, c1[1], c2[1], opp_g])
    b_arr = np.array([base_b, c1[2], c2[2], opp_b])
    flat = flat_index_array(r_arr, g_arr, b_arr, N)
    return flat, w


# ---------------------------------------------------------------------------
# Vectorized interpolation (batch of M points)
# ---------------------------------------------------------------------------

def vectorized_trilinear(
    rgb: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute trilinear weights for M points simultaneously.

    Args:
        rgb: (M, 3) array with values in [0, 1].
        N: LUT grid size.

    Returns:
        (indices, weights): both (M, 8).
    """
    M = rgb.shape[0]
    scaled = np.clip(rgb, 0.0, 1.0) * (N - 1)
    floor = np.floor(scaled).astype(np.int32)
    floor = np.clip(floor, 0, N - 2)
    frac = scaled - floor  # (M, 3)

    r0, g0, b0 = floor[:, 0], floor[:, 1], floor[:, 2]
    fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]

    # 8 corners: enumerate (dr, dg, db) in {0,1}^3
    indices = np.empty((M, 8), dtype=np.int64)
    weights = np.empty((M, 8), dtype=np.float64)

    cfr = 1.0 - fr
    cfg = 1.0 - fg
    cfb = 1.0 - fb

    idx = 0
    for dr in (0, 1):
        wr = fr if dr else cfr
        ri = r0 + dr
        for dg in (0, 1):
            wg = fg if dg else cfg
            gi = g0 + dg
            for db in (0, 1):
                wb = fb if db else cfb
                bi = b0 + db
                indices[:, idx] = flat_index_array(ri, gi, bi, N)
                weights[:, idx] = wr * wg * wb
                idx += 1

    return indices, weights


def vectorized_tetrahedral(
    rgb: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute tetrahedral weights for M points simultaneously.

    Uses boolean masking to handle all 6 tetrahedra without Python loops
    over individual points.

    Args:
        rgb: (M, 3) array with values in [0, 1].
        N: LUT grid size.

    Returns:
        (indices, weights): both (M, 4).
    """
    M = rgb.shape[0]
    scaled = np.clip(rgb, 0.0, 1.0) * (N - 1)
    floor = np.floor(scaled).astype(np.int32)
    floor = np.clip(floor, 0, N - 2)
    frac = scaled - floor  # (M, 3)

    r0, g0, b0 = floor[:, 0], floor[:, 1], floor[:, 2]
    fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]

    # Output arrays
    corner_r = np.empty((M, 4), dtype=np.int32)
    corner_g = np.empty((M, 4), dtype=np.int32)
    corner_b = np.empty((M, 4), dtype=np.int32)
    w = np.empty((M, 4), dtype=np.float64)

    # Vertex 0 (base) and vertex 3 (opposite) are always the same
    corner_r[:, 0] = r0
    corner_g[:, 0] = g0
    corner_b[:, 0] = b0
    corner_r[:, 3] = r0 + 1
    corner_g[:, 3] = g0 + 1
    corner_b[:, 3] = b0 + 1

    # 6 cases based on ordering of (fr, fg, fb)
    # Case 1: fr >= fg >= fb
    m1 = (fr >= fg) & (fg >= fb)
    corner_r[m1, 1] = r0[m1] + 1; corner_g[m1, 1] = g0[m1];     corner_b[m1, 1] = b0[m1]
    corner_r[m1, 2] = r0[m1] + 1; corner_g[m1, 2] = g0[m1] + 1; corner_b[m1, 2] = b0[m1]
    w[m1, 0] = 1 - fr[m1]; w[m1, 1] = fr[m1] - fg[m1]
    w[m1, 2] = fg[m1] - fb[m1]; w[m1, 3] = fb[m1]

    # Case 2: fr >= fb >= fg  (and NOT case 1, i.e. fb > fg)
    m2 = (fr >= fb) & (fb > fg)
    corner_r[m2, 1] = r0[m2] + 1; corner_g[m2, 1] = g0[m2];     corner_b[m2, 1] = b0[m2]
    corner_r[m2, 2] = r0[m2] + 1; corner_g[m2, 2] = g0[m2];     corner_b[m2, 2] = b0[m2] + 1
    w[m2, 0] = 1 - fr[m2]; w[m2, 1] = fr[m2] - fb[m2]
    w[m2, 2] = fb[m2] - fg[m2]; w[m2, 3] = fg[m2]

    # Case 3: fg > fr >= fb  (and NOT case 1)
    m3 = (fg > fr) & (fr >= fb)
    corner_r[m3, 1] = r0[m3];     corner_g[m3, 1] = g0[m3] + 1; corner_b[m3, 1] = b0[m3]
    corner_r[m3, 2] = r0[m3] + 1; corner_g[m3, 2] = g0[m3] + 1; corner_b[m3, 2] = b0[m3]
    w[m3, 0] = 1 - fg[m3]; w[m3, 1] = fg[m3] - fr[m3]
    w[m3, 2] = fr[m3] - fb[m3]; w[m3, 3] = fb[m3]

    # Case 4: fg >= fb > fr  (and fb > fr)
    m4 = (fg >= fb) & (fb > fr)
    corner_r[m4, 1] = r0[m4];     corner_g[m4, 1] = g0[m4] + 1; corner_b[m4, 1] = b0[m4]
    corner_r[m4, 2] = r0[m4];     corner_g[m4, 2] = g0[m4] + 1; corner_b[m4, 2] = b0[m4] + 1
    w[m4, 0] = 1 - fg[m4]; w[m4, 1] = fg[m4] - fb[m4]
    w[m4, 2] = fb[m4] - fr[m4]; w[m4, 3] = fr[m4]

    # Case 5: fb > fr >= fg  (and fb > fr, so not case 2)
    m5 = (fb > fr) & (fr >= fg)
    corner_r[m5, 1] = r0[m5];     corner_g[m5, 1] = g0[m5];     corner_b[m5, 1] = b0[m5] + 1
    corner_r[m5, 2] = r0[m5] + 1; corner_g[m5, 2] = g0[m5];     corner_b[m5, 2] = b0[m5] + 1
    w[m5, 0] = 1 - fb[m5]; w[m5, 1] = fb[m5] - fr[m5]
    w[m5, 2] = fr[m5] - fg[m5]; w[m5, 3] = fg[m5]

    # Case 6: fb > fg > fr  (and NOT case 5, i.e. fg > fr)
    m6 = (fb > fg) & (fg > fr)
    corner_r[m6, 1] = r0[m6];     corner_g[m6, 1] = g0[m6];     corner_b[m6, 1] = b0[m6] + 1
    corner_r[m6, 2] = r0[m6];     corner_g[m6, 2] = g0[m6] + 1; corner_b[m6, 2] = b0[m6] + 1
    w[m6, 0] = 1 - fb[m6]; w[m6, 1] = fb[m6] - fg[m6]
    w[m6, 2] = fg[m6] - fr[m6]; w[m6, 3] = fr[m6]

    indices = flat_index_array(corner_r, corner_g, corner_b, N)
    return indices, w


# ---------------------------------------------------------------------------
# LUT application
# ---------------------------------------------------------------------------

def apply_lut_to_colors(
    colors: np.ndarray,
    lut: np.ndarray,
    N: int,
    kernel: str = "tetrahedral",
) -> np.ndarray:
    """Apply a 3D LUT to an array of colors.

    Args:
        colors: (M, 3) array of RGB values in [0, 1].
        lut: (N, N, N, 3) LUT array indexed as [r, g, b, ch].
        N: LUT grid size.
        kernel: "trilinear" or "tetrahedral".

    Returns:
        (M, 3) array of transformed colors.
    """
    # Flatten LUT for indexed access (match flat index convention):
    # flat = b*N*N + g*N + r, so transpose to (b, g, r, ch) before reshape.
    lut_flat = np.transpose(lut, (2, 1, 0, 3)).reshape(-1, 3)

    if kernel == "trilinear":
        indices, weights = vectorized_trilinear(colors, N)
    elif kernel == "tetrahedral":
        indices, weights = vectorized_tetrahedral(colors, N)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    # Gather LUT values at corner indices: (M, K, 3)
    K = indices.shape[1]  # 8 for trilinear, 4 for tetrahedral
    gathered = lut_flat[indices]  # (M, K, 3)

    # Weighted sum: (M, 3)
    result = np.einsum("mk,mkc->mc", weights, gathered)
    return result.astype(np.float32)
