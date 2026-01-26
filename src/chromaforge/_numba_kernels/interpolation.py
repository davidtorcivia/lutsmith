"""JIT-compiled interpolation kernels for fast LUT application.

Used primarily in validation where millions of pixels need to be
interpolated through the LUT.
"""

from __future__ import annotations

import numpy as np

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:

    @nb.njit(cache=True)
    def _apply_lut_trilinear_numba(
        colors: np.ndarray,
        lut: np.ndarray,
        N: int,
    ) -> np.ndarray:
        """Apply LUT via trilinear interpolation (Numba JIT).

        Args:
            colors: (M, 3) float32 in [0, 1].
            lut: (N, N, N, 3) float32 LUT indexed as [r, g, b, ch].
            N: Grid size.

        Returns:
            (M, 3) interpolated output colors.
        """
        M = colors.shape[0]
        out = np.empty((M, 3), dtype=np.float32)
        Nf = np.float32(N - 1)

        for i in range(M):
            # Scale to grid
            rf = colors[i, 0] * Nf
            gf = colors[i, 1] * Nf
            bf = colors[i, 2] * Nf

            # Clamp
            rf = max(np.float32(0.0), min(rf, Nf))
            gf = max(np.float32(0.0), min(gf, Nf))
            bf = max(np.float32(0.0), min(bf, Nf))

            # Integer + fractional parts
            r0 = int(rf)
            g0 = int(gf)
            b0 = int(bf)

            r1 = min(r0 + 1, N - 1)
            g1 = min(g0 + 1, N - 1)
            b1 = min(b0 + 1, N - 1)

            dr = rf - r0
            dg = gf - g0
            db = bf - b0

            # Trilinear weights (8 corners)
            w000 = (1 - dr) * (1 - dg) * (1 - db)
            w100 = dr * (1 - dg) * (1 - db)
            w010 = (1 - dr) * dg * (1 - db)
            w110 = dr * dg * (1 - db)
            w001 = (1 - dr) * (1 - dg) * db
            w101 = dr * (1 - dg) * db
            w011 = (1 - dr) * dg * db
            w111 = dr * dg * db

            for ch in range(3):
                out[i, ch] = (
                    w000 * lut[r0, g0, b0, ch]
                    + w100 * lut[r1, g0, b0, ch]
                    + w010 * lut[r0, g1, b0, ch]
                    + w110 * lut[r1, g1, b0, ch]
                    + w001 * lut[r0, g0, b1, ch]
                    + w101 * lut[r1, g0, b1, ch]
                    + w011 * lut[r0, g1, b1, ch]
                    + w111 * lut[r1, g1, b1, ch]
                )

        return out

    @nb.njit(cache=True)
    def _apply_lut_tetrahedral_numba(
        colors: np.ndarray,
        lut: np.ndarray,
        N: int,
    ) -> np.ndarray:
        """Apply LUT via tetrahedral interpolation (Numba JIT).

        Args:
            colors: (M, 3) float32 in [0, 1].
            lut: (N, N, N, 3) float32 LUT indexed as [r, g, b, ch].
            N: Grid size.

        Returns:
            (M, 3) interpolated output colors.
        """
        M = colors.shape[0]
        out = np.empty((M, 3), dtype=np.float32)
        Nf = np.float32(N - 1)

        for i in range(M):
            rf = colors[i, 0] * Nf
            gf = colors[i, 1] * Nf
            bf = colors[i, 2] * Nf

            rf = max(np.float32(0.0), min(rf, Nf))
            gf = max(np.float32(0.0), min(gf, Nf))
            bf = max(np.float32(0.0), min(bf, Nf))

            r0 = int(rf)
            g0 = int(gf)
            b0 = int(bf)

            r1 = min(r0 + 1, N - 1)
            g1 = min(g0 + 1, N - 1)
            b1 = min(b0 + 1, N - 1)

            dr = rf - r0
            dg = gf - g0
            db = bf - b0

            # Determine tetrahedron by comparing fractional parts
            for ch in range(3):
                c0 = lut[r0, g0, b0, ch]

                if dr >= dg and dg >= db:
                    # T1: r >= g >= b
                    out[i, ch] = (
                        c0
                        + dr * (lut[r1, g0, b0, ch] - c0)
                        + dg * (lut[r1, g1, b0, ch] - lut[r1, g0, b0, ch])
                        + db * (lut[r1, g1, b1, ch] - lut[r1, g1, b0, ch])
                    )
                elif dr >= db and db >= dg:
                    # T2: r >= b >= g
                    out[i, ch] = (
                        c0
                        + dr * (lut[r1, g0, b0, ch] - c0)
                        + db * (lut[r1, g0, b1, ch] - lut[r1, g0, b0, ch])
                        + dg * (lut[r1, g1, b1, ch] - lut[r1, g0, b1, ch])
                    )
                elif dg >= dr and dr >= db:
                    # T3: g >= r >= b
                    out[i, ch] = (
                        c0
                        + dg * (lut[r0, g1, b0, ch] - c0)
                        + dr * (lut[r1, g1, b0, ch] - lut[r0, g1, b0, ch])
                        + db * (lut[r1, g1, b1, ch] - lut[r1, g1, b0, ch])
                    )
                elif dg >= db and db >= dr:
                    # T4: g >= b >= r
                    out[i, ch] = (
                        c0
                        + dg * (lut[r0, g1, b0, ch] - c0)
                        + db * (lut[r0, g1, b1, ch] - lut[r0, g1, b0, ch])
                        + dr * (lut[r1, g1, b1, ch] - lut[r0, g1, b1, ch])
                    )
                elif db >= dr and dr >= dg:
                    # T5: b >= r >= g
                    out[i, ch] = (
                        c0
                        + db * (lut[r0, g0, b1, ch] - c0)
                        + dr * (lut[r1, g0, b1, ch] - lut[r0, g0, b1, ch])
                        + dg * (lut[r1, g1, b1, ch] - lut[r1, g0, b1, ch])
                    )
                else:
                    # T6: b >= g >= r
                    out[i, ch] = (
                        c0
                        + db * (lut[r0, g0, b1, ch] - c0)
                        + dg * (lut[r0, g1, b1, ch] - lut[r0, g0, b1, ch])
                        + dr * (lut[r1, g1, b1, ch] - lut[r0, g1, b1, ch])
                    )

        return out


def apply_lut_fast(
    colors: np.ndarray,
    lut: np.ndarray,
    N: int,
    kernel: str = "tetrahedral",
) -> np.ndarray:
    """Apply LUT to colors using JIT-compiled kernel (with NumPy fallback).

    Args:
        colors: (M, 3) float32 in [0, 1].
        lut: (N, N, N, 3) float32.
        N: Grid size.
        kernel: "trilinear" or "tetrahedral".

    Returns:
        (M, 3) interpolated output.
    """
    colors = np.ascontiguousarray(colors, dtype=np.float32)
    lut = np.ascontiguousarray(lut, dtype=np.float32)

    if HAS_NUMBA:
        if kernel == "trilinear":
            return _apply_lut_trilinear_numba(colors, lut, N)
        else:
            return _apply_lut_tetrahedral_numba(colors, lut, N)
    else:
        # Fall back to pure NumPy implementation
        from chromaforge.core.interpolation import apply_lut_to_colors
        return apply_lut_to_colors(colors, lut, N, kernel)
