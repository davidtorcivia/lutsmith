"""Perceptual color difference metrics.

Provides DeltaE 2000 computation using colour-science when available,
with a built-in fallback for basic functionality.
"""

from __future__ import annotations

import numpy as np


def delta_e_2000(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    """Compute CIE Delta E 2000 between two sets of sRGB colors.

    Args:
        rgb_a: (..., 3) sRGB values in [0, 1].
        rgb_b: (..., 3) sRGB values in [0, 1].

    Returns:
        (...) array of DeltaE values.
    """
    lab_a = srgb_to_lab(rgb_a)
    lab_b = srgb_to_lab(rgb_b)

    try:
        import colour
        return colour.delta_E(lab_a, lab_b, method="CIE 2000")
    except ImportError:
        return _delta_e_76(lab_a, lab_b)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to CIELAB.

    Uses colour-science for accuracy, with a built-in fallback.

    Args:
        rgb: (..., 3) sRGB values in [0, 1].

    Returns:
        (..., 3) CIELAB values.
    """
    try:
        import colour
        rgb_clipped = np.clip(rgb, 0.0, 1.0)
        xyz = colour.sRGB_to_XYZ(rgb_clipped)
        return colour.XYZ_to_Lab(xyz)
    except ImportError:
        return _srgb_to_lab_builtin(rgb)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to Oklab perceptual color space.

    Oklab is more perceptually uniform than CIELAB and has simpler math.

    Args:
        rgb: (..., 3) sRGB values in [0, 1].

    Returns:
        (..., 3) Oklab values (L, a, b).
    """
    # Linearize sRGB
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((np.clip(rgb, 0, None) + 0.055) / 1.055) ** 2.4,
    )

    # Linear sRGB -> LMS (Oklab matrix)
    l = 0.4122214708 * linear[..., 0] + 0.5363325363 * linear[..., 1] + 0.0514459929 * linear[..., 2]
    m = 0.2119034982 * linear[..., 0] + 0.6806995451 * linear[..., 1] + 0.1073969566 * linear[..., 2]
    s = 0.0883024619 * linear[..., 0] + 0.2817188376 * linear[..., 1] + 0.6299787005 * linear[..., 2]

    # Cube root
    l_ = np.cbrt(np.maximum(l, 0.0))
    m_ = np.cbrt(np.maximum(m, 0.0))
    s_ = np.cbrt(np.maximum(s, 0.0))

    # LMS' -> Oklab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_val = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, a, b_val], axis=-1)


def _srgb_to_lab_builtin(rgb: np.ndarray) -> np.ndarray:
    """Built-in sRGB -> CIELAB conversion (no dependencies)."""
    # Linearize sRGB
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((np.clip(rgb, 0, None) + 0.055) / 1.055) ** 2.4,
    )

    # sRGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ M.T

    # XYZ -> Lab (D65 white point)
    wp = np.array([0.95047, 1.00000, 1.08883])
    xyz_n = xyz / wp

    delta = 6.0 / 29.0
    fx = np.where(xyz_n[..., 0] > delta**3,
                  np.cbrt(np.maximum(xyz_n[..., 0], 1e-10)),
                  xyz_n[..., 0] / (3 * delta**2) + 4.0 / 29.0)
    fy = np.where(xyz_n[..., 1] > delta**3,
                  np.cbrt(np.maximum(xyz_n[..., 1], 1e-10)),
                  xyz_n[..., 1] / (3 * delta**2) + 4.0 / 29.0)
    fz = np.where(xyz_n[..., 2] > delta**3,
                  np.cbrt(np.maximum(xyz_n[..., 2], 1e-10)),
                  xyz_n[..., 2] / (3 * delta**2) + 4.0 / 29.0)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def _delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 DeltaE (Euclidean distance in Lab). Fallback."""
    diff = lab1 - lab2
    return np.sqrt(np.sum(diff ** 2, axis=-1))
