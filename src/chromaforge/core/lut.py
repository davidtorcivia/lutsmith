"""LUT data structure, identity generation, application, and utilities.

All LUTs use the convention: shape (N, N, N, 3), indexed as lut[r, g, b, ch].
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from scipy.ndimage import gaussian_filter

from chromaforge.config import LUT_CLAMP_MIN, LUT_CLAMP_MAX
from chromaforge.core.interpolation import apply_lut_to_colors
from chromaforge.core.types import LUTData, InterpolationKernel


def identity_lut(N: int) -> np.ndarray:
    """Generate an identity 3D LUT.

    Each node maps to its own normalized coordinate:
    lut[r, g, b] = (r/(N-1), g/(N-1), b/(N-1)).

    Args:
        N: Grid size per axis.

    Returns:
        (N, N, N, 3) float32 array.
    """
    coords = np.linspace(0.0, 1.0, N, dtype=np.float32)
    r, g, b = np.meshgrid(coords, coords, coords, indexing="ij")
    lut = np.stack([r, g, b], axis=-1)
    return lut


def identity_lut_flat(N: int) -> np.ndarray:
    """Generate identity LUT values as a flat (N^3, 3) array.

    Flat index convention: flat = b*N*N + g*N + r (R fastest).

    Returns:
        (N^3, 3) float32 array where row i contains the RGB identity
        value for flat index i.
    """
    lut = identity_lut(N)
    # Ravel in memory order: since lut is (N, N, N, 3) with [r, g, b, ch],
    # and our flat convention is b*N*N + g*N + r, we need to transpose
    # to (b, g, r, 3) before flattening.
    lut_bgr = np.transpose(lut, (2, 1, 0, 3))  # (N, N, N, 3) -> [b, g, r, ch]
    return lut_bgr.reshape(-1, 3).copy()


def apply_lut(
    image: np.ndarray,
    lut: np.ndarray,
    N: int,
    kernel: str = "tetrahedral",
) -> np.ndarray:
    """Apply a 3D LUT to an image.

    Args:
        image: (H, W, 3) or (M, 3) float32 array in [0, 1].
        lut: (N, N, N, 3) LUT array.
        N: Grid size.
        kernel: Interpolation kernel name.

    Returns:
        Transformed image with same shape as input.
    """
    original_shape = image.shape
    if image.ndim == 3 and len(original_shape) == 3 and original_shape[2] == 3:
        flat = image.reshape(-1, 3)
    elif image.ndim == 2 and original_shape[1] == 3:
        flat = image
    else:
        raise ValueError(f"Expected (H,W,3) or (M,3), got {original_shape}")

    result = apply_lut_to_colors(flat, lut, N, kernel)
    return result.reshape(original_shape)


def clip_lut(
    lut: np.ndarray,
    lo: float = LUT_CLAMP_MIN,
    hi: float = LUT_CLAMP_MAX,
) -> np.ndarray:
    """Clamp LUT values to a safe range.

    Args:
        lut: (N, N, N, 3) array.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clipped copy.
    """
    return np.clip(lut, lo, hi)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep: 0 at x<=0, 1 at x>=1, smooth in between."""
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def estimate_shadow_thresholds(
    input_rgb: np.ndarray,
    base_shadow: float = 0.25,
    base_deep: float = 0.08,
    shadow_percentile: float = 30.0,
    deep_percentile: float = 10.0,
    min_gap: float = 0.02,
    max_shadow: float = 0.6,
) -> tuple[float, float, dict]:
    """Estimate shadow smoothing thresholds from input samples.

    Uses luminance percentiles to adapt thresholds for low-contrast
    or log-like inputs, while preserving default behavior otherwise.

    Returns:
        (shadow_threshold, deep_threshold, info)
    """
    if input_rgb is None or input_rgb.size == 0:
        return base_shadow, base_deep, {
            "auto_applied": False,
            "reason": "no_samples",
        }

    lum = (
        0.2126 * input_rgb[..., 0]
        + 0.7152 * input_rgb[..., 1]
        + 0.0722 * input_rgb[..., 2]
    )
    lum = lum[np.isfinite(lum)]
    if lum.size == 0:
        return base_shadow, base_deep, {
            "auto_applied": False,
            "reason": "non_finite",
        }

    p99 = float(np.percentile(lum, 99))
    mean = float(np.mean(lum))
    info = {"p99": p99, "mean": mean}

    # Heuristic: compressed/log-like inputs have low highlights and mid-heavy means.
    compressed = (p99 < 0.9) and (0.05 < mean < 0.6)
    if not compressed:
        info["auto_applied"] = False
        return base_shadow, base_deep, info

    p_deep = float(np.percentile(lum, deep_percentile))
    p_shadow = float(np.percentile(lum, shadow_percentile))

    deep = max(base_deep, p_deep)
    shadow = max(base_shadow, p_shadow)

    shadow = min(shadow, max_shadow)
    if shadow <= deep + min_gap:
        shadow = min(max_shadow, deep + min_gap)
    deep = min(deep, shadow - min_gap)

    deep = float(np.clip(deep, 0.0, 1.0))
    shadow = float(np.clip(shadow, 0.0, 1.0))

    info.update({
        "auto_applied": True,
        "p_deep": p_deep,
        "p_shadow": p_shadow,
        "shadow_percentile": shadow_percentile,
        "deep_percentile": deep_percentile,
    })

    return shadow, deep, info


def smooth_lut_shadows(
    lut: np.ndarray,
    N: int,
    shadow_threshold: float = 0.25,
    deep_threshold: float = 0.08,
) -> np.ndarray:
    """Apply graduated 3D spatial smoothing to the shadow region of a LUT.

    Uses a two-tier approach: heavy blur for deep shadows (where solver
    data is sparsest and noisiest), moderate blur for mid-shadows, and
    no smoothing above the shadow threshold.

    The deep-shadow blur is applied as two consecutive passes (effective
    sigma ~ sigma * sqrt(2)) to reach further without an enormous kernel,
    suppressing per-node noise that causes RGB channel divergence on
    waveform monitors.

    Args:
        lut: (N, N, N, 3) LUT array.
        N: Grid size per axis.
        shadow_threshold: Upper luminance boundary of the shadow region.
            Nodes above this are not smoothed.
        deep_threshold: Luminance boundary between deep and mid shadows.
            Below this, the strongest blur applies.

    Returns:
        (N, N, N, 3) LUT with smoothed shadows.
    """
    # Deep-shadow blur: two passes of sigma=3.0 (effective ~4.2)
    sigma_deep = 3.0
    # Mid-shadow blur: single pass
    sigma_mid = 1.5

    blurred_deep = np.empty_like(lut)
    blurred_mid = np.empty_like(lut)
    for ch in range(3):
        # Two-pass blur for deep shadows — stronger smoothing
        first = gaussian_filter(lut[..., ch], sigma=sigma_deep)
        blurred_deep[..., ch] = gaussian_filter(first, sigma=sigma_deep)
        blurred_mid[..., ch] = gaussian_filter(lut[..., ch], sigma=sigma_mid)

    # Input-space luminance grid (stable regardless of the grade)
    coords = np.linspace(0.0, 1.0, N, dtype=np.float32)
    ri, gi, bi = np.meshgrid(coords, coords, coords, indexing="ij")
    luminance = 0.2126 * ri + 0.7152 * gi + 0.0722 * bi

    # Tier 1 blend: deep vs mid smoothed
    # 0 = deep (strong blur), 1 = mid (moderate blur)
    t_deep = _smoothstep(luminance / max(deep_threshold, 1e-8))
    smoothed = (blurred_deep * (1.0 - t_deep[..., np.newaxis])
                + blurred_mid * t_deep[..., np.newaxis])

    # Tier 2 blend: smoothed vs original
    # 0 = shadow (use smoothed), 1 = bright (keep original)
    t_shadow = _smoothstep(
        (luminance - deep_threshold)
        / max(shadow_threshold - deep_threshold, 1e-8)
    )
    result = (smoothed * (1.0 - t_shadow[..., np.newaxis])
              + lut * t_shadow[..., np.newaxis])

    return result.astype(np.float32)


def lut_stats(lut: np.ndarray) -> dict:
    """Compute basic statistics of a LUT.

    Returns:
        Dict with min, max, mean per channel and overall.
    """
    return {
        "min_per_channel": lut.min(axis=(0, 1, 2)).tolist(),
        "max_per_channel": lut.max(axis=(0, 1, 2)).tolist(),
        "mean_per_channel": lut.mean(axis=(0, 1, 2)).tolist(),
        "global_min": float(lut.min()),
        "global_max": float(lut.max()),
    }


def lut_total_variation(lut: np.ndarray) -> float:
    """Compute normalized total variation of a LUT.

    Sum of absolute differences between adjacent nodes along each axis,
    normalized by the total number of node-channel pairs.
    """
    N = lut.shape[0]
    tv_r = np.sum(np.abs(np.diff(lut, axis=0)))
    tv_g = np.sum(np.abs(np.diff(lut, axis=1)))
    tv_b = np.sum(np.abs(np.diff(lut, axis=2)))
    return float((tv_r + tv_g + tv_b) / (3 * N ** 3))


def lut_neutral_monotonicity(lut: np.ndarray) -> tuple[bool, int]:
    """Check monotonicity along the neutral (gray) axis.

    The diagonal lut[i, i, i] should be monotonically increasing
    in all channels for a well-behaved LUT.

    Returns:
        (is_monotonic, num_violations)
    """
    N = lut.shape[0]
    neutral = np.array([lut[i, i, i] for i in range(N)])
    violations = 0
    for i in range(1, N):
        if np.any(neutral[i] < neutral[i - 1]):
            violations += 1
    return violations == 0, violations


def lut_oog_percentage(lut: np.ndarray) -> float:
    """Compute percentage of LUT values outside [0, 1]."""
    N = lut.shape[0]
    total_values = N ** 3 * 3
    oog = int(np.sum((lut < 0.0) | (lut > 1.0)))
    return float(oog / total_values * 100.0)


def create_lut_data(
    array: np.ndarray,
    kernel: InterpolationKernel = InterpolationKernel.TETRAHEDRAL,
    title: str = "ChromaForge LUT",
) -> LUTData:
    """Create a LUTData object from a raw array.

    Args:
        array: (N, N, N, 3) float32 array.
        kernel: Interpolation kernel used during fitting.
        title: Human-readable title.

    Returns:
        LUTData instance.
    """
    N = array.shape[0]
    return LUTData(
        array=array.astype(np.float32),
        size=N,
        title=title,
        kernel=kernel,
    )
