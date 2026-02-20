"""Per-pair normalization helpers for batch extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NORMALIZATION_MODES = frozenset({
    "none",
    "luma_affine",
    "rgb_affine",
})


@dataclass
class NormalizationDiagnostics:
    mode: str
    gains: list[float]
    biases: list[float]


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def _robust_affine(src: np.ndarray, tgt: np.ndarray) -> tuple[float, float]:
    src_lo = _pct(src, 5.0)
    src_hi = _pct(src, 95.0)
    tgt_lo = _pct(tgt, 5.0)
    tgt_hi = _pct(tgt, 95.0)
    gain = (tgt_hi - tgt_lo) / max(src_hi - src_lo, 1e-8)
    bias = tgt_lo - gain * src_lo
    return float(gain), float(bias)


def apply_pair_normalization(
    source: np.ndarray,
    target: np.ndarray,
    mode: str | None,
) -> tuple[np.ndarray, np.ndarray, NormalizationDiagnostics]:
    """Apply optional per-pair normalization to source/target arrays.

    Modes:
        none        : no-op.
        luma_affine : robust affine normalization of source luminance to target luminance.
        rgb_affine  : robust per-channel affine normalization of source to target.
    """
    requested = "none" if mode is None else mode.strip().lower()
    if requested not in NORMALIZATION_MODES:
        raise ValueError(
            f"Invalid normalization mode '{mode}'. "
            f"Use one of: {', '.join(sorted(NORMALIZATION_MODES))}"
        )

    if requested == "none":
        return source, target, NormalizationDiagnostics(mode="none", gains=[1.0, 1.0, 1.0], biases=[0.0, 0.0, 0.0])

    src = source.astype(np.float32, copy=True)
    tgt = target.astype(np.float32, copy=False)

    if requested == "luma_affine":
        src_l = 0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]
        tgt_l = 0.2126 * tgt[..., 0] + 0.7152 * tgt[..., 1] + 0.0722 * tgt[..., 2]
        g, b = _robust_affine(src_l.reshape(-1), tgt_l.reshape(-1))
        src = src * g + b
        diag = NormalizationDiagnostics(mode=requested, gains=[g, g, g], biases=[b, b, b])
        return src, tgt, diag

    gains = []
    biases = []
    for c in range(3):
        g, b = _robust_affine(src[..., c].reshape(-1), tgt[..., c].reshape(-1))
        src[..., c] = src[..., c] * g + b
        gains.append(g)
        biases.append(b)
    diag = NormalizationDiagnostics(mode=requested, gains=gains, biases=biases)
    return src, tgt, diag

