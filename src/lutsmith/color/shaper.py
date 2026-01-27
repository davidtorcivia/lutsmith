"""Shaper LUT generation for HDR/Log footage.

A shaper (1D LUT) redistributes the input domain to better utilize
the 3D LUT grid resolution. Critical for scene-linear inputs where
most data is concentrated in a narrow range.

Key rule: Do NOT apply a log shaper to already-log footage (double-logging).
"""

from __future__ import annotations

import logging

import numpy as np

from lutsmith.core.types import TransferFunction

logger = logging.getLogger(__name__)


def get_shaper(
    transfer_function: TransferFunction,
    histogram: np.ndarray | None = None,
) -> tuple[callable, callable]:
    """Get appropriate shaper functions for the input encoding.

    Args:
        transfer_function: Detected input encoding.
        histogram: Optional value histogram for adaptive shaper.

    Returns:
        (forward, inverse): Shaper and its inverse as callables.
    """
    if transfer_function == TransferFunction.LINEAR:
        logger.info("Scene-linear input: applying log shaper")
        return log_shaper_forward, log_shaper_inverse
    elif transfer_function in (
        TransferFunction.LOG_C3,
        TransferFunction.LOG_C4,
        TransferFunction.SLOG3,
        TransferFunction.VLOG,
    ):
        logger.info("Log-encoded input (%s): using identity shaper", transfer_function.value)
        return identity_forward, identity_inverse
    else:
        logger.info("Unknown encoding: using conservative shaper")
        return identity_forward, identity_inverse


def identity_forward(x: np.ndarray) -> np.ndarray:
    """Identity shaper (no transform)."""
    return x.copy()


def identity_inverse(x: np.ndarray) -> np.ndarray:
    """Identity shaper inverse."""
    return x.copy()


def log_shaper_forward(x: np.ndarray, mid_gray: float = 0.18, stops: float = 10.0) -> np.ndarray:
    """Log2-based shaper for scene-linear data.

    Maps scene-linear values to [0, 1] using a log2 curve centered
    around mid-gray. Allocates equal grid resolution per stop.

    Args:
        x: Scene-linear values (can be > 1.0).
        mid_gray: Scene-linear mid-gray value (default 0.18).
        stops: Number of stops to encode above and below mid-gray.

    Returns:
        Shaped values in [0, 1].
    """
    # log2(x / mid_gray) gives stops relative to mid-gray
    # Normalize to [0, 1] range: 0 = -stops, 0.5 = mid_gray, 1 = +stops
    log_val = np.log2(np.maximum(x, 1e-10) / mid_gray)
    normalized = (log_val + stops) / (2.0 * stops)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def log_shaper_inverse(y: np.ndarray, mid_gray: float = 0.18, stops: float = 10.0) -> np.ndarray:
    """Inverse of log_shaper_forward."""
    log_val = y * (2.0 * stops) - stops
    return (mid_gray * np.power(2.0, log_val)).astype(np.float32)


def enforce_monotonic(values: np.ndarray) -> np.ndarray:
    """Enforce strict monotonicity on a 1D shaper curve.

    ARRI cameras reject non-monotonic shapers. Uses isotonic regression
    if scikit-learn is available, otherwise a simple clamping approach.

    Args:
        values: 1D array of shaper values.

    Returns:
        Monotonically increasing version of the input.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(increasing=True)
        x = np.arange(len(values), dtype=np.float64)
        result = ir.fit_transform(x, values.astype(np.float64))
        return result.astype(np.float32)
    except ImportError:
        # Fallback: simple forward pass ensuring monotonicity
        result = values.copy()
        for i in range(1, len(result)):
            if result[i] <= result[i - 1]:
                result[i] = result[i - 1] + 1e-7
        return result


def generate_1d_shaper(
    forward: callable,
    size: int = 4096,
) -> np.ndarray:
    """Generate a 1D shaper LUT from a forward function.

    Args:
        forward: Shaper forward function.
        size: Number of entries in the 1D LUT.

    Returns:
        (size, 3) array for the 1D shaper (same value per channel).
    """
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y = forward(x)
    y = enforce_monotonic(y)
    return np.stack([y, y, y], axis=-1)
