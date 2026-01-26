"""Color space transfer function definitions.

Provides forward/inverse transfer functions for common camera encodings.
Uses OCIO when available for maximum accuracy.
"""

from __future__ import annotations

import numpy as np

from chromaforge.core.types import TransferFunction


# ---- ARRI LogC3 (Alexa Classic) ----

def logc3_to_linear(x: np.ndarray) -> np.ndarray:
    """ARRI LogC3 (EI 800) to scene-linear."""
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809

    return np.where(
        x > e * cut + f,
        (np.power(10.0, (x - d) / c) - b) / a,
        (x - f) / e,
    ).astype(np.float32)


def linear_to_logc3(x: np.ndarray) -> np.ndarray:
    """Scene-linear to ARRI LogC3 (EI 800)."""
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809

    return np.where(
        x > cut,
        c * np.log10(np.maximum(a * x + b, 1e-10)) + d,
        e * x + f,
    ).astype(np.float32)


# ---- ARRI LogC4 (Alexa 35) ----

def logc4_to_linear(x: np.ndarray) -> np.ndarray:
    """ARRI LogC4 to scene-linear."""
    a = 2231.82630906905
    b = 64.0
    c = 0.0740718950408889
    s = 7.0
    t = 1.0

    return np.where(
        x >= 0.0,
        (np.power(2.0, (x - t) * s) - b) / a,
        x * c,
    ).astype(np.float32)


def linear_to_logc4(x: np.ndarray) -> np.ndarray:
    """Scene-linear to ARRI LogC4."""
    a = 2231.82630906905
    b = 64.0
    c = 0.0740718950408889
    s = 7.0
    t = 1.0

    cut = (1.0 - b) / a

    return np.where(
        x >= cut,
        (np.log2(np.maximum(a * x + b, 1e-10)) / s) + t,
        x / c,
    ).astype(np.float32)


# ---- Sony S-Log3 ----

def slog3_to_linear(x: np.ndarray) -> np.ndarray:
    """Sony S-Log3 to scene-linear."""
    return np.where(
        x >= 171.2102946929 / 1023.0,
        np.power(10.0, ((x * 1023.0 - 420.0) / 261.5)) * (0.18 + 0.01) - 0.01,
        (x * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0),
    ).astype(np.float32)


def linear_to_slog3(x: np.ndarray) -> np.ndarray:
    """Scene-linear to Sony S-Log3."""
    return np.where(
        x >= 0.01125000,
        (420.0 + np.log10(np.maximum((x + 0.01) / (0.18 + 0.01), 1e-10)) * 261.5) / 1023.0,
        (x * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0,
    ).astype(np.float32)


# ---- Panasonic V-Log ----

def vlog_to_linear(x: np.ndarray) -> np.ndarray:
    """Panasonic V-Log to scene-linear."""
    cut_inv = 0.181
    b = 0.00873
    c = 0.241514
    d = 0.598206

    return np.where(
        x < cut_inv,
        (x - 0.125) / 5.6,
        np.power(10.0, (x - d) / c) - b,
    ).astype(np.float32)


def linear_to_vlog(x: np.ndarray) -> np.ndarray:
    """Scene-linear to Panasonic V-Log."""
    cut = 0.01
    b = 0.00873
    c = 0.241514
    d = 0.598206

    return np.where(
        x < cut,
        5.6 * x + 0.125,
        c * np.log10(np.maximum(x + b, 1e-10)) + d,
    ).astype(np.float32)


# ---- Lookup table for transfer functions ----

TRANSFER_FUNCTIONS = {
    TransferFunction.LOG_C3: (logc3_to_linear, linear_to_logc3),
    TransferFunction.LOG_C4: (logc4_to_linear, linear_to_logc4),
    TransferFunction.SLOG3: (slog3_to_linear, linear_to_slog3),
    TransferFunction.VLOG: (vlog_to_linear, linear_to_vlog),
}


def get_transfer_functions(tf: TransferFunction):
    """Get (to_linear, from_linear) functions for a transfer function.

    Returns:
        (to_linear, from_linear) callables, or (None, None) for LINEAR/UNKNOWN.
    """
    return TRANSFER_FUNCTIONS.get(tf, (None, None))
