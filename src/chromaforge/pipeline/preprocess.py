"""Image preprocessing: loading, sanitization, and transfer function detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from chromaforge.core.types import TransferFunction, PipelineConfig
from chromaforge.color.shaper import (
    get_shaper,
    identity_forward,
    identity_inverse,
)
from chromaforge.io.image import load_image

logger = logging.getLogger(__name__)


def sanitize_image(img: np.ndarray) -> np.ndarray:
    """Remove NaN/Inf values that would poison the solver.

    A single NaN in the sparse matrix propagates through the solver
    and corrupts the entire LUT.

    Args:
        img: (H, W, 3) float32 image.

    Returns:
        Sanitized copy with NaN -> 0, Inf -> 1, -Inf -> -1.
        Preserves finite negative values to keep full-range color.
    """
    result = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
    return result.astype(np.float32)


def _resize_to_match(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize images to common dimensions using the minimum of each axis.

    Uses area-based interpolation (antialiased downscale) to preserve color
    accuracy, which matters for LUT extraction.

    Args:
        source: (H1, W1, 3) float32 image.
        target: (H2, W2, 3) float32 image.

    Returns:
        (source, target) resized to (min_H, min_W, 3).
    """
    from scipy.ndimage import zoom

    h = min(source.shape[0], target.shape[0])
    w = min(source.shape[1], target.shape[1])

    def _resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if img.shape[0] == target_h and img.shape[1] == target_w:
            return img
        zoom_h = target_h / img.shape[0]
        zoom_w = target_w / img.shape[1]
        return zoom(img, (zoom_h, zoom_w, 1.0), order=1).astype(np.float32)

    return _resize(source, h, w), _resize(target, h, w)


def detect_transfer_function(
    img: np.ndarray,
    metadata: Optional[dict] = None,
) -> TransferFunction:
    """Auto-detect the transfer function of an image.

    Uses heuristics based on value distribution and metadata.

    Args:
        img: (H, W, 3) float32 image.
        metadata: Optional metadata from image loader.

    Returns:
        Detected TransferFunction.
    """
    # Check metadata for hints
    if metadata:
        fmt = metadata.get("format", "").lower()
        if "exr" in fmt or "float" in fmt:
            # EXR files are typically scene-linear
            max_val = float(np.max(img))
            if max_val > 2.0:
                return TransferFunction.LINEAR

    # Heuristic: examine value distribution
    flat = img.ravel()
    max_val = float(np.max(flat))
    mean_val = float(np.mean(flat))
    p99 = float(np.percentile(flat[flat > 0], 99)) if np.any(flat > 0) else 0

    # Scene-linear typically has values >> 1.0 in highlights
    if max_val > 2.0:
        return TransferFunction.LINEAR

    # Log-encoded footage: most values in 0.1-0.7 range
    if 0.05 < mean_val < 0.55 and p99 < 0.85:
        # Could be LogC3, LogC4, S-Log3, or V-Log
        # Without more info, return UNKNOWN
        return TransferFunction.UNKNOWN

    # Standard 8/16-bit imagery
    return TransferFunction.UNKNOWN


def _resolve_shaper(
    transfer_function: TransferFunction,
    include_shaper: Optional[bool],
) -> tuple[callable, callable, bool, str]:
    """Resolve shaper functions and whether to apply them.

    Returns:
        (forward, inverse, applied, mode)
    """
    if include_shaper is False:
        return identity_forward, identity_inverse, False, "disabled"

    forward, inverse = get_shaper(transfer_function)
    applied = forward is not identity_forward
    mode = "forced" if include_shaper is True else "auto"
    return forward, inverse, applied, mode


def preprocess_pair(
    source_path: Path | str,
    target_path: Path | str,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Load and preprocess an image pair for LUT extraction.

    Args:
        source_path: Path to the source (ungraded) image.
        target_path: Path to the target (graded) image.
        config: Pipeline configuration.

    Returns:
        (source, target, metadata, shaper): Sanitized float32 images, metadata,
        and shaper info dict with forward/inverse functions.
    """
    logger.info("Loading source image: %s", source_path)
    source, source_meta = load_image(source_path)

    logger.info("Loading target image: %s", target_path)
    target, target_meta = load_image(target_path)

    # Resize to matching dimensions if needed
    if source.shape[:2] != target.shape[:2]:
        source, target = _resize_to_match(source, target)
        logger.info(
            "Resized images to common dimensions: %dx%d",
            source.shape[1], source.shape[0],
        )

    # Sanitize
    logger.info("Sanitizing images...")
    source = sanitize_image(source)
    target = sanitize_image(target)

    # Detect transfer function if auto
    tf = config.transfer_function
    if tf == TransferFunction.AUTO:
        tf = detect_transfer_function(source, source_meta)
        logger.info("Auto-detected transfer function: %s", tf.value)

    # Resolve and apply shaper (input only)
    shaper_forward, shaper_inverse, shaper_applied, shaper_mode = _resolve_shaper(
        tf, config.include_shaper
    )
    if shaper_applied:
        logger.info("Applying shaper (%s) to source input", shaper_mode)
        source = shaper_forward(source)

    metadata = {
        "source_meta": source_meta,
        "target_meta": target_meta,
        "transfer_function": tf,
        "width": source.shape[1],
        "height": source.shape[0],
        "total_pixels": source.shape[0] * source.shape[1],
        "shaper_applied": shaper_applied,
        "shaper_mode": shaper_mode,
        "shaper_name": "log2" if shaper_applied else "identity",
    }

    shaper = {
        "forward": shaper_forward,
        "inverse": shaper_inverse,
        "applied": shaper_applied,
        "mode": shaper_mode,
    }

    return source, target, metadata, shaper
