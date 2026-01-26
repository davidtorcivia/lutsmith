"""Image preprocessing: loading, sanitization, and transfer function detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from chromaforge.core.types import TransferFunction, PipelineConfig
from chromaforge.errors import ImageDimensionError, ImageError
from chromaforge.io.image import load_image

logger = logging.getLogger(__name__)


def sanitize_image(img: np.ndarray) -> np.ndarray:
    """Remove NaN/Inf values that would poison the solver.

    A single NaN in the sparse matrix propagates through the solver
    and corrupts the entire LUT.

    Args:
        img: (H, W, 3) float32 image.

    Returns:
        Sanitized copy with NaN -> 0, Inf -> 1, -Inf -> 0, negatives clamped.
    """
    result = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    result = np.clip(result, 0.0, None)  # Clamp negatives
    return result.astype(np.float32)


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


def preprocess_pair(
    source_path: Path | str,
    target_path: Path | str,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and preprocess an image pair for LUT extraction.

    Args:
        source_path: Path to the source (ungraded) image.
        target_path: Path to the target (graded) image.
        config: Pipeline configuration.

    Returns:
        (source, target, metadata): Sanitized float32 images and metadata.
    """
    logger.info("Loading source image: %s", source_path)
    source, source_meta = load_image(source_path)

    logger.info("Loading target image: %s", target_path)
    target, target_meta = load_image(target_path)

    # Validate matching dimensions
    if source.shape[:2] != target.shape[:2]:
        raise ImageDimensionError(
            f"Source {source.shape[:2]} and target {target.shape[:2]} "
            f"dimensions do not match"
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

    metadata = {
        "source_meta": source_meta,
        "target_meta": target_meta,
        "transfer_function": tf,
        "width": source.shape[1],
        "height": source.shape[0],
        "total_pixels": source.shape[0] * source.shape[1],
    }

    return source, target, metadata
