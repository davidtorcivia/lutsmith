"""Image I/O abstraction layer.

Tries OpenImageIO first for professional format support (EXR, DPX, 10-bit),
falls back to imageio v3 for common formats (PNG, TIFF, JPEG).

All images are returned as float32 arrays normalized to [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from chromaforge.config import (
    IMAGE_EXTENSIONS,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
)
from chromaforge.errors import ImageDimensionError, ImageFormatError

logger = logging.getLogger(__name__)

# Detect available backends
_HAS_OIIO = False
try:
    import OpenImageIO as oiio
    _HAS_OIIO = True
    logger.debug("OpenImageIO available")
except ImportError:
    pass

_HAS_IMAGEIO = False
try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
    logger.debug("imageio available")
except ImportError:
    pass


def validate_input_path(filepath: str | Path) -> Path:
    """Validate an input file path for security.

    Args:
        filepath: Path to validate.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If file does not exist.
        ImageFormatError: If extension is not allowed.
    """
    path = Path(filepath).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if not path.is_file():
        raise ImageFormatError(f"Not a regular file: {path}")

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ImageFormatError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    return path


def validate_output_path(filepath: str | Path) -> Path:
    """Validate an output file path.

    Args:
        filepath: Path to validate.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If parent directory does not exist.
    """
    import os
    path = Path(filepath).resolve()

    if not path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")

    if not os.access(path.parent, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {path.parent}")

    return path


def _validate_dimensions(width: int, height: int) -> None:
    """Check image dimensions before memory allocation."""
    if width <= 0 or height <= 0:
        raise ImageDimensionError(f"Invalid image dimensions: {width}x{height}")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ImageDimensionError(
            f"Image dimension {max(width, height)} exceeds "
            f"maximum allowed {MAX_IMAGE_DIMENSION}"
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageDimensionError(
            f"Image has {width * height:,} pixels, exceeds "
            f"maximum allowed {MAX_IMAGE_PIXELS:,}"
        )


def _load_oiio(path: Path) -> tuple[np.ndarray, dict]:
    """Load image using OpenImageIO."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise ImageFormatError(f"OIIO failed to open: {path}\n{oiio.geterror()}")

    try:
        spec = inp.spec()
        _validate_dimensions(spec.width, spec.height)

        # Read as float32
        data = inp.read_image(oiio.FLOAT)
        if data is None:
            raise ImageFormatError(f"OIIO failed to read: {path}\n{oiio.geterror()}")

        # Ensure 3 channels
        if spec.nchannels == 1:
            data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        elif spec.nchannels == 4:
            data = data[:, :, :3]  # Drop alpha
        elif spec.nchannels != 3:
            raise ImageFormatError(
                f"Unsupported channel count: {spec.nchannels}"
            )

        metadata = {
            "width": spec.width,
            "height": spec.height,
            "channels": spec.nchannels,
            "format": str(spec.format),
            "backend": "oiio",
        }

        return data.astype(np.float32), metadata
    finally:
        inp.close()


def _load_imageio(path: Path) -> tuple[np.ndarray, dict]:
    """Load image using imageio v3."""
    raw = iio.imread(str(path))

    _validate_dimensions(raw.shape[1], raw.shape[0])

    # Normalize to float32 [0, 1]
    if raw.dtype == np.uint8:
        data = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        data = raw.astype(np.float32) / 65535.0
    elif raw.dtype in (np.float32, np.float64):
        data = raw.astype(np.float32)
    else:
        data = raw.astype(np.float32)
        dinfo = np.iinfo(raw.dtype) if np.issubdtype(raw.dtype, np.integer) else None
        if dinfo is not None:
            data = data / dinfo.max

    # Ensure 3 channels
    if data.ndim == 2:
        data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    elif data.ndim == 3 and data.shape[2] == 4:
        data = data[:, :, :3]
    elif data.ndim == 3 and data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    elif data.ndim != 3 or data.shape[2] != 3:
        raise ImageFormatError(
            f"Unsupported image shape: {data.shape}"
        )

    metadata = {
        "width": data.shape[1],
        "height": data.shape[0],
        "channels": data.shape[2],
        "format": str(raw.dtype),
        "backend": "imageio",
    }

    return data, metadata


def load_image(filepath: str | Path) -> tuple[np.ndarray, dict]:
    """Load an image file as a float32 array.

    Tries OpenImageIO first (better for professional formats), falls back
    to imageio v3 (wider availability).

    Args:
        filepath: Path to image file.

    Returns:
        (array, metadata): (H, W, 3) float32 array and metadata dict.
    """
    path = validate_input_path(filepath)

    if _HAS_OIIO:
        try:
            logger.debug("Loading with OIIO: %s", path)
            return _load_oiio(path)
        except Exception as e:
            if not _HAS_IMAGEIO:
                raise
            logger.debug("OIIO failed, falling back to imageio: %s", e)

    if _HAS_IMAGEIO:
        logger.debug("Loading with imageio: %s", path)
        return _load_imageio(path)

    raise ImportError(
        "No image I/O backend available. "
        "Install imageio (`pip install imageio`) or "
        "OpenImageIO for professional format support."
    )


def save_image(
    array: np.ndarray,
    filepath: str | Path,
    bit_depth: int = 16,
) -> Path:
    """Save an image array to file.

    Args:
        array: (H, W, 3) float32 array in [0, 1].
        filepath: Output path.
        bit_depth: 8 or 16 bits per channel.

    Returns:
        Resolved output path.
    """
    path = validate_output_path(filepath)

    if bit_depth == 16:
        out = (np.clip(array, 0.0, 1.0) * 65535).astype(np.uint16)
    elif bit_depth == 8:
        out = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    if _HAS_OIIO and path.suffix.lower() in (".exr", ".dpx", ".hdr"):
        # Use OIIO for professional formats
        spec = oiio.ImageSpec(array.shape[1], array.shape[0], 3, oiio.UINT16 if bit_depth == 16 else oiio.UINT8)
        out_file = oiio.ImageOutput.create(str(path))
        if out_file is None:
            raise ImageFormatError(f"Cannot create output: {oiio.geterror()}")
        out_file.open(str(path), spec)
        out_file.write_image(out)
        out_file.close()
    elif _HAS_IMAGEIO:
        iio.imwrite(str(path), out)
    else:
        raise ImportError("No image I/O backend available for saving.")

    logger.info("Saved image: %s (%d-bit)", path, bit_depth)
    return path


def get_image_dimensions(filepath: str | Path) -> tuple[int, int]:
    """Get image dimensions without loading the full image.

    Returns:
        (width, height) tuple.
    """
    path = validate_input_path(filepath)

    if _HAS_OIIO:
        inp = oiio.ImageInput.open(str(path))
        if inp:
            spec = inp.spec()
            inp.close()
            return spec.width, spec.height

    if _HAS_IMAGEIO:
        props = iio.improps(str(path))
        if hasattr(props, "shape") and props.shape is not None:
            return props.shape[1], props.shape[0]
        # Fall back to full load for metadata
        img = iio.imread(str(path))
        return img.shape[1], img.shape[0]

    raise ImportError("No image I/O backend available.")
