"""Custom exception hierarchy for ChromaForge."""


class ChromaForgeError(Exception):
    """Base exception for all ChromaForge errors."""


class ImageError(ChromaForgeError):
    """Errors related to image loading or processing."""


class ImageFormatError(ImageError):
    """Unsupported or corrupted image format."""


class ImageDimensionError(ImageError):
    """Image dimensions exceed limits or are mismatched."""


class SolverError(ChromaForgeError):
    """Errors during the linear system solve."""


class SolverDivergenceError(SolverError):
    """Solver failed to converge."""


class ExportError(ChromaForgeError):
    """Errors during LUT export."""


class LUTFormatError(ExportError):
    """Invalid or corrupted LUT file format."""


class ValidationError(ChromaForgeError):
    """Input validation failures."""


class PipelineError(ChromaForgeError):
    """Errors during pipeline execution."""


class PipelineCancelledError(PipelineError):
    """Pipeline was cancelled by the user."""


class CoverageWarning(UserWarning):
    """Low coverage in some regions of the color space."""
