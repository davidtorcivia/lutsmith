"""Custom exception hierarchy for LutSmith."""


class LutSmithError(Exception):
    """Base exception for all LutSmith errors."""


class ImageError(LutSmithError):
    """Errors related to image loading or processing."""


class ImageFormatError(ImageError):
    """Unsupported or corrupted image format."""


class ImageDimensionError(ImageError):
    """Image dimensions exceed limits or are mismatched."""


class SolverError(LutSmithError):
    """Errors during the linear system solve."""


class SolverDivergenceError(SolverError):
    """Solver failed to converge."""


class ExportError(LutSmithError):
    """Errors during LUT export."""


class LUTFormatError(ExportError):
    """Invalid or corrupted LUT file format."""


class ValidationError(LutSmithError):
    """Input validation failures."""


class PipelineError(LutSmithError):
    """Errors during pipeline execution."""


class PipelineCancelledError(PipelineError):
    """Pipeline was cancelled by the user."""


class CoverageWarning(UserWarning):
    """Low coverage in some regions of the color space."""
