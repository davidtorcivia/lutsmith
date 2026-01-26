"""Core data types, enums, and indexing helpers for ChromaForge.

CRITICAL CONVENTION:
    LUT arrays have shape (N, N, N, 3) indexed as lut[r, g, b, channel].
    Flat index: flat = b * N * N + g * N + r  (R varies fastest, matching .cube spec).
    This convention MUST be used consistently in ALL modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Indexing helpers -- single source of truth for the flat <-> 3D mapping
# ---------------------------------------------------------------------------

def flat_index(r: int, g: int, b: int, N: int) -> int:
    """Convert 3D grid indices to flat index. R varies fastest."""
    return b * N * N + g * N + r


def grid_indices(flat: int, N: int) -> tuple[int, int, int]:
    """Convert flat index to (r, g, b) grid indices."""
    r = flat % N
    g = (flat // N) % N
    b = flat // (N * N)
    return r, g, b


def flat_index_array(r: np.ndarray, g: np.ndarray, b: np.ndarray, N: int) -> np.ndarray:
    """Vectorized flat index computation for arrays of indices."""
    return b * N * N + g * N + r


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InterpolationKernel(str, Enum):
    """Interpolation method for 3D LUT lookup."""
    TRILINEAR = "trilinear"
    TETRAHEDRAL = "tetrahedral"


class RobustLoss(str, Enum):
    """Robust loss function for IRLS."""
    L2 = "l2"
    HUBER = "huber"


class TransferFunction(str, Enum):
    """Input transfer function / encoding."""
    LINEAR = "linear"
    LOG_C3 = "log_c3"
    LOG_C4 = "log_c4"
    SLOG3 = "slog3"
    VLOG = "vlog"
    AUTO = "auto"
    UNKNOWN = "unknown"


class ExportFormat(str, Enum):
    """LUT export format."""
    CUBE = "cube"
    AML = "aml"
    ALF4 = "alf4"
    CLF = "clf"


class WeightingMode(str, Enum):
    """Sample weighting strategy."""
    COVERAGE_FAIR = "coverage_fair"
    FREQUENCY = "frequency"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BinStatistics:
    """Aggregated statistics for a single color-space bin."""
    mean_input: np.ndarray   # (3,) mean input RGB
    mean_output: np.ndarray  # (3,) mean output RGB
    count: int               # number of samples in this bin
    variance: np.ndarray     # (3,) output variance per channel (from Welford)
    spatial_centroid: np.ndarray  # (2,) mean (x, y) position
    bin_index: int           # flat bin index

    @property
    def confidence(self) -> float:
        """Confidence score based on count and variance."""
        from chromaforge.config import DEFAULT_MIN_SAMPLES_PER_BIN, EPSILON
        k = 0.5
        var_mag = float(np.mean(self.variance))
        count_factor = min(1.0, self.count / max(DEFAULT_MIN_SAMPLES_PER_BIN, 1))
        return (1.0 / (1.0 + k * var_mag)) * count_factor

    @property
    def contradiction_score(self) -> float:
        """Score indicating likelihood of local adjustment."""
        from chromaforge.config import EPSILON
        var_mag = float(np.mean(self.variance))
        out_mag = float(np.linalg.norm(self.mean_output))
        return var_mag / (out_mag + 0.05)


@dataclass
class SamplePoint:
    """A single data point for the regression system."""
    input_rgb: np.ndarray   # (3,) input color
    output_rgb: np.ndarray  # (3,) output color
    alpha: float            # per-sample weight


@dataclass
class LUTData:
    """Container for a 3D LUT with metadata."""
    array: np.ndarray  # (N, N, N, 3) float32, indexed as [r, g, b, ch]
    size: int          # N (grid resolution per axis)
    title: str = ""
    kernel: InterpolationKernel = InterpolationKernel.TETRAHEDRAL
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        expected = (self.size, self.size, self.size, 3)
        if self.array.shape != expected:
            raise ValueError(f"LUT array shape {self.array.shape} != expected {expected}")


@dataclass
class QualityMetrics:
    """Quality metrics from LUT validation."""
    mean_delta_e: float = 0.0
    median_delta_e: float = 0.0
    p95_delta_e: float = 0.0
    max_delta_e: float = 0.0
    max_delta_e_location: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    total_variation: float = 0.0
    neutral_monotonic: bool = True
    neutral_mono_violations: int = 0
    oog_percentage: float = 0.0
    coverage_percentage: float = 0.0
    num_occupied_bins: int = 0
    num_total_bins: int = 0


@dataclass
class PipelineConfig:
    """Configuration for the full LUT extraction pipeline."""
    # Input
    source_path: Optional[Path] = None
    target_path: Optional[Path] = None

    # Output
    output_path: Optional[Path] = None
    format: ExportFormat = ExportFormat.CUBE
    title: str = "ChromaForge LUT"

    # Core parameters
    lut_size: int = 33
    smoothness: float = 0.1
    prior_strength: float = 0.01
    kernel: InterpolationKernel = InterpolationKernel.TETRAHEDRAL

    # Advanced
    bin_resolution: int = 64
    min_samples_per_bin: int = 3
    robust_loss: RobustLoss = RobustLoss.HUBER
    huber_delta: float = 1.0
    irls_iterations: int = 3
    transfer_function: TransferFunction = TransferFunction.AUTO
    weighting: WeightingMode = WeightingMode.COVERAGE_FAIR
    alignment: str = "none"  # "none" or "auto"
    shadow_auto: bool = True
    shadow_threshold: Optional[float] = None
    deep_shadow_threshold: Optional[float] = None

    # Refinement
    enable_refinement: bool = False
    refinement_iterations: int = 2

    # Output options
    include_shaper: Optional[bool] = None  # None = auto, True = force, False = disable
    generate_coverage_report: bool = True
    generate_metrics: bool = True


@dataclass
class PipelineResult:
    """Result from a full pipeline run."""
    lut: LUTData
    metrics: QualityMetrics
    coverage_map: Optional[np.ndarray] = None  # (N, N, N) distances
    diagnostics: dict = field(default_factory=dict)
    output_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, float, str], None]
"""Callback signature: (stage_name, fraction_complete, message)."""

CancelCheck = Callable[[], bool]
"""Returns True if the operation should be cancelled."""
