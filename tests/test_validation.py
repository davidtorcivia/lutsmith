"""Tests for LUT validation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from chromaforge.core.lut import identity_lut
from chromaforge.pipeline.validation import validate_lut


class TestValidation:
    """Tests for validation metrics computation."""

    def test_identity_zero_error(self):
        """Identity LUT applied to matching pairs should give dE ~= 0."""
        N = 9
        lut = identity_lut(N)

        rng = np.random.default_rng(42)
        M = 500
        colors = rng.random((M, 3), dtype=np.float32)

        metrics = validate_lut(
            source_rgb=colors,
            target_rgb=colors,  # target == source for identity
            lut=lut,
            N=N,
            kernel="tetrahedral",
            subsample=False,
        )

        # Mean DeltaE should be very small for identity
        assert metrics.mean_delta_e < 1.0
        assert metrics.median_delta_e < 1.0
        assert metrics.p95_delta_e < 2.0

    def test_health_metrics_identity(self):
        """Identity LUT should have good health metrics."""
        N = 9
        lut = identity_lut(N)
        rng = np.random.default_rng(42)
        M = 200
        colors = rng.random((M, 3), dtype=np.float32)

        metrics = validate_lut(
            source_rgb=colors,
            target_rgb=colors,
            lut=lut,
            N=N,
            subsample=False,
        )

        # Identity should be monotonic
        assert metrics.neutral_monotonic is True
        assert metrics.neutral_mono_violations == 0

        # Identity should have 0% OOG
        assert metrics.oog_percentage < 0.1

    def test_metrics_with_gain(self):
        """Known gain transform should produce low error when LUT matches."""
        N = 9
        lut = identity_lut(N)
        # Apply gain to the LUT itself
        gain = np.array([1.1, 0.95, 1.0], dtype=np.float32)
        lut_gained = np.clip(lut * gain[np.newaxis, np.newaxis, np.newaxis, :], 0, 1)

        rng = np.random.default_rng(42)
        M = 300
        source = rng.random((M, 3), dtype=np.float32)
        target = np.clip(source * gain, 0, 1).astype(np.float32)

        metrics = validate_lut(
            source_rgb=source,
            target_rgb=target,
            lut=lut_gained,
            N=N,
            subsample=False,
        )

        # Should be reasonably accurate
        assert metrics.mean_delta_e < 5.0

    def test_total_variation_positive(self):
        """Total variation should be non-negative."""
        N = 5
        lut = identity_lut(N)
        rng = np.random.default_rng(42)
        M = 100
        colors = rng.random((M, 3), dtype=np.float32)

        metrics = validate_lut(
            source_rgb=colors,
            target_rgb=colors,
            lut=lut,
            N=N,
            subsample=False,
        )

        assert metrics.total_variation >= 0.0

    def test_metrics_output_fields(self):
        """All expected fields should be present in QualityMetrics."""
        N = 5
        lut = identity_lut(N)
        rng = np.random.default_rng(42)
        colors = rng.random((100, 3), dtype=np.float32)

        metrics = validate_lut(
            source_rgb=colors,
            target_rgb=colors,
            lut=lut,
            N=N,
            subsample=False,
        )

        assert hasattr(metrics, "mean_delta_e")
        assert hasattr(metrics, "median_delta_e")
        assert hasattr(metrics, "p95_delta_e")
        assert hasattr(metrics, "max_delta_e")
        assert hasattr(metrics, "total_variation")
        assert hasattr(metrics, "neutral_monotonic")
        assert hasattr(metrics, "oog_percentage")
