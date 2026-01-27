"""Tests for shaper LUT generation and monotonicity enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from lutsmith.color.shaper import (
    enforce_monotonic,
    get_shaper,
    identity_forward,
    identity_inverse,
    log_shaper_forward,
    log_shaper_inverse,
    generate_1d_shaper,
)
from lutsmith.core.types import TransferFunction


class TestShaperFunctions:
    """Tests for shaper forward/inverse functions."""

    def test_identity_roundtrip(self):
        """Identity shaper should pass through unchanged."""
        x = np.linspace(0, 1, 100, dtype=np.float32)
        y = identity_forward(x)
        np.testing.assert_array_equal(y, x)
        x_back = identity_inverse(y)
        np.testing.assert_array_equal(x_back, x)

    def test_log_shaper_roundtrip(self):
        """Log shaper forward -> inverse should recover input."""
        x = np.linspace(0.001, 1.0, 100, dtype=np.float32)
        y = log_shaper_forward(x)
        x_back = log_shaper_inverse(y)
        np.testing.assert_allclose(x_back, x, atol=1e-4)

    def test_log_shaper_monotonic(self):
        """Log shaper output should be monotonically increasing."""
        x = np.linspace(0.001, 1.0, 100, dtype=np.float32)
        y = log_shaper_forward(x)
        assert np.all(np.diff(y) >= -1e-8)

    def test_log_shaper_range(self):
        """Log shaper output should be in [0, 1]."""
        x = np.linspace(0.001, 1.0, 100, dtype=np.float32)
        y = log_shaper_forward(x)
        assert y.min() >= -0.01
        assert y.max() <= 1.01


class TestGetShaper:
    """Tests for get_shaper factory."""

    def test_linear_gets_log_shaper(self):
        """Linear input should get a log shaper."""
        forward, inverse = get_shaper(TransferFunction.LINEAR)
        assert forward is not None
        assert inverse is not None

    def test_log_input_gets_identity(self):
        """Log-encoded input should get identity shaper."""
        forward, inverse = get_shaper(TransferFunction.LOG_C3)
        # For log inputs, shaper should be identity-like
        x = np.linspace(0, 1, 50, dtype=np.float32)
        np.testing.assert_allclose(forward(x), x, atol=1e-5)


class TestEnforceMonotonic:
    """Tests for monotonicity enforcement."""

    def test_already_monotonic(self):
        """Already monotonic values should be unchanged."""
        x = np.linspace(0, 1, 20, dtype=np.float32)
        result = enforce_monotonic(x)
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_non_monotonic_fixed(self):
        """Non-monotonic values should be made monotonic."""
        x = np.array([0.0, 0.5, 0.3, 0.8, 1.0], dtype=np.float32)
        result = enforce_monotonic(x)
        # Should be non-decreasing
        assert np.all(np.diff(result) >= -1e-8)

    def test_constant_preserved(self):
        """Constant values should remain constant (they are monotonic)."""
        x = np.full(10, 0.5, dtype=np.float32)
        result = enforce_monotonic(x)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)


class TestGenerate1DShaper:
    """Tests for 1D shaper LUT generation."""

    def test_output_shape(self):
        """Generated shaper should have (size, 3) shape."""
        shaper = generate_1d_shaper(
            TransferFunction.LINEAR, size=256
        )
        assert shaper.shape == (256, 3)

    def test_range(self):
        """Shaper values should be in [0, 1]."""
        shaper = generate_1d_shaper(TransferFunction.LINEAR, size=64)
        assert shaper.min() >= -0.01
        assert shaper.max() <= 1.01
