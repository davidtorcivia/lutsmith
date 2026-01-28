"""Tests for baseline transform fitting."""

from __future__ import annotations

import numpy as np
import pytest

from lutsmith.core.baseline import (
    PiecewiseLinearCurve,
    BaselineTransform,
    _pava_isotonic,
    fit_baseline,
    evaluate_baseline_lut,
    baseline_quality_gate,
)


class TestPAVA:
    """Tests for Pool Adjacent Violators Algorithm."""

    def test_already_monotone(self):
        """Monotone input should be unchanged."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones(5)
        result = _pava_isotonic(y, w)
        np.testing.assert_allclose(result, y)

    def test_reverse_order(self):
        """Reverse-sorted input should become constant (weighted mean)."""
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        w = np.ones(5)
        result = _pava_isotonic(y, w)
        expected_mean = 3.0
        np.testing.assert_allclose(result, expected_mean)

    def test_single_violation(self):
        """Single violation should be pooled."""
        y = np.array([1.0, 3.0, 2.0, 4.0])
        w = np.ones(4)
        result = _pava_isotonic(y, w)
        # Elements at index 1 and 2 should be pooled to their mean (2.5)
        assert result[0] <= result[1]
        assert result[1] <= result[2]
        assert result[2] <= result[3]
        np.testing.assert_allclose(result[1], result[2])

    def test_monotonicity_guarantee(self):
        """Output should always be non-decreasing."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(5, 50)
            y = rng.standard_normal(n)
            w = rng.random(n) + 0.1
            result = _pava_isotonic(y, w)
            assert np.all(np.diff(result) >= -1e-12), "PAVA output not monotone"

    def test_weighted_pooling(self):
        """Weights should affect the pooled values."""
        y = np.array([3.0, 1.0])  # violation
        w = np.array([1.0, 3.0])  # second element has 3x weight
        result = _pava_isotonic(y, w)
        # Pooled mean should be (3*1 + 1*3) / (1 + 3) = 1.5
        np.testing.assert_allclose(result, 1.5)

    def test_single_element(self):
        """Single element should be unchanged."""
        y = np.array([42.0])
        w = np.array([1.0])
        result = _pava_isotonic(y, w)
        np.testing.assert_allclose(result, [42.0])


class TestPiecewiseLinearCurve:
    """Tests for PiecewiseLinearCurve."""

    def test_evaluate_at_knots(self):
        """Evaluating at knot positions should return knot values."""
        curve = PiecewiseLinearCurve(
            knots_x=np.array([0.0, 0.5, 1.0]),
            knots_y=np.array([0.0, 0.4, 1.0]),
        )
        result = curve.evaluate(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(result, [0.0, 0.4, 1.0])

    def test_interpolation(self):
        """Should linearly interpolate between knots."""
        curve = PiecewiseLinearCurve(
            knots_x=np.array([0.0, 1.0]),
            knots_y=np.array([0.0, 2.0]),
        )
        result = curve.evaluate(np.array([0.25, 0.5, 0.75]))
        np.testing.assert_allclose(result, [0.5, 1.0, 1.5])

    def test_invert_roundtrip(self):
        """Evaluating curve then its inverse should recover input."""
        curve = PiecewiseLinearCurve(
            knots_x=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            knots_y=np.array([0.0, 0.1, 0.5, 0.8, 1.0]),
        )
        inv = curve.invert()
        x = np.linspace(0.05, 0.95, 20)
        y = curve.evaluate(x)
        x_recovered = inv.evaluate(y)
        np.testing.assert_allclose(x_recovered, x, atol=0.02)

    def test_identity_curve(self):
        """Identity curve: y=x should invert to itself."""
        curve = PiecewiseLinearCurve(
            knots_x=np.linspace(0, 1, 11),
            knots_y=np.linspace(0, 1, 11),
        )
        inv = curve.invert()
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(inv.evaluate(x), x, atol=1e-10)


class TestBaselineTransform:
    """Tests for BaselineTransform evaluation."""

    def test_identity_transform(self):
        """Identity M, zero b, identity curves -> output == input."""
        identity_curve = PiecewiseLinearCurve(
            knots_x=np.linspace(-0.5, 1.5, 20),
            knots_y=np.linspace(-0.5, 1.5, 20),
        )
        baseline = BaselineTransform(
            M=np.eye(3),
            b=np.zeros(3),
            curves=[identity_curve, identity_curve, identity_curve],
        )
        rng = np.random.default_rng(42)
        rgb = rng.random((50, 3))
        result = baseline.evaluate(rgb)
        np.testing.assert_allclose(result, rgb, atol=1e-10)

    def test_gain_transform(self):
        """Diagonal M with identity curves should apply gain."""
        identity_curve = PiecewiseLinearCurve(
            knots_x=np.linspace(-1, 2, 30),
            knots_y=np.linspace(-1, 2, 30),
        )
        M = np.diag([1.2, 0.9, 1.1])
        b = np.array([0.05, -0.02, 0.03])
        baseline = BaselineTransform(M=M, b=b, curves=[identity_curve] * 3)

        rgb = np.array([[0.5, 0.5, 0.5]])
        result = baseline.evaluate(rgb)
        expected = rgb @ M.T + b
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_output_shape(self):
        """Evaluate should preserve input shape."""
        identity_curve = PiecewiseLinearCurve(
            knots_x=np.linspace(0, 1, 10),
            knots_y=np.linspace(0, 1, 10),
        )
        baseline = BaselineTransform(
            M=np.eye(3), b=np.zeros(3), curves=[identity_curve] * 3,
        )
        rgb = np.random.default_rng(42).random((37, 3))
        result = baseline.evaluate(rgb)
        assert result.shape == (37, 3)


class TestFitBaseline:
    """Tests for baseline fitting."""

    def test_recover_affine_gain(self):
        """Fitting a pure affine transform should recover M and b."""
        rng = np.random.default_rng(42)
        M = 1000
        input_rgb = rng.random((M, 3))
        gain = np.diag([1.2, 0.9, 1.1])
        bias = np.array([0.05, -0.02, 0.03])
        output_rgb = np.clip(input_rgb @ gain.T + bias, 0, 1)
        alpha = np.ones(M)

        baseline = fit_baseline(input_rgb, output_rgb, alpha, n_iter=3)

        # Evaluate and check accuracy
        pred = baseline.evaluate(input_rgb)
        residuals = np.abs(pred - output_rgb)
        assert np.mean(residuals) < 0.02, f"Mean residual too high: {np.mean(residuals)}"

    def test_recover_with_curve(self):
        """Fitting data with nonlinear curve should produce low residual."""
        rng = np.random.default_rng(55)
        M = 1000
        input_rgb = rng.random((M, 3))
        # Apply a gamma-like curve
        output_rgb = np.clip(input_rgb ** 0.8, 0, 1)
        alpha = np.ones(M)

        baseline = fit_baseline(input_rgb, output_rgb, alpha, n_iter=3)

        pred = baseline.evaluate(input_rgb)
        residuals = np.abs(pred - output_rgb)
        assert np.mean(residuals) < 0.03, f"Mean residual too high: {np.mean(residuals)}"

    def test_baseline_has_monotone_curves(self):
        """Fitted curves should be monotone non-decreasing."""
        rng = np.random.default_rng(42)
        M = 500
        input_rgb = rng.random((M, 3))
        output_rgb = np.clip(input_rgb * 1.1 + 0.02, 0, 1)
        alpha = np.ones(M)

        baseline = fit_baseline(input_rgb, output_rgb, alpha, n_iter=2)

        for c, curve in enumerate(baseline.curves):
            assert np.all(np.diff(curve.knots_y) >= -1e-12), (
                f"Channel {c} curve is not monotone"
            )


class TestEvaluateBaselineLUT:
    """Tests for evaluate_baseline_lut."""

    def test_output_shape(self):
        """LUT should have shape (N, N, N, 3)."""
        identity_curve = PiecewiseLinearCurve(
            knots_x=np.linspace(-0.5, 1.5, 20),
            knots_y=np.linspace(-0.5, 1.5, 20),
        )
        baseline = BaselineTransform(
            M=np.eye(3), b=np.zeros(3), curves=[identity_curve] * 3,
        )
        for N in [3, 5]:
            lut = evaluate_baseline_lut(baseline, N)
            assert lut.shape == (N, N, N, 3)
            assert lut.dtype == np.float32

    def test_identity_baseline_gives_identity_lut(self):
        """Identity baseline should produce identity LUT."""
        from lutsmith.core.lut import identity_lut

        identity_curve = PiecewiseLinearCurve(
            knots_x=np.linspace(-0.5, 1.5, 100),
            knots_y=np.linspace(-0.5, 1.5, 100),
        )
        baseline = BaselineTransform(
            M=np.eye(3), b=np.zeros(3), curves=[identity_curve] * 3,
        )
        N = 5
        lut = evaluate_baseline_lut(baseline, N)
        id_lut = identity_lut(N)
        np.testing.assert_allclose(lut, id_lut, atol=0.02)


class TestBaselineQualityGate:
    """Tests for baseline_quality_gate."""

    def test_identity_data_fails_gate(self):
        """When output==input, baseline should NOT pass (no improvement)."""
        rng = np.random.default_rng(42)
        M = 500
        input_rgb = rng.random((M, 3))
        output_rgb = input_rgb.copy()  # identity mapping
        alpha = np.ones(M)

        baseline = fit_baseline(input_rgb, output_rgb, alpha, n_iter=2)
        # For identity data, baseline can't do better than identity
        passed = baseline_quality_gate(baseline, input_rgb, output_rgb, alpha)
        # This should fail or barely pass (ratio ≈ 1.0, not < 0.95)
        # Note: with perfect identity data the baseline IS the identity,
        # so ratio ≈ 1.0 and it should NOT pass the gate
        assert not passed, "Identity data should not pass quality gate"

    def test_gain_data_passes_gate(self):
        """Significant gain transform should pass quality gate."""
        rng = np.random.default_rng(42)
        M = 1000
        input_rgb = rng.random((M, 3))
        gain = np.array([1.3, 0.7, 1.2])
        output_rgb = np.clip(input_rgb * gain, 0, 1)
        alpha = np.ones(M)

        baseline = fit_baseline(input_rgb, output_rgb, alpha, n_iter=3)
        passed = baseline_quality_gate(baseline, input_rgb, output_rgb, alpha)
        assert passed, "Gain transform should pass quality gate"
