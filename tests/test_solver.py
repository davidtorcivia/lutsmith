"""Tests for the sparse solver and IRLS."""

from __future__ import annotations

import numpy as np
import pytest

from lutsmith.core.solver import solve_lsmr, solve_irls, solve_per_channel
from lutsmith.core.lut import identity_lut
from lutsmith.core.interpolation import apply_lut_to_colors


class TestSolveLSMR:
    """Tests for the basic LSMR wrapper."""

    def test_simple_system(self):
        """Solve a trivial overdetermined system."""
        from scipy.sparse import eye
        A = eye(10, format="csr")
        b = np.arange(10, dtype=np.float64)
        x, info = solve_lsmr(A, b)
        np.testing.assert_allclose(x, b, atol=1e-6)
        assert info["converged"]

    def test_convergence_info(self):
        """Info dict should contain convergence data."""
        from scipy.sparse import eye
        A = eye(5, format="csr")
        b = np.ones(5)
        _, info = solve_lsmr(A, b)
        assert "istop" in info
        assert "iterations" in info
        assert "converged" in info


class TestSolveIRLS:
    """Tests for IRLS robust solver."""

    def test_l2_mode(self):
        """L2 loss should be equivalent to single LSMR call."""
        from scipy.sparse import eye
        A = eye(10, format="csr")
        b = np.arange(10, dtype=np.float64)

        x, info = solve_irls(A, b, loss="l2")
        np.testing.assert_allclose(x, b, atol=1e-6)
        assert info.get("irls_iterations", 0) == 0

    def test_huber_convergence(self):
        """Huber IRLS should converge for a well-conditioned system."""
        from scipy.sparse import eye
        A = eye(20, format="csr")
        b = np.random.default_rng(42).standard_normal(20)

        x, info = solve_irls(A, b, loss="huber", max_iter=5)
        np.testing.assert_allclose(x, b, atol=1e-3)

    def test_unknown_loss_raises(self):
        """Unknown loss function should raise ValueError."""
        from scipy.sparse import eye
        A = eye(5, format="csr")
        b = np.ones(5)
        with pytest.raises(ValueError, match="Unknown loss"):
            solve_irls(A, b, loss="magic")


class TestSolvePerChannel:
    """Tests for the per-channel solver orchestration."""

    def test_identity_recovery(self):
        """Given identity mapping (output == input), solver should recover identity LUT."""
        rng = np.random.default_rng(42)
        N = 5
        M = 200
        input_rgb = rng.random((M, 3), dtype=np.float32)
        output_rgb = input_rgb.copy()  # identity mapping
        alpha = np.ones(M, dtype=np.float32)

        lut, infos = solve_per_channel(
            input_rgb, output_rgb, alpha,
            N=N,
            lambda_s=0.1,
            lambda_r=0.01,
            kernel="tetrahedral",
            loss="l2",
            irls_iterations=0,
            parallel=False,
        )

        # Check that the LUT approximates identity
        id_lut = identity_lut(N)
        np.testing.assert_allclose(lut, id_lut, atol=0.1)

    def test_gain_recovery(self):
        """Solver should recover a simple gain transform."""
        rng = np.random.default_rng(55)
        N = 5
        M = 300
        input_rgb = rng.random((M, 3), dtype=np.float32)
        gain = np.array([1.1, 0.9, 1.05], dtype=np.float32)
        output_rgb = np.clip(input_rgb * gain, 0, 1).astype(np.float32)
        alpha = np.ones(M, dtype=np.float32)

        lut, infos = solve_per_channel(
            input_rgb, output_rgb, alpha,
            N=N,
            lambda_s=0.05,
            lambda_r=0.001,
            kernel="tetrahedral",
            loss="l2",
            irls_iterations=0,
            parallel=False,
        )

        # Apply LUT and check accuracy
        test_colors = rng.random((100, 3), dtype=np.float32)
        predicted = apply_lut_to_colors(test_colors, lut, N, "tetrahedral")
        expected = np.clip(test_colors * gain, 0, 1)
        np.testing.assert_allclose(predicted, expected, atol=0.15)

    def test_output_shape(self):
        """Output LUT should have correct shape."""
        rng = np.random.default_rng(42)
        N = 5
        M = 100
        input_rgb = rng.random((M, 3), dtype=np.float32)
        output_rgb = input_rgb.copy()
        alpha = np.ones(M, dtype=np.float32)

        lut, infos = solve_per_channel(
            input_rgb, output_rgb, alpha,
            N=N, lambda_s=0.1, lambda_r=0.01,
            loss="l2", irls_iterations=0, parallel=False,
        )

        assert lut.shape == (N, N, N, 3)
        assert lut.dtype == np.float32
        assert len(infos) == 3
