"""Tests for sparse matrix construction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from lutsmith.core.matrix import (
    compute_interpolation_weights,
    build_data_matrix,
    build_data_rhs,
    build_smoothness_matrix,
    build_prior_matrix,
    build_full_system,
)
from lutsmith.core.types import flat_index


class TestInterpolationWeights:
    """Tests for pre-computed interpolation weights."""

    def test_shapes(self, random_colors, small_N):
        """Output shapes should match input."""
        M = 30
        colors = random_colors[:M]
        indices, weights = compute_interpolation_weights(colors, small_N, "tetrahedral")
        assert indices.shape[0] == M
        assert weights.shape[0] == M
        assert indices.shape[1] == weights.shape[1]

    def test_tetrahedral_4_cols(self, random_colors, small_N):
        """Tetrahedral should produce 4 columns."""
        indices, weights = compute_interpolation_weights(
            random_colors[:10], small_N, "tetrahedral"
        )
        assert indices.shape[1] == 4

    def test_trilinear_8_cols(self, random_colors, small_N):
        """Trilinear should produce 8 columns."""
        indices, weights = compute_interpolation_weights(
            random_colors[:10], small_N, "trilinear"
        )
        assert indices.shape[1] == 8


class TestDataMatrix:
    """Tests for data fidelity matrix construction."""

    def test_dimensions(self, random_colors, small_N):
        """Data matrix should be M x N^3."""
        M = 20
        colors = random_colors[:M]
        N = small_N
        indices, weights = compute_interpolation_weights(colors, N, "tetrahedral")
        alpha = np.ones(M, dtype=np.float32)

        A_data = build_data_matrix(indices, weights, alpha, N)
        assert A_data.shape == (M, N ** 3)

    def test_sqrt_scaling(self, random_colors, small_N):
        """Rows should be scaled by sqrt(alpha)."""
        M = 5
        N = small_N
        colors = random_colors[:M]
        indices, weights = compute_interpolation_weights(colors, N, "tetrahedral")

        # Uniform alpha=4 -> rows scaled by 2
        alpha_4 = np.full(M, 4.0, dtype=np.float32)
        alpha_1 = np.ones(M, dtype=np.float32)

        A4 = build_data_matrix(indices, weights, alpha_4, N)
        A1 = build_data_matrix(indices, weights, alpha_1, N)

        # A4 rows should be 2x A1 rows
        np.testing.assert_allclose(A4.toarray(), 2.0 * A1.toarray(), atol=1e-5)


class TestSmoothnessMatrix:
    """Tests for smoothness regularization matrix."""

    def test_dimensions(self, small_N):
        """Smoothness matrix should have N^3 columns."""
        N = small_N
        A_smooth, b_smooth = build_smoothness_matrix(N, lambda_s=0.1)
        assert A_smooth.shape[1] == N ** 3
        assert len(b_smooth) == N ** 3

    def test_zero_lambda_gives_zero(self, small_N):
        """lambda_s=0 should produce zero matrix."""
        A_smooth, _ = build_smoothness_matrix(small_N, lambda_s=0.0)
        assert sparse.linalg.norm(A_smooth) < 1e-10


class TestPriorMatrix:
    """Tests for identity prior matrix."""

    def test_dimensions(self, small_N):
        """Prior matrix should be N^3 x N^3 diagonal."""
        N = small_N
        N3 = N ** 3
        distances = np.ones(N3, dtype=np.float32)
        prior_ch = np.linspace(0, 1, N3, dtype=np.float32)
        A_prior, b_prior = build_prior_matrix(
            N, lambda_r=0.01, distances=distances, prior_channel=prior_ch,
        )
        assert A_prior.shape == (N3, N3)
        assert len(b_prior) == N3

    def test_zero_lambda_gives_zero(self, small_N):
        """lambda_r=0 should produce zero matrix."""
        N3 = small_N ** 3
        distances = np.ones(N3, dtype=np.float32)
        prior_ch = np.linspace(0, 1, N3, dtype=np.float32)
        A_prior, _ = build_prior_matrix(
            small_N, lambda_r=0.0, distances=distances, prior_channel=prior_ch,
        )
        assert sparse.linalg.norm(A_prior) < 1e-10


class TestFullSystem:
    """Tests for complete system assembly."""

    def test_full_system_shape(self, small_N, random_colors):
        """Full system A should have N^3 columns."""
        N = small_N
        M = 20
        colors = random_colors[:M]
        output_r = np.random.rand(M).astype(np.float32)
        alpha = np.ones(M, dtype=np.float32)

        A, b = build_full_system(
            input_rgb=colors,
            output_channel=output_r,
            sample_alpha=alpha,
            N=N,
            lambda_s=0.1,
            lambda_r=0.01,
        )

        assert A.shape[1] == N ** 3
        assert A.shape[0] == b.shape[0]
        assert A.shape[0] >= M  # At least data rows


class TestPriorBoost:
    """Tests for prior_boost parameter in build_prior_matrix."""

    def test_boost_increases_prior_strength(self, small_N):
        """prior_boost > 1 should increase diagonal entries."""
        N = small_N
        N3 = N ** 3
        distances = np.ones(N3, dtype=np.float32)
        prior_ch = np.linspace(0, 1, N3, dtype=np.float32)

        # Without boost
        A_base, b_base = build_prior_matrix(
            N, lambda_r=0.01, distances=distances, prior_channel=prior_ch,
        )

        # With boost of 4.0 (should scale by sqrt(4) = 2)
        boost = np.full(N3, 4.0)
        A_boosted, b_boosted = build_prior_matrix(
            N, lambda_r=0.01, distances=distances, prior_channel=prior_ch,
            prior_boost=boost,
        )

        # Diagonal of boosted should be 2x the base (sqrt(4) = 2)
        diag_base = A_base.diagonal()
        diag_boosted = A_boosted.diagonal()
        np.testing.assert_allclose(diag_boosted, 2.0 * diag_base, atol=1e-10)

    def test_boost_of_one_is_no_op(self, small_N):
        """prior_boost of all ones should not change the result."""
        N = small_N
        N3 = N ** 3
        distances = np.ones(N3, dtype=np.float32)
        prior_ch = np.linspace(0, 1, N3, dtype=np.float32)

        A_base, b_base = build_prior_matrix(
            N, lambda_r=0.01, distances=distances, prior_channel=prior_ch,
        )

        boost = np.ones(N3)
        A_boosted, b_boosted = build_prior_matrix(
            N, lambda_r=0.01, distances=distances, prior_channel=prior_ch,
            prior_boost=boost,
        )

        np.testing.assert_allclose(
            A_boosted.toarray(), A_base.toarray(), atol=1e-12
        )
        np.testing.assert_allclose(b_boosted, b_base, atol=1e-12)

    def test_none_boost_is_no_op(self, small_N):
        """prior_boost=None should be identical to no boost."""
        N = small_N
        N3 = N ** 3
        distances = np.ones(N3, dtype=np.float32)
        prior_ch = np.linspace(0, 1, N3, dtype=np.float32)

        A_base, b_base = build_prior_matrix(
            N, lambda_r=0.01, distances=distances, prior_channel=prior_ch,
        )

        A_none, b_none = build_prior_matrix(
            N, lambda_r=0.01, distances=distances, prior_channel=prior_ch,
            prior_boost=None,
        )

        np.testing.assert_allclose(
            A_none.toarray(), A_base.toarray(), atol=1e-12
        )
        np.testing.assert_allclose(b_none, b_base, atol=1e-12)
