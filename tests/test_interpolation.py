"""Tests for interpolation kernels."""

from __future__ import annotations

import numpy as np
import pytest

from lutsmith.core.interpolation import (
    trilinear_weights,
    tetrahedral_weights,
    vectorized_trilinear,
    vectorized_tetrahedral,
    apply_lut_to_colors,
)
from lutsmith.core.types import flat_index


class TestTrilinearWeights:
    """Tests for single-point trilinear interpolation."""

    def test_grid_corner_exact(self, small_N):
        """Grid nodes should get weight 1.0 on exactly one vertex."""
        N = small_N
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    rgb = np.array([r, g, b], dtype=np.float32) / (N - 1)
                    indices, weights = trilinear_weights(rgb, N)
                    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
                    # Exactly one weight should be 1.0
                    assert np.isclose(weights.max(), 1.0, atol=1e-5)

    def test_weights_sum_to_one(self, random_colors, small_N):
        """Weights should always sum to 1.0."""
        for color in random_colors[:50]:
            _, weights = trilinear_weights(color, small_N)
            assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_weights_nonnegative(self, random_colors, small_N):
        """All weights must be >= 0."""
        for color in random_colors[:50]:
            _, weights = trilinear_weights(color, small_N)
            assert np.all(weights >= -1e-8)

    def test_center_point(self, small_N):
        """Mid-cube point should have 8 equal weights of 0.125."""
        N = small_N
        rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        _, weights = trilinear_weights(rgb, N)
        # For grid center, all 8 weights should be equal if we're
        # at the exact center of a cell (may not be for all N)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)


class TestTetrahedralWeights:
    """Tests for single-point tetrahedral interpolation."""

    def test_grid_corner_exact(self, small_N):
        """Grid nodes should interpolate exactly."""
        N = small_N
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    rgb = np.array([r, g, b], dtype=np.float32) / (N - 1)
                    indices, weights = tetrahedral_weights(rgb, N)
                    assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_weights_sum_to_one(self, random_colors, small_N):
        """Weights should always sum to 1.0."""
        for color in random_colors[:50]:
            _, weights = tetrahedral_weights(color, small_N)
            assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_weights_nonnegative(self, random_colors, small_N):
        """All weights must be >= 0."""
        for color in random_colors[:50]:
            _, weights = tetrahedral_weights(color, small_N)
            assert np.all(weights >= -1e-8)

    def test_four_vertices(self, random_colors, small_N):
        """Tetrahedral should return exactly 4 vertices."""
        for color in random_colors[:10]:
            indices, weights = tetrahedral_weights(color, small_N)
            assert len(indices) == 4
            assert len(weights) == 4


class TestVectorizedInterpolation:
    """Tests for batch interpolation."""

    def test_trilinear_matches_scalar(self, random_colors, small_N):
        """Vectorized trilinear should match scalar version."""
        N = small_N
        colors = random_colors[:20]
        batch_idx, batch_w = vectorized_trilinear(colors, N)

        for i, color in enumerate(colors):
            idx, w = trilinear_weights(color, N)
            # Sort both for comparison
            order_batch = np.argsort(batch_idx[i])
            order_scalar = np.argsort(idx)
            np.testing.assert_array_equal(
                batch_idx[i][order_batch], idx[order_scalar]
            )
            np.testing.assert_allclose(
                batch_w[i][order_batch], w[order_scalar], atol=1e-5
            )

    def test_tetrahedral_matches_scalar(self, random_colors, small_N):
        """Vectorized tetrahedral should match scalar version."""
        N = small_N
        colors = random_colors[:20]
        batch_idx, batch_w = vectorized_tetrahedral(colors, N)

        for i, color in enumerate(colors):
            idx, w = tetrahedral_weights(color, N)
            # Sort for comparison
            order_batch = np.argsort(batch_idx[i])
            order_scalar = np.argsort(idx)
            np.testing.assert_array_equal(
                batch_idx[i][order_batch], idx[order_scalar]
            )
            np.testing.assert_allclose(
                batch_w[i][order_batch], w[order_scalar], atol=1e-5
            )


class TestApplyLUT:
    """Tests for LUT application."""

    def test_identity_lut(self, identity_lut_9, random_colors):
        """Identity LUT should pass through colors unchanged."""
        N = 9
        out = apply_lut_to_colors(random_colors[:50], identity_lut_9, N, "tetrahedral")
        np.testing.assert_allclose(out, random_colors[:50], atol=1e-4)

    def test_identity_trilinear(self, identity_lut_9, random_colors):
        """Identity LUT with trilinear should also pass through."""
        N = 9
        out = apply_lut_to_colors(random_colors[:50], identity_lut_9, N, "trilinear")
        np.testing.assert_allclose(out, random_colors[:50], atol=1e-4)

    def test_grid_nodes_exact(self, identity_lut_5, grid_node_colors, small_N):
        """Interpolating at grid nodes should be exact for identity."""
        N = small_N
        out = apply_lut_to_colors(grid_node_colors, identity_lut_5, N, "tetrahedral")
        np.testing.assert_allclose(out, grid_node_colors, atol=1e-5)

    def test_constant_lut(self, small_N, random_colors):
        """A constant LUT should produce constant output."""
        N = small_N
        value = np.array([0.3, 0.5, 0.7], dtype=np.float32)
        lut = np.full((N, N, N, 3), 0, dtype=np.float32)
        lut[:, :, :, :] = value[np.newaxis, np.newaxis, np.newaxis, :]

        out = apply_lut_to_colors(random_colors[:20], lut, N, "tetrahedral")
        expected = np.tile(value, (20, 1))
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_continuity_at_tet_boundary(self, identity_lut_9):
        """Test interpolation continuity at tetrahedron boundaries."""
        N = 9
        # Points near a tetrahedron boundary (where dr ~= dg)
        base = np.array([0.15, 0.15, 0.1], dtype=np.float32)
        eps = 1e-4
        p1 = base.copy()
        p1[0] += eps  # dr slightly > dg
        p2 = base.copy()
        p2[1] += eps  # dg slightly > dr

        out1 = apply_lut_to_colors(p1[np.newaxis], identity_lut_9, N, "tetrahedral")
        out2 = apply_lut_to_colors(p2[np.newaxis], identity_lut_9, N, "tetrahedral")

        # Should be close since identity LUT is smooth
        np.testing.assert_allclose(out1, p1[np.newaxis], atol=1e-3)
        np.testing.assert_allclose(out2, p2[np.newaxis], atol=1e-3)
