"""Tests for opponent color space utilities."""

from __future__ import annotations

import numpy as np
import pytest

from lutsmith.core.opponent import (
    OPPONENT_BASIS,
    OPPONENT_BASIS_INV,
    rgb_to_opponent,
    opponent_to_rgb,
    compute_neutral_chroma_boost,
)


class TestOpponentBasis:
    """Tests for the opponent basis matrix properties."""

    def test_orthonormality(self):
        """Basis should be orthonormal: U @ U.T == I."""
        product = OPPONENT_BASIS @ OPPONENT_BASIS.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-12)

    def test_inverse_is_transpose(self):
        """For orthonormal matrices, inverse == transpose."""
        np.testing.assert_allclose(OPPONENT_BASIS_INV, OPPONENT_BASIS.T, atol=1e-12)

    def test_inverse_recovers_identity(self):
        """U^{-1} @ U == I."""
        product = OPPONENT_BASIS_INV.T @ OPPONENT_BASIS.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-12)

    def test_rows_are_unit_length(self):
        """Each row of the basis should have unit norm."""
        for i in range(3):
            norm = np.linalg.norm(OPPONENT_BASIS[i])
            np.testing.assert_allclose(norm, 1.0, atol=1e-12)


class TestRGBOpponentRoundtrip:
    """Tests for RGB <-> opponent conversions."""

    def test_roundtrip_random(self):
        """RGB -> opponent -> RGB should be identity."""
        rng = np.random.default_rng(42)
        rgb = rng.random((100, 3))
        recovered = opponent_to_rgb(rgb_to_opponent(rgb))
        np.testing.assert_allclose(recovered, rgb, atol=1e-12)

    def test_roundtrip_single(self):
        """Single color roundtrip."""
        rgb = np.array([[0.5, 0.3, 0.8]])
        recovered = opponent_to_rgb(rgb_to_opponent(rgb))
        np.testing.assert_allclose(recovered, rgb, atol=1e-12)

    def test_roundtrip_corners(self, grid_corner_colors):
        """All 8 RGB cube corners should roundtrip exactly."""
        corners = grid_corner_colors.astype(np.float64)
        recovered = opponent_to_rgb(rgb_to_opponent(corners))
        np.testing.assert_allclose(recovered, corners, atol=1e-12)

    def test_black_and_white(self):
        """Black and white should roundtrip."""
        bw = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        recovered = opponent_to_rgb(rgb_to_opponent(bw))
        np.testing.assert_allclose(recovered, bw, atol=1e-12)


class TestNeutralAxis:
    """Tests for neutral axis behavior in opponent space."""

    def test_neutral_has_zero_chroma(self):
        """Neutral colors (r==g==b) should have C1=C2=0."""
        neutral = np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
            [1.0, 1.0, 1.0],
        ])
        opp = rgb_to_opponent(neutral)
        # C1 and C2 columns should be zero
        np.testing.assert_allclose(opp[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(opp[:, 2], 0.0, atol=1e-12)

    def test_neutral_luminance_is_monotonic(self):
        """Neutral ramp should have monotonically increasing Y."""
        vals = np.linspace(0, 1, 11)
        neutral = np.column_stack([vals, vals, vals])
        opp = rgb_to_opponent(neutral)
        y_channel = opp[:, 0]
        assert np.all(np.diff(y_channel) > 0)


class TestNeutralChromaBoost:
    """Tests for compute_neutral_chroma_boost."""

    def test_output_shape(self):
        """Output should be (N^3,) for any N."""
        for N in [3, 5, 9]:
            boost = compute_neutral_chroma_boost(N)
            assert boost.shape == (N ** 3,)

    def test_all_at_least_one(self):
        """All boost values should be >= 1.0."""
        boost = compute_neutral_chroma_boost(9)
        assert np.all(boost >= 1.0)

    def test_neutral_nodes_have_max_boost(self):
        """Nodes on the neutral axis (r==g==b) should have highest boost."""
        N = 9
        boost = compute_neutral_chroma_boost(N, k=3.0, sigma=0.12)
        max_boost = 1.0 + 3.0  # at zero distance from neutral

        # Check diagonal nodes
        for i in range(N):
            flat_idx = i * N * N + i * N + i  # b=i, g=i, r=i
            # These nodes are exactly on the neutral axis
            np.testing.assert_allclose(boost[flat_idx], max_boost, atol=1e-10)

    def test_far_nodes_near_one(self):
        """Nodes far from neutral should have boost close to 1.0."""
        N = 9
        boost = compute_neutral_chroma_boost(N, k=3.0, sigma=0.12)

        # Corner (1,0,0) = pure red, far from neutral
        # flat = b*N*N + g*N + r, so r=N-1, g=0, b=0 -> flat = 0*N*N + 0*N + (N-1) = N-1
        idx_red = N - 1  # r=N-1, g=0, b=0
        assert boost[idx_red] < 1.05  # very close to 1.0

    def test_k_zero_gives_all_ones(self):
        """k=0 should give uniform boost of 1.0."""
        boost = compute_neutral_chroma_boost(5, k=0.0, sigma=0.12)
        np.testing.assert_allclose(boost, 1.0, atol=1e-12)

    def test_larger_sigma_gives_broader_boost(self):
        """Wider sigma should give higher boost at off-neutral nodes."""
        N = 9
        narrow = compute_neutral_chroma_boost(N, k=3.0, sigma=0.05)
        broad = compute_neutral_chroma_boost(N, k=3.0, sigma=0.5)

        # Off-neutral node: broad sigma should be closer to max
        off_neutral_idx = 1  # r=1, g=0, b=0
        assert broad[off_neutral_idx] > narrow[off_neutral_idx]
