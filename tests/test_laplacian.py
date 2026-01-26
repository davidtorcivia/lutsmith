"""Tests for 3D discrete Laplacian construction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from chromaforge.core.laplacian import build_laplacian, build_laplacian_vectorized


@pytest.fixture(params=["loop", "vectorized"])
def build_fn(request):
    """Test both Laplacian construction methods."""
    if request.param == "loop":
        return build_laplacian
    return build_laplacian_vectorized


class TestLaplacian:
    """Tests for Laplacian matrix properties."""

    def test_symmetric(self, build_fn):
        """Laplacian should be symmetric."""
        L = build_fn(5)
        diff = L - L.T
        assert sparse.linalg.norm(diff) < 1e-10

    def test_row_sums_zero(self, build_fn):
        """Each row should sum to zero."""
        L = build_fn(5)
        row_sums = np.array(L.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_constant_in_null_space(self, build_fn):
        """L * constant_vector = 0."""
        L = build_fn(5)
        N3 = 5 ** 3
        ones = np.ones(N3)
        result = L @ ones
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_shape(self, build_fn):
        """Matrix should be N^3 x N^3."""
        for N in [3, 5, 7]:
            L = build_fn(N)
            expected = N ** 3
            assert L.shape == (expected, expected)

    def test_positive_diagonal(self, build_fn):
        """Diagonal entries should be positive (= number of neighbors)."""
        L = build_fn(5)
        diag = L.diagonal()
        assert np.all(diag >= 0)
        # Interior nodes should have 6 neighbors
        # Corner nodes should have 3 neighbors

    def test_negative_off_diagonal(self, build_fn):
        """Off-diagonal entries should be -1 or 0."""
        L = build_fn(4)
        dense = L.toarray()
        np.fill_diagonal(dense, 0)
        # Non-zero off-diag should be -1
        nonzero = dense[dense != 0]
        np.testing.assert_allclose(nonzero, -1.0)

    def test_interior_node_degree(self, build_fn):
        """Interior node should have degree 6 (6-connected)."""
        L = build_fn(5)
        # Node (2,2,2) in a 5^3 grid is interior
        from chromaforge.core.types import flat_index
        idx = flat_index(2, 2, 2, 5)
        assert L[idx, idx] == 6.0

    def test_corner_node_degree(self, build_fn):
        """Corner node should have degree 3."""
        L = build_fn(5)
        from chromaforge.core.types import flat_index
        idx = flat_index(0, 0, 0, 5)
        assert L[idx, idx] == 3.0

    def test_edge_node_degree(self, build_fn):
        """Edge node (not corner, not face) should have degree 4."""
        L = build_fn(5)
        from chromaforge.core.types import flat_index
        # (1, 0, 0) is on an edge
        idx = flat_index(1, 0, 0, 5)
        assert L[idx, idx] == 4.0

    def test_face_node_degree(self, build_fn):
        """Face center node should have degree 5."""
        L = build_fn(5)
        from chromaforge.core.types import flat_index
        # (2, 2, 0) is on a face
        idx = flat_index(2, 2, 0, 5)
        assert L[idx, idx] == 5.0

    def test_both_implementations_agree(self):
        """Loop and vectorized implementations should produce identical results."""
        N = 5
        L_loop = build_laplacian(N)
        L_vec = build_laplacian_vectorized(N)
        diff = L_loop - L_vec
        assert sparse.linalg.norm(diff) < 1e-10
