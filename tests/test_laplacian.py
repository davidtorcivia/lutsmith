"""Tests for 3D discrete Laplacian construction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from lutsmith.core.laplacian import (
    build_laplacian,
    build_laplacian_extended,
    build_laplacian_vectorized,
)


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
        from lutsmith.core.types import flat_index
        idx = flat_index(2, 2, 2, 5)
        assert L[idx, idx] == 6.0

    def test_corner_node_degree(self, build_fn):
        """Corner node should have degree 3."""
        L = build_fn(5)
        from lutsmith.core.types import flat_index
        idx = flat_index(0, 0, 0, 5)
        assert L[idx, idx] == 3.0

    def test_edge_node_degree(self, build_fn):
        """Edge node (not corner, not face) should have degree 4."""
        L = build_fn(5)
        from lutsmith.core.types import flat_index
        # (1, 0, 0) is on an edge
        idx = flat_index(1, 0, 0, 5)
        assert L[idx, idx] == 4.0

    def test_face_node_degree(self, build_fn):
        """Face center node should have degree 5."""
        L = build_fn(5)
        from lutsmith.core.types import flat_index
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


class TestLaplacianExtended:
    """Tests for extended connectivity (18 and 26)."""

    @pytest.fixture(params=[18, 26])
    def connectivity(self, request):
        return request.param

    def test_symmetric(self, connectivity):
        """Extended Laplacian should be symmetric."""
        L = build_laplacian_extended(5, connectivity)
        diff = L - L.T
        assert sparse.linalg.norm(diff) < 1e-10

    def test_row_sums_zero(self, connectivity):
        """Each row should sum to zero."""
        L = build_laplacian_extended(5, connectivity)
        row_sums = np.array(L.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_constant_in_null_space(self, connectivity):
        """L * constant_vector = 0."""
        L = build_laplacian_extended(5, connectivity)
        ones = np.ones(5 ** 3)
        result = L @ ones
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_shape(self, connectivity):
        """Matrix should be N^3 x N^3."""
        for N in [3, 5]:
            L = build_laplacian_extended(N, connectivity)
            expected = N ** 3
            assert L.shape == (expected, expected)

    def test_positive_diagonal(self, connectivity):
        """Diagonal entries should be non-negative."""
        L = build_laplacian_extended(5, connectivity)
        diag = L.diagonal()
        assert np.all(diag >= -1e-10)

    def test_interior_degree_normalized_to_six(self, connectivity):
        """Interior node diagonal should be normalized to 6.0."""
        N = 5
        L = build_laplacian_extended(N, connectivity)
        from lutsmith.core.types import flat_index
        idx = flat_index(2, 2, 2, N)
        np.testing.assert_allclose(L[idx, idx], 6.0, atol=1e-10)

    def test_more_neighbors_than_6connected(self, connectivity):
        """Extended connectivity should have more non-zero entries per row."""
        N = 5
        L_6 = build_laplacian_vectorized(N)
        L_ext = build_laplacian_extended(N, connectivity)
        # Extended should have more non-zeros
        assert L_ext.nnz > L_6.nnz

    def test_connectivity_6_delegates_to_vectorized(self):
        """connectivity=6 should produce identical result to build_laplacian_vectorized."""
        N = 5
        L_6 = build_laplacian_vectorized(N)
        L_ext = build_laplacian_extended(N, 6)
        diff = L_6 - L_ext
        assert sparse.linalg.norm(diff) < 1e-10

    def test_invalid_connectivity_raises(self):
        """Unsupported connectivity should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported connectivity"):
            build_laplacian_extended(5, 12)

    def test_26_has_more_entries_than_18(self):
        """26-connected should have more non-zeros than 18-connected."""
        N = 5
        L_18 = build_laplacian_extended(N, 18)
        L_26 = build_laplacian_extended(N, 26)
        assert L_26.nnz > L_18.nnz
