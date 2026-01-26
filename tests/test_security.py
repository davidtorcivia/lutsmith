"""Security tests for input validation and boundary conditions."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from chromaforge.config import MAX_CUBE_SIZE, MAX_IMAGE_DIMENSION
from chromaforge.errors import (
    ImageError,
    ImageDimensionError,
    ImageFormatError,
    LUTFormatError,
    ExportError,
)


class TestInputPathValidation:
    """Tests for input path security."""

    def test_nonexistent_path(self):
        """Non-existent file should raise."""
        from chromaforge.io.image import validate_input_path
        with pytest.raises(ImageError):
            validate_input_path("/completely/bogus/path.png")

    def test_invalid_extension(self, tmp_path):
        """Disallowed extension should raise."""
        from chromaforge.io.image import validate_input_path
        p = tmp_path / "malicious.exe"
        p.write_text("not an image")
        with pytest.raises(ImageFormatError):
            validate_input_path(p)

    def test_valid_extension_accepted(self, tmp_path):
        """Valid image extension should pass."""
        from chromaforge.io.image import validate_input_path
        p = tmp_path / "test.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes
        # Should not raise (validates path, not contents)
        result = validate_input_path(p)
        assert result.exists()


class TestOutputPathValidation:
    """Tests for output path security."""

    def test_parent_must_exist(self):
        """Output path with non-existent parent should raise."""
        from chromaforge.io.image import validate_output_path
        with pytest.raises(ImageError):
            validate_output_path("/nonexistent/directory/output.png")


class TestImageDimensions:
    """Tests for image dimension limits."""

    def test_sanitize_nan(self):
        """NaN values should be replaced with 0."""
        from chromaforge.pipeline.preprocess import sanitize_image

        img = np.array([[[float("nan"), 0.5, 0.5]]], dtype=np.float32)
        result = sanitize_image(img)
        assert np.all(np.isfinite(result))

    def test_sanitize_inf(self):
        """Inf values should be clamped."""
        from chromaforge.pipeline.preprocess import sanitize_image

        img = np.array([[[float("inf"), 0.5, -float("inf")]]], dtype=np.float32)
        result = sanitize_image(img)
        assert np.all(np.isfinite(result))

    def test_sanitize_negative(self):
        """Negative values should be preserved."""
        from chromaforge.pipeline.preprocess import sanitize_image

        img = np.array([[[-0.5, 0.5, 0.5]]], dtype=np.float32)
        result = sanitize_image(img)
        assert result.min() < 0.0
        assert np.isclose(result.min(), -0.5)


class TestCubeFileSecurity:
    """Tests for .cube file parsing security."""

    def test_oversized_lut_rejected(self, tmp_path):
        """LUT size exceeding MAX_CUBE_SIZE should be rejected."""
        from chromaforge.io.cube import read_cube

        p = tmp_path / "huge.cube"
        p.write_text(f"LUT_3D_SIZE {MAX_CUBE_SIZE + 10}\n")
        with pytest.raises(LUTFormatError, match="out of range"):
            read_cube(p)

    def test_nan_in_cube_rejected(self, tmp_path):
        """NaN values in .cube data should be rejected."""
        from chromaforge.io.cube import read_cube

        p = tmp_path / "nan.cube"
        lines = ["LUT_3D_SIZE 2\n"]
        for i in range(8):
            if i == 0:
                lines.append("nan 0.0 0.0\n")
            else:
                lines.append("0.5 0.5 0.5\n")
        p.write_text("".join(lines))
        with pytest.raises(LUTFormatError, match="Non-finite"):
            read_cube(p)

    def test_cube_write_size_limit(self, tmp_path):
        """Writing LUT larger than MAX_CUBE_SIZE should raise."""
        from chromaforge.io.cube import write_cube

        N = MAX_CUBE_SIZE + 1
        lut = np.zeros((N, N, N, 3), dtype=np.float32)
        with pytest.raises(ExportError):
            write_cube(tmp_path / "huge.cube", lut)


class TestNaNPropagation:
    """Tests that NaN/Inf doesn't silently propagate through the pipeline."""

    def test_interpolation_clamped_input(self):
        """Interpolation should handle out-of-range input gracefully."""
        from chromaforge.core.interpolation import apply_lut_to_colors
        from chromaforge.core.lut import identity_lut

        N = 5
        lut = identity_lut(N)
        # Input slightly outside [0, 1]
        colors = np.array([[1.5, -0.5, 0.5], [0.0, 0.0, 2.0]], dtype=np.float32)
        result = apply_lut_to_colors(colors, lut, N, "tetrahedral")
        assert np.all(np.isfinite(result))

    def test_solver_handles_edge_cases(self):
        """Solver should produce finite output even with minimal data."""
        from chromaforge.core.solver import solve_per_channel

        rng = np.random.default_rng(42)
        N = 3
        M = 10
        input_rgb = rng.random((M, 3), dtype=np.float32)
        output_rgb = rng.random((M, 3), dtype=np.float32)
        alpha = np.ones(M, dtype=np.float32)

        lut, infos = solve_per_channel(
            input_rgb, output_rgb, alpha,
            N=N,
            lambda_s=1.0,  # High smoothness for stability
            lambda_r=0.1,  # Strong prior
            loss="l2",
            irls_iterations=0,
            parallel=False,
        )

        assert np.all(np.isfinite(lut))


class TestLUTIndexingConsistency:
    """Tests that indexing convention is consistent throughout."""

    def test_flat_index_inverse(self):
        """flat_index and grid_indices should be inverses."""
        from chromaforge.core.types import flat_index, grid_indices

        N = 7
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    flat = flat_index(r, g, b, N)
                    r2, g2, b2 = grid_indices(flat, N)
                    assert (r, g, b) == (r2, g2, b2), (
                        f"Roundtrip failed: ({r},{g},{b}) -> {flat} -> ({r2},{g2},{b2})"
                    )

    def test_flat_index_uniqueness(self):
        """Every (r,g,b) should map to a unique flat index."""
        from chromaforge.core.types import flat_index

        N = 5
        indices = set()
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    idx = flat_index(r, g, b, N)
                    assert idx not in indices, f"Duplicate flat index {idx}"
                    indices.add(idx)
        assert len(indices) == N ** 3

    def test_flat_index_range(self):
        """All flat indices should be in [0, N^3)."""
        from chromaforge.core.types import flat_index

        N = 9
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    idx = flat_index(r, g, b, N)
                    assert 0 <= idx < N ** 3
