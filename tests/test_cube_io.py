"""Tests for .cube file read/write round-trip and format correctness."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from lutsmith.core.lut import identity_lut
from lutsmith.io.cube import write_cube, read_cube
from lutsmith.errors import LUTFormatError


class TestCubeRoundTrip:
    """Tests for write -> read round-trip integrity."""

    def test_identity_roundtrip(self, tmp_cube_path):
        """Identity LUT should survive write/read unchanged."""
        N = 5
        lut = identity_lut(N)
        write_cube(tmp_cube_path, lut, title="Identity Test")
        lut_read, meta = read_cube(tmp_cube_path)

        assert meta["size"] == N
        assert meta["title"] == "Identity Test"
        np.testing.assert_allclose(lut_read, lut, atol=1e-5)

    def test_random_roundtrip(self, tmp_cube_path):
        """Random LUT should survive write/read with minimal precision loss."""
        N = 9
        rng = np.random.default_rng(42)
        lut = rng.random((N, N, N, 3), dtype=np.float32)
        write_cube(tmp_cube_path, lut, title="Random LUT")
        lut_read, meta = read_cube(tmp_cube_path)

        assert meta["size"] == N
        # .cube has 6 decimal places, so atol ~1e-6
        np.testing.assert_allclose(lut_read, lut, atol=2e-6)

    def test_domain_roundtrip(self, tmp_cube_path):
        """Domain min/max should be preserved."""
        N = 5
        lut = identity_lut(N)
        dmin = (0.0, 0.0, 0.0)
        dmax = (1.0, 1.0, 1.0)
        write_cube(tmp_cube_path, lut, domain_min=dmin, domain_max=dmax)
        _, meta = read_cube(tmp_cube_path)

        np.testing.assert_allclose(meta["domain_min"], list(dmin))
        np.testing.assert_allclose(meta["domain_max"], list(dmax))

    def test_shaper_roundtrip(self, tmp_cube_path):
        """1D shaper data should round-trip when present."""
        N = 3
        lut = identity_lut(N)
        shaper = np.linspace(0.0, 1.0, 4, dtype=np.float32)
        shaper = np.stack([shaper, shaper, shaper], axis=-1)

        write_cube(tmp_cube_path, lut, shaper=shaper)
        lut_read, meta = read_cube(tmp_cube_path)

        np.testing.assert_allclose(lut_read, lut, atol=1e-5)
        assert meta["shaper_1d"] is not None
        np.testing.assert_allclose(meta["shaper_1d"], shaper, atol=1e-6)


class TestCubeOrdering:
    """Tests for correct R-fastest data ordering."""

    def test_ordering_correctness(self, tmp_cube_path):
        """Verify .cube file ordering: R varies fastest."""
        N = 3
        # Create a LUT where each node has a unique value encoding its position
        lut = np.zeros((N, N, N, 3), dtype=np.float32)
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    lut[r, g, b, 0] = r / (N - 1)
                    lut[r, g, b, 1] = g / (N - 1)
                    lut[r, g, b, 2] = b / (N - 1)

        write_cube(tmp_cube_path, lut)
        lut_read, _ = read_cube(tmp_cube_path)
        np.testing.assert_allclose(lut_read, lut, atol=1e-5)


class TestCubeMalformed:
    """Tests for malformed .cube file rejection."""

    def test_missing_size(self, tmp_path):
        """File without LUT_3D_SIZE should raise."""
        p = tmp_path / "no_size.cube"
        p.write_text("TITLE \"test\"\n0.0 0.0 0.0\n")
        with pytest.raises(LUTFormatError, match="No LUT_3D_SIZE"):
            read_cube(p)

    def test_wrong_data_count(self, tmp_path):
        """File with wrong number of data lines should raise."""
        p = tmp_path / "wrong_count.cube"
        lines = ["LUT_3D_SIZE 2\n"]
        # Need 8 entries but provide only 4
        for _ in range(4):
            lines.append("0.5 0.5 0.5\n")
        p.write_text("".join(lines))
        with pytest.raises(LUTFormatError, match="Expected 8"):
            read_cube(p)

    def test_nan_rejection(self, tmp_path):
        """NaN values in data should raise."""
        p = tmp_path / "nan.cube"
        lines = ["LUT_3D_SIZE 2\n"]
        for i in range(8):
            if i == 3:
                lines.append("nan 0.5 0.5\n")
            else:
                lines.append("0.5 0.5 0.5\n")
        p.write_text("".join(lines))
        with pytest.raises(LUTFormatError, match="Non-finite"):
            read_cube(p)

    def test_inf_rejection(self, tmp_path):
        """Inf values in data should raise."""
        p = tmp_path / "inf.cube"
        lines = ["LUT_3D_SIZE 2\n"]
        for i in range(8):
            if i == 5:
                lines.append("0.5 inf 0.5\n")
            else:
                lines.append("0.5 0.5 0.5\n")
        p.write_text("".join(lines))
        with pytest.raises(LUTFormatError, match="Non-finite"):
            read_cube(p)

    def test_size_too_large(self, tmp_path):
        """Size exceeding MAX_CUBE_SIZE should raise."""
        p = tmp_path / "toobig.cube"
        p.write_text("LUT_3D_SIZE 999\n")
        with pytest.raises(LUTFormatError, match="out of range"):
            read_cube(p)

    def test_file_not_found(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_cube("/nonexistent/path/test.cube")
