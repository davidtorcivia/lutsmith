"""Tests for Hald CLUT identity generation and reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from chromaforge.hald.identity import (
    generate_hald_identity,
    hald_image_size,
    hald_lut_size,
)
from chromaforge.hald.reconstruct import reconstruct_from_hald
from chromaforge.hald.resample import resample_lut


class TestHaldIdentity:
    """Tests for Hald identity image generation."""

    def test_image_size(self):
        """Level 8 should give 512x512 image."""
        assert hald_image_size(8) == 512

    def test_lut_size(self):
        """Level 8 should give 64^3 LUT."""
        assert hald_lut_size(8) == 64

    def test_identity_shape(self):
        """Generated identity image should have correct shape."""
        level = 4  # Small for fast test
        img = generate_hald_identity(level)
        size = hald_image_size(level)
        assert img.shape == (size, size, 3)

    def test_identity_range(self):
        """All values should be in [0, 1]."""
        img = generate_hald_identity(4)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_identity_dtype(self):
        """Image should be float32."""
        img = generate_hald_identity(4)
        assert img.dtype == np.float32


class TestHaldRoundTrip:
    """Tests for Hald identity -> reconstruct round-trip."""

    def test_identity_roundtrip(self):
        """Reconstructing from identity should give identity LUT."""
        level = 4
        lut_size = hald_lut_size(level)
        identity_img = generate_hald_identity(level)
        lut = reconstruct_from_hald(identity_img, level)

        assert lut.shape == (lut_size, lut_size, lut_size, 3)

        # Check a few known values
        # (0,0,0) should map to (0,0,0)
        np.testing.assert_allclose(lut[0, 0, 0], [0, 0, 0], atol=1e-3)

        # (N-1, N-1, N-1) should map to (1,1,1)
        np.testing.assert_allclose(
            lut[lut_size - 1, lut_size - 1, lut_size - 1],
            [1, 1, 1],
            atol=1e-3,
        )

    def test_known_transform(self):
        """Apply a known gain to Hald identity and reconstruct."""
        level = 4
        lut_size = hald_lut_size(level)
        identity_img = generate_hald_identity(level)

        # Apply simple gain: 1.1x red channel
        processed = identity_img.copy()
        processed[:, :, 0] = np.clip(processed[:, :, 0] * 1.1, 0, 1)

        lut = reconstruct_from_hald(processed, level)

        # Check mid-gray: lut[N/2, N/2, N/2, 0] should be ~0.5*1.1
        mid = lut_size // 2
        expected_r = min(mid / (lut_size - 1) * 1.1, 1.0)
        assert abs(lut[mid, mid, mid, 0] - expected_r) < 0.05

    def test_reconstruct_shape(self):
        """Reconstructed LUT should have correct shape."""
        level = 4
        lut_size = hald_lut_size(level)
        img = generate_hald_identity(level)
        lut = reconstruct_from_hald(img, level)
        assert lut.shape == (lut_size, lut_size, lut_size, 3)
        assert lut.dtype == np.float32


class TestResample:
    """Tests for LUT resampling between grid sizes."""

    def test_same_size_copy(self):
        """Resampling to same size should return a copy."""
        from chromaforge.core.lut import identity_lut
        lut = identity_lut(9)
        resampled = resample_lut(lut, 9)
        np.testing.assert_array_equal(lut, resampled)
        # Should be a separate copy
        assert lut is not resampled

    def test_resample_preserves_range(self):
        """Resampled identity LUT should stay in [0, 1]."""
        from chromaforge.core.lut import identity_lut
        lut = identity_lut(9)
        resampled = resample_lut(lut, 17)
        assert resampled.min() >= -0.01
        assert resampled.max() <= 1.01

    def test_resample_shape(self):
        """Output shape should match target size."""
        from chromaforge.core.lut import identity_lut
        lut = identity_lut(9)
        resampled = resample_lut(lut, 17)
        assert resampled.shape == (17, 17, 17, 3)

    def test_resample_identity_fidelity(self):
        """Resampled identity LUT should remain close to identity."""
        from chromaforge.core.lut import identity_lut
        src = identity_lut(17)
        resampled = resample_lut(src, 33)
        target = identity_lut(33)
        np.testing.assert_allclose(resampled, target, atol=1e-3)
