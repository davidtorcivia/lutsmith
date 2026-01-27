"""Tests for pixel binning and aggregation."""

from __future__ import annotations

import numpy as np
import pytest


class TestBinPixels:
    """Tests for the binning implementation."""

    def _make_coords(self, M):
        """Create dummy pixel coordinates."""
        return np.zeros(M, dtype=np.float32), np.zeros(M, dtype=np.float32)

    def test_count_conservation(self):
        """Total counts across all bins should equal number of pixels."""
        from lutsmith._numba_kernels.binning import bin_pixels_numpy

        rng = np.random.default_rng(42)
        M = 1000
        source = rng.random((M, 3), dtype=np.float32)
        target = rng.random((M, 3), dtype=np.float32)
        bin_res = 8
        px, py = self._make_coords(M)

        counts, sum_input, mean_output, m2_output, sx, sy = bin_pixels_numpy(
            source, target, bin_res, px, py
        )

        total_counted = counts.sum()
        assert total_counted == M

    def test_mean_correctness(self):
        """For a single bin, mean should match numpy.mean."""
        rng = np.random.default_rng(42)
        M = 100
        # Put all pixels in one bin by making them all very similar
        source = np.full((M, 3), 0.5, dtype=np.float32) + rng.random((M, 3), dtype=np.float32) * 0.01
        target = rng.random((M, 3), dtype=np.float32)
        bin_res = 4
        px, py = self._make_coords(M)

        from lutsmith._numba_kernels.binning import bin_pixels_numpy
        counts, sum_input, mean_output, m2_output, sx, sy = bin_pixels_numpy(
            source, target, bin_res, px, py
        )

        # Find the bin with count == M (or most)
        max_count_idx = np.argmax(counts)
        assert counts[max_count_idx] == M

        # Mean output should match direct calculation
        np.testing.assert_allclose(
            mean_output[max_count_idx],
            target.mean(axis=0),
            atol=1e-4,
        )

    def test_bin_resolution(self):
        """Number of bins should be bin_res^3."""
        from lutsmith._numba_kernels.binning import bin_pixels_numpy
        bin_res = 8
        M = 100
        source = np.random.default_rng(42).random((M, 3), dtype=np.float32)
        target = source.copy()
        px, py = self._make_coords(M)

        counts, _, _, _, _, _ = bin_pixels_numpy(source, target, bin_res, px, py)
        assert len(counts) == bin_res ** 3

    def test_numba_vs_numpy_agreement(self):
        """Numba and NumPy implementations should produce identical results."""
        try:
            from lutsmith._numba_kernels.binning import (
                bin_pixels,
                bin_pixels_numpy,
            )
            import numba  # noqa: F401
        except ImportError:
            pytest.skip("Numba not available")

        rng = np.random.default_rng(42)
        M = 500
        source = rng.random((M, 3), dtype=np.float32)
        target = rng.random((M, 3), dtype=np.float32)
        bin_res = 8

        np_result = bin_pixels_numpy(
            source, target, bin_res,
            np.zeros(M, dtype=np.float32), np.zeros(M, dtype=np.float32),
        )
        nb_result = bin_pixels(source, target, bin_res)

        np_counts, np_sum_in, np_mean_out, np_m2, _, _ = np_result
        nb_counts, nb_sum_in, nb_mean_out, nb_m2, _, _ = nb_result

        np.testing.assert_array_equal(np_counts, nb_counts)
        np.testing.assert_allclose(np_mean_out, nb_mean_out, atol=1e-4)


class TestSampling:
    """Tests for sampling.py bin aggregation."""

    def test_bins_to_samples(self):
        """bins_to_samples should produce arrays of correct shape."""
        from lutsmith.pipeline.sampling import (
            bin_and_aggregate,
            bins_to_samples,
            compute_sample_weights,
        )
        from lutsmith.core.types import PipelineConfig

        rng = np.random.default_rng(42)
        H, W = 32, 32
        source = rng.random((H, W, 3), dtype=np.float32)
        target = rng.random((H, W, 3), dtype=np.float32)

        config = PipelineConfig(bin_resolution=8, min_samples_per_bin=3)
        bins, occupied = bin_and_aggregate(source, target, config)

        assert len(bins) > 0
        assert len(occupied) == len(bins)

        weights = compute_sample_weights(bins)
        input_rgb, output_rgb, alpha = bins_to_samples(bins, weights)

        assert input_rgb.shape[1] == 3
        assert output_rgb.shape[1] == 3
        assert len(alpha) == len(input_rgb)
        assert len(alpha) > 0
