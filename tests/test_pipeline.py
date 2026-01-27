"""Tests for the full pipeline runner."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from lutsmith.core.types import PipelineConfig, ExportFormat
from lutsmith.errors import PipelineError, PipelineCancelledError


class TestPipelineRunner:
    """Tests for run_pipeline end-to-end."""

    def test_full_pipeline_synthetic(self, sample_image_pair, tmp_path):
        """Full pipeline on synthetic image pair should succeed."""
        from lutsmith.pipeline.runner import run_pipeline

        src_path, tgt_path = sample_image_pair
        out_path = tmp_path / "output.cube"

        config = PipelineConfig(
            source_path=src_path,
            target_path=tgt_path,
            output_path=out_path,
            lut_size=5,  # Small for speed
            smoothness=0.1,
            prior_strength=0.01,
            bin_resolution=8,
            min_samples_per_bin=2,
            irls_iterations=1,
            enable_refinement=False,
            format=ExportFormat.CUBE,
        )

        result = run_pipeline(config)

        assert result.lut.array.shape == (5, 5, 5, 3)
        assert result.metrics.mean_delta_e >= 0
        assert out_path.exists()
        assert result.output_path == out_path

    def test_pipeline_no_paths_raises(self, tmp_path):
        """Pipeline without source/target paths should raise."""
        from lutsmith.pipeline.runner import run_pipeline

        config = PipelineConfig(output_path=tmp_path / "out.cube")
        with pytest.raises(PipelineError, match="Source and target"):
            run_pipeline(config)

    def test_pipeline_progress_callback(self, sample_image_pair, tmp_path):
        """Progress callback should be called with valid stages."""
        from lutsmith.pipeline.runner import run_pipeline

        src_path, tgt_path = sample_image_pair
        out_path = tmp_path / "output.cube"

        stages_seen = []

        def on_progress(stage, fraction, message):
            stages_seen.append(stage)

        config = PipelineConfig(
            source_path=src_path,
            target_path=tgt_path,
            output_path=out_path,
            lut_size=5,
            bin_resolution=8,
            min_samples_per_bin=2,
            irls_iterations=0,
            enable_refinement=False,
        )

        run_pipeline(config, progress_callback=on_progress)

        # Should have seen at least preprocess, sampling, solving
        assert "preprocess" in stages_seen
        assert "sampling" in stages_seen or "sample" in stages_seen
        assert "solving" in stages_seen or "solve" in stages_seen

    def test_pipeline_cancellation(self, sample_image_pair, tmp_path):
        """Pipeline should respect cancellation."""
        from lutsmith.pipeline.runner import run_pipeline

        src_path, tgt_path = sample_image_pair

        config = PipelineConfig(
            source_path=src_path,
            target_path=tgt_path,
            output_path=tmp_path / "out.cube",
            lut_size=5,
            bin_resolution=8,
        )

        # Cancel immediately
        with pytest.raises(PipelineCancelledError):
            run_pipeline(config, cancel_check=lambda: True)

    def test_pipeline_result_has_diagnostics(self, sample_image_pair, tmp_path):
        """Result should contain timing diagnostics."""
        from lutsmith.pipeline.runner import run_pipeline

        src_path, tgt_path = sample_image_pair

        config = PipelineConfig(
            source_path=src_path,
            target_path=tgt_path,
            output_path=tmp_path / "out.cube",
            lut_size=5,
            bin_resolution=8,
            min_samples_per_bin=2,
            irls_iterations=0,
            enable_refinement=False,
        )

        result = run_pipeline(config)
        assert "total_time" in result.diagnostics
        assert result.diagnostics["total_time"] > 0
