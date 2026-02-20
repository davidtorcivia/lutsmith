"""Tests for multi-pair LUT extraction workflow."""

from __future__ import annotations

import numpy as np

from lutsmith.cli.app import _parse_pair_manifest
from lutsmith.pipeline.batch_manifest import ManifestEntry
from lutsmith.core.types import ExportFormat, PipelineConfig


def test_parse_pair_manifest_resolves_relative_paths(tmp_path):
    """Manifest parser should ignore comments and resolve relative paths."""
    src1 = tmp_path / "a_src.png"
    tgt1 = tmp_path / "a_tgt.png"
    src2 = tmp_path / "b_src.png"
    tgt2 = tmp_path / "b_tgt.png"
    for p in (src1, tgt1, src2, tgt2):
        p.write_bytes(b"")

    manifest = tmp_path / "pairs.csv"
    manifest.write_text(
        "# source,target pairs\n"
        "a_src.png,a_tgt.png\n"
        "\n"
        "b_src.png,b_tgt.png\n",
        encoding="utf-8",
    )

    pairs = _parse_pair_manifest(manifest)

    assert len(pairs) == 2
    assert pairs[0] == ManifestEntry(source=src1.resolve(), target=tgt1.resolve(), weight=1.0)
    assert pairs[1] == ManifestEntry(source=src2.resolve(), target=tgt2.resolve(), weight=1.0)


def test_parse_pair_manifest_with_header_and_weight(tmp_path):
    """Manifest parser should support source,target,weight format."""
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    src.write_bytes(b"")
    tgt.write_bytes(b"")

    manifest = tmp_path / "pairs_weighted.csv"
    manifest.write_text(
        "source,target,weight\n"
        "src.png,tgt.png,2.5\n",
        encoding="utf-8",
    )

    pairs = _parse_pair_manifest(manifest)
    assert pairs == [ManifestEntry(source=src.resolve(), target=tgt.resolve(), weight=2.5)]


def test_parse_pair_manifest_with_transfer_and_normalization(tmp_path):
    """Manifest parser should accept transfer_fn and normalization columns."""
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    src.write_bytes(b"")
    tgt.write_bytes(b"")

    manifest = tmp_path / "pairs_full.csv"
    manifest.write_text(
        "source,target,weight,cluster,transfer_fn,normalization\n"
        "src.png,tgt.png,1.0,scene_a,log_c4,rgb_affine\n",
        encoding="utf-8",
    )

    entries = _parse_pair_manifest(manifest)
    assert len(entries) == 1
    assert entries[0] == ManifestEntry(
        source=src.resolve(),
        target=tgt.resolve(),
        weight=1.0,
        cluster="scene_a",
        transfer_fn="log_c4",
        normalization="rgb_affine",
    )


def test_run_multi_pipeline_succeeds(tmp_image_dir, tmp_path):
    """Multi-pair extraction should produce a LUT and diagnostics."""
    from lutsmith.io.image import save_image
    from lutsmith.pipeline.runner import run_multi_pipeline

    rng = np.random.default_rng(123)
    source1 = rng.random((64, 64, 3), dtype=np.float32)
    source2 = rng.random((64, 64, 3), dtype=np.float32)

    # Keep transform consistent across pairs to emulate shared look.
    target1 = np.clip(source1 * 1.08 + 0.02, 0, 1).astype(np.float32)
    target2 = np.clip(source2 * 1.08 + 0.02, 0, 1).astype(np.float32)

    src1 = tmp_image_dir / "src1.png"
    tgt1 = tmp_image_dir / "tgt1.png"
    src2 = tmp_image_dir / "src2.png"
    tgt2 = tmp_image_dir / "tgt2.png"
    save_image(source1, src1, bit_depth=8)
    save_image(target1, tgt1, bit_depth=8)
    save_image(source2, src2, bit_depth=8)
    save_image(target2, tgt2, bit_depth=8)

    out_path = tmp_path / "batch_output.cube"
    config = PipelineConfig(
        output_path=out_path,
        lut_size=5,
        smoothness=0.1,
        prior_strength=0.01,
        bin_resolution=8,
        min_samples_per_bin=2,
        irls_iterations=1,
        enable_refinement=False,
        format=ExportFormat.CUBE,
    )

    result = run_multi_pipeline([(src1, tgt1), (src2, tgt2)], config)

    assert result.lut.array.shape == (5, 5, 5, 3)
    assert result.metrics.mean_delta_e >= 0
    assert out_path.exists()
    assert result.diagnostics["num_pairs"] == 2
    assert result.output_path == out_path


def test_run_multi_pipeline_records_pair_weight_shares(tmp_image_dir, tmp_path):
    """Pair weighting should be reflected in diagnostics weight shares."""
    from lutsmith.io.image import save_image
    from lutsmith.pipeline.runner import run_multi_pipeline

    rng = np.random.default_rng(321)
    source1 = rng.random((64, 64, 3), dtype=np.float32)
    source2 = rng.random((64, 64, 3), dtype=np.float32)
    target1 = np.clip(source1 * 1.05 + 0.01, 0, 1).astype(np.float32)
    target2 = np.clip(source2 * 1.05 + 0.01, 0, 1).astype(np.float32)

    src1 = tmp_image_dir / "w_src1.png"
    tgt1 = tmp_image_dir / "w_tgt1.png"
    src2 = tmp_image_dir / "w_src2.png"
    tgt2 = tmp_image_dir / "w_tgt2.png"
    save_image(source1, src1, bit_depth=8)
    save_image(target1, tgt1, bit_depth=8)
    save_image(source2, src2, bit_depth=8)
    save_image(target2, tgt2, bit_depth=8)

    out_path = tmp_path / "batch_weighted.cube"
    config = PipelineConfig(
        output_path=out_path,
        lut_size=5,
        bin_resolution=8,
        min_samples_per_bin=2,
        irls_iterations=1,
        format=ExportFormat.CUBE,
    )

    result = run_multi_pipeline(
        [(src1, tgt1), (src2, tgt2)],
        config,
        pair_weights=[1.0, 3.0],
        pair_balance="equal",
    )

    pairs = result.diagnostics["pairs"]
    assert len(pairs) == 2
    assert pairs[1]["weight_share"] > pairs[0]["weight_share"]


def test_run_multi_pipeline_outlier_rejection(tmp_image_dir, tmp_path):
    """Outlier-pair rejection should drop strongly inconsistent pairs."""
    from lutsmith.io.image import save_image
    from lutsmith.pipeline.runner import run_multi_pipeline

    rng = np.random.default_rng(777)
    source1 = rng.random((64, 64, 3), dtype=np.float32)
    source2 = rng.random((64, 64, 3), dtype=np.float32)
    source3 = rng.random((64, 64, 3), dtype=np.float32)

    # Two consistent pairs, one strong outlier.
    target1 = np.clip(source1 * 1.08 + 0.02, 0, 1).astype(np.float32)
    target2 = np.clip(source2 * 1.08 + 0.02, 0, 1).astype(np.float32)
    target3 = np.clip(1.0 - source3, 0, 1).astype(np.float32)

    paths = []
    for idx, (src, tgt) in enumerate(
        [(source1, target1), (source2, target2), (source3, target3)], start=1
    ):
        src_path = tmp_image_dir / f"o_src{idx}.png"
        tgt_path = tmp_image_dir / f"o_tgt{idx}.png"
        save_image(src, src_path, bit_depth=8)
        save_image(tgt, tgt_path, bit_depth=8)
        paths.append((src_path, tgt_path))

    out_path = tmp_path / "batch_outlier.cube"
    config = PipelineConfig(
        output_path=out_path,
        lut_size=5,
        bin_resolution=8,
        min_samples_per_bin=2,
        irls_iterations=1,
        format=ExportFormat.CUBE,
    )

    result = run_multi_pipeline(
        paths,
        config,
        outlier_sigma=1.0,
        min_pairs_after_outlier=2,
    )

    outlier_info = result.diagnostics["outlier_rejection"]
    assert outlier_info["enabled"] is True
    assert outlier_info["applied"] is True
    assert len(outlier_info["dropped_pair_indices"]) >= 1


def test_run_multi_pipeline_with_pair_overrides(tmp_image_dir, tmp_path):
    """Per-pair transfer/normalization overrides should flow into diagnostics."""
    from lutsmith.io.image import save_image
    from lutsmith.pipeline.runner import run_multi_pipeline

    rng = np.random.default_rng(404)
    src = rng.random((64, 64, 3), dtype=np.float32)
    tgt = np.clip(src * 1.06 + 0.015, 0, 1).astype(np.float32)

    src_path = tmp_image_dir / "ov_src.png"
    tgt_path = tmp_image_dir / "ov_tgt.png"
    save_image(src, src_path, bit_depth=8)
    save_image(tgt, tgt_path, bit_depth=8)

    out_path = tmp_path / "override_output.cube"
    config = PipelineConfig(
        output_path=out_path,
        lut_size=5,
        bin_resolution=8,
        min_samples_per_bin=2,
        irls_iterations=1,
        format=ExportFormat.CUBE,
    )

    result = run_multi_pipeline(
        [(src_path, tgt_path)],
        config,
        pair_transfer_fns=["auto"],
        pair_normalization_modes=["rgb_affine"],
    )

    assert result.output_path == out_path
    assert result.diagnostics["normalization_modes"] == ["rgb_affine"]
    assert result.diagnostics["pairs"][0]["normalization_mode"] == "rgb_affine"
