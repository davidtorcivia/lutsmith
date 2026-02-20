"""Tests for shot routing helpers."""

from __future__ import annotations

import numpy as np

from lutsmith.core.types import PipelineConfig
from lutsmith.pipeline.routing import (
    ShotFrameEntry,
    compute_shot_signatures,
    parse_shot_manifest,
    run_shot_routing,
    temporal_route_assignments,
)


def test_parse_shot_manifest_variants(tmp_path):
    f1 = tmp_path / "f1.png"
    f2 = tmp_path / "f2.png"
    f1.write_bytes(b"")
    f2.write_bytes(b"")

    manifest = tmp_path / "shots.csv"
    manifest.write_text(
        "shot_id,frame_path,transfer_fn\n"
        "A,f1.png,log_c4\n"
        "B,f2.png,\n",
        encoding="utf-8",
    )
    rows = parse_shot_manifest(manifest)
    assert len(rows) == 2
    assert rows[0].shot_id == "A"
    assert rows[0].transfer_fn == "log_c4"
    assert rows[1].transfer_fn is None


def test_temporal_route_assignments_switch_penalty():
    # Without switch penalty, frame-by-frame minima alternate.
    D = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.1],
            [0.1, 0.2],
            [0.2, 0.1],
        ],
        dtype=np.float64,
    )
    raw = temporal_route_assignments(D, temporal_window=1, switch_penalty=0.0)
    smooth = temporal_route_assignments(D, temporal_window=1, switch_penalty=0.25)

    assert raw.tolist() == [0, 1, 0, 1]
    # Penalty should reduce switching, preferring one cluster.
    assert smooth.tolist() in ([0, 0, 0, 0], [1, 1, 1, 1])


def test_compute_shot_signatures(tmp_image_dir, tmp_path):
    from lutsmith.io.image import save_image

    rng = np.random.default_rng(123)
    img1 = rng.random((32, 32, 3), dtype=np.float32)
    img2 = rng.random((32, 32, 3), dtype=np.float32)

    p1 = tmp_image_dir / "s1.png"
    p2 = tmp_image_dir / "s2.png"
    save_image(img1, p1, bit_depth=8)
    save_image(img2, p2, bit_depth=8)

    manifest = tmp_path / "shots2.csv"
    manifest.write_text(
        "shot_id,frame_path\n"
        f"001,{p1}\n"
        f"002,{p2}\n",
        encoding="utf-8",
    )
    entries = parse_shot_manifest(manifest)
    config = PipelineConfig()
    order, sig, diag = compute_shot_signatures(entries, config)
    assert order == ["001", "002"]
    assert sig.shape == (2, 8)
    assert len(diag) == 2


def test_run_shot_routing_generates_rows_and_csv(tmp_path):
    shot_entries = [
        ShotFrameEntry("s1", tmp_path / "a.png"),
        ShotFrameEntry("s2", tmp_path / "b.png"),
        ShotFrameEntry("s3", tmp_path / "c.png"),
    ]
    centroid_rows = [
        {"cluster_id": "0", "cluster_label": "scene_a", "lut_path": "a.cube"},
        {"cluster_id": "1", "cluster_label": "scene_b", "lut_path": "b.cube"},
    ]
    centroid_app = np.vstack([np.zeros(8, dtype=np.float64), np.ones(8, dtype=np.float64)])
    shot_sig = np.vstack(
        [
            np.full(8, 0.1, dtype=np.float64),
            np.full(8, 0.9, dtype=np.float64),
            np.full(8, 0.2, dtype=np.float64),
        ]
    )
    shot_diag = [
        {"shot_id": "s1", "frame_count": 1, "frames": ["a.png"]},
        {"shot_id": "s2", "frame_count": 1, "frames": ["b.png"]},
        {"shot_id": "s3", "frame_count": 1, "frames": ["c.png"]},
    ]

    import lutsmith.pipeline.routing as routing_mod

    original = routing_mod.compute_shot_signatures
    routing_mod.compute_shot_signatures = lambda *args, **kwargs: (
        ["s1", "s2", "s3"],
        shot_sig,
        shot_diag,
    )
    try:
        out = tmp_path / "routing.csv"
        result = run_shot_routing(
            shot_entries,
            centroid_rows,
            centroid_app,
            PipelineConfig(),
            output_path=out,
            temporal_window=1,
            switch_penalty=0.0,
        )
    finally:
        routing_mod.compute_shot_signatures = original

    assert result.output_path == out
    assert out.exists()
    assert len(result.rows) == 3
    assert [r["cluster_label"] for r in result.rows] == ["scene_a", "scene_b", "scene_a"]
    assert result.cluster_switches == 2
