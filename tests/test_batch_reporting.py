"""Tests for batch metrics reporting helpers."""

from __future__ import annotations

import numpy as np

from lutsmith.core.types import (
    InterpolationKernel,
    LUTData,
    PipelineResult,
    QualityMetrics,
)
from lutsmith.pipeline.batch_manifest import ManifestEntry
from lutsmith.pipeline.reporting import build_batch_metrics_rows, write_batch_metrics_csv
from lutsmith.pipeline.reporting import (
    build_cluster_assignment_rows,
    build_cluster_centroid_rows,
    write_cluster_assignments_csv,
    write_cluster_centroids_csv,
    read_cluster_centroids_csv,
)


def _fake_result(tmp_path, name: str, mean_de: float, cluster_label: str | None = None) -> PipelineResult:
    lut = LUTData(array=np.zeros((2, 2, 2, 3), dtype=np.float32), size=2, kernel=InterpolationKernel.TETRAHEDRAL)
    metrics = QualityMetrics(
        mean_delta_e=mean_de,
        median_delta_e=mean_de,
        p95_delta_e=mean_de,
        max_delta_e=mean_de + 1.0,
        total_variation=0.123,
        coverage_percentage=12.5,
        oog_percentage=0.2,
    )
    diag = {
        "num_pairs": 10,
        "num_pairs_used": 9,
        "occupied_bins": 123,
        "total_time": 1.234,
    }
    if cluster_label:
        diag["cluster_label"] = cluster_label
        diag["cluster_indices"] = [1, 2, 3]
    return PipelineResult(
        lut=lut,
        metrics=metrics,
        diagnostics=diag,
        output_path=tmp_path / f"{name}.cube",
    )


def test_build_batch_metrics_rows_master_and_cluster(tmp_path):
    master = _fake_result(tmp_path, "master", 1.1)
    cluster = _fake_result(tmp_path, "cluster_a", 2.2, cluster_label="scene_a")

    rows = build_batch_metrics_rows(master, [cluster])
    assert len(rows) == 2
    assert rows[0]["label"] == "master"
    assert rows[1]["label"] == "scene_a"
    assert rows[1]["cluster_indices"] == "1,2,3"


def test_write_batch_metrics_csv(tmp_path):
    rows = [
        {
            "label": "master",
            "output_path": "out.cube",
            "mean_de2000": "1.0000",
            "median_de2000": "1.0000",
            "p95_de2000": "1.0000",
            "max_de2000": "2.0000",
            "total_variation": "0.100000",
            "coverage_pct": "10.000",
            "oog_pct": "0.100",
            "num_pairs": "10",
            "num_pairs_used": "9",
            "occupied_bins": "123",
            "outlier_dropped_count": "1",
            "total_time_s": "2.000",
            "cluster_indices": "",
        }
    ]
    out = tmp_path / "metrics.csv"
    path = write_batch_metrics_csv(rows, out)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "label,output_path,mean_de2000" in text
    assert "master,out.cube,1.0000" in text


def test_cluster_artifact_csv_roundtrip(tmp_path):
    entries = [
        ManifestEntry(source=tmp_path / "a.png", target=tmp_path / "b.png", weight=1.0),
        ManifestEntry(source=tmp_path / "c.png", target=tmp_path / "d.png", weight=1.0),
    ]
    assignments = np.array([0, 1], dtype=np.int64)
    id_to_label = {0: "scene_a", 1: "scene_b"}
    sig_meta = [
        {"transfer_function": "log_c4", "occupied_bins": 10, "appearance_signature": [0.1] * 8},
        {"transfer_function": "log_c4", "occupied_bins": 11, "appearance_signature": [0.2] * 8},
    ]
    t_sig = np.vstack([np.full(16, 0.1), np.full(16, 0.2)])
    a_sig = np.vstack([np.full(8, 0.1), np.full(8, 0.2)])
    lut_paths = {"scene_a": "a.cube", "scene_b": "b.cube"}

    assign_rows = build_cluster_assignment_rows(entries, assignments, id_to_label, signature_meta=sig_meta)
    centroid_rows = build_cluster_centroid_rows(assignments, t_sig, a_sig, id_to_label, cluster_lut_paths=lut_paths)

    assign_path = write_cluster_assignments_csv(assign_rows, tmp_path / "assign.csv")
    cent_path = write_cluster_centroids_csv(centroid_rows, tmp_path / "cent.csv")
    assert assign_path.exists()
    assert cent_path.exists()

    rows, app = read_cluster_centroids_csv(cent_path)
    assert len(rows) == 2
    assert app.shape == (2, 8)
