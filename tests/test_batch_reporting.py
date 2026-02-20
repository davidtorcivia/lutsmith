"""Tests for batch metrics reporting helpers."""

from __future__ import annotations

import numpy as np

from lutsmith.core.types import (
    InterpolationKernel,
    LUTData,
    PipelineResult,
    QualityMetrics,
)
from lutsmith.pipeline.reporting import build_batch_metrics_rows, write_batch_metrics_csv


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
