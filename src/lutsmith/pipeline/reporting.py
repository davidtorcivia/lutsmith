"""Reporting helpers for batch extraction outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from lutsmith.core.types import PipelineResult


def _fmt_float(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return ""


def build_batch_metrics_rows(
    master: Optional[PipelineResult],
    clusters: list[PipelineResult],
) -> list[dict[str, str]]:
    """Build tabular rows for batch master/cluster metrics export."""
    rows: list[dict[str, str]] = []
    results: list[tuple[str, Optional[PipelineResult]]] = [("master", master)]
    results.extend(
        (
            str(result.diagnostics.get("cluster_label", f"cluster_{i + 1:02d}")),
            result,
        )
        for i, result in enumerate(clusters)
    )

    for label, result in results:
        if result is None:
            continue
        m = result.metrics
        d = result.diagnostics
        outlier = d.get("outlier_rejection", {}) if isinstance(d.get("outlier_rejection"), dict) else {}
        dropped = outlier.get("dropped_pair_indices", [])
        row = {
            "label": label,
            "output_path": str(result.output_path) if result.output_path else "",
            "mean_de2000": _fmt_float(m.mean_delta_e, 4),
            "median_de2000": _fmt_float(m.median_delta_e, 4),
            "p95_de2000": _fmt_float(m.p95_delta_e, 4),
            "max_de2000": _fmt_float(m.max_delta_e, 4),
            "total_variation": _fmt_float(m.total_variation, 6),
            "coverage_pct": _fmt_float(m.coverage_percentage, 3),
            "oog_pct": _fmt_float(m.oog_percentage, 3),
            "num_pairs": str(d.get("num_pairs", "")),
            "num_pairs_used": str(d.get("num_pairs_used", d.get("num_pairs", ""))),
            "occupied_bins": str(d.get("occupied_bins", "")),
            "outlier_dropped_count": str(len(dropped) if isinstance(dropped, list) else 0),
            "total_time_s": _fmt_float(d.get("total_time"), 3),
            "cluster_indices": ",".join(str(x) for x in d.get("cluster_indices", []))
            if isinstance(d.get("cluster_indices"), list)
            else "",
        }
        rows.append(row)
    return rows


def write_batch_metrics_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    """Write batch metrics rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "label",
        "output_path",
        "mean_de2000",
        "median_de2000",
        "p95_de2000",
        "max_de2000",
        "total_variation",
        "coverage_pct",
        "oog_pct",
        "num_pairs",
        "num_pairs_used",
        "occupied_bins",
        "outlier_dropped_count",
        "total_time_s",
        "cluster_indices",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return output_path

