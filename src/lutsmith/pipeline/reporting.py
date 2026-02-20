"""Reporting helpers for batch extraction outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from lutsmith.core.types import PipelineResult
from lutsmith.pipeline.batch_manifest import ManifestEntry

TRANSFORM_SIGNATURE_DIM = 16
APPEARANCE_SIGNATURE_DIM = 8


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


def build_cluster_assignment_rows(
    entries: list[ManifestEntry],
    assignments: np.ndarray,
    cluster_id_to_label: dict[int, str],
    signature_meta: Optional[list[dict]] = None,
) -> list[dict[str, str]]:
    """Build per-pair assignment rows for cluster export."""
    rows: list[dict[str, str]] = []
    for i, entry in enumerate(entries):
        cid = int(assignments[i])
        label = cluster_id_to_label.get(cid, f"cluster_{cid + 1:02d}")
        meta = signature_meta[i] if signature_meta is not None and i < len(signature_meta) else {}
        rows.append(
            {
                "pair_index": str(i + 1),
                "source": str(entry.source),
                "target": str(entry.target),
                "weight": _fmt_float(entry.weight, 6),
                "cluster_id": str(cid),
                "cluster_label": label,
                "transfer_fn_override": str(entry.transfer_fn or ""),
                "normalization": str(entry.normalization or ""),
                "detected_transfer_function": str(meta.get("transfer_function", "")),
                "occupied_bins": str(meta.get("occupied_bins", "")),
            }
        )
    return rows


def write_cluster_assignments_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    """Write pair->cluster assignment table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "pair_index",
        "source",
        "target",
        "weight",
        "cluster_id",
        "cluster_label",
        "transfer_fn_override",
        "normalization",
        "detected_transfer_function",
        "occupied_bins",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def build_cluster_centroid_rows(
    assignments: np.ndarray,
    transform_signatures: np.ndarray,
    appearance_signatures: np.ndarray,
    cluster_id_to_label: dict[int, str],
    cluster_lut_paths: Optional[dict[str, str]] = None,
) -> list[dict[str, str]]:
    """Build centroid/signature rows for each cluster."""
    rows: list[dict[str, str]] = []
    cluster_ids = sorted(set(int(x) for x in assignments.tolist()))
    for cid in cluster_ids:
        mask = assignments == cid
        if not np.any(mask):
            continue
        label = cluster_id_to_label.get(cid, f"cluster_{cid + 1:02d}")
        t_cent = np.mean(transform_signatures[mask], axis=0)
        a_cent = np.mean(appearance_signatures[mask], axis=0)

        row: dict[str, str] = {
            "cluster_id": str(cid),
            "cluster_label": label,
            "count": str(int(np.sum(mask))),
            "lut_path": str(cluster_lut_paths.get(label, "")) if cluster_lut_paths else "",
        }
        for i in range(TRANSFORM_SIGNATURE_DIM):
            row[f"t_sig_{i:02d}"] = _fmt_float(float(t_cent[i]), 8)
        for i in range(APPEARANCE_SIGNATURE_DIM):
            row[f"a_sig_{i:02d}"] = _fmt_float(float(a_cent[i]), 8)
        rows.append(row)
    return rows


def write_cluster_centroids_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    """Write cluster centroid/signature table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["cluster_id", "cluster_label", "count", "lut_path"]
    headers.extend([f"t_sig_{i:02d}" for i in range(TRANSFORM_SIGNATURE_DIM)])
    headers.extend([f"a_sig_{i:02d}" for i in range(APPEARANCE_SIGNATURE_DIM)])
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def read_cluster_centroids_csv(path: Path) -> tuple[list[dict], np.ndarray]:
    """Read centroid CSV and return rows + appearance centroid matrix."""
    rows = []
    app = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            app_vec = []
            for i in range(APPEARANCE_SIGNATURE_DIM):
                key = f"a_sig_{i:02d}"
                app_vec.append(float(r.get(key, "0") or "0"))
            app.append(app_vec)
    if not rows:
        raise ValueError(f"No centroid rows found in {path}")
    return rows, np.asarray(app, dtype=np.float64)
