"""Shot routing helpers for cluster-LUT runtime assignment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from lutsmith.core.types import PipelineConfig, TransferFunction
from lutsmith.pipeline.clustering import source_appearance_signature_from_image
from lutsmith.pipeline.preprocess import _resolve_shaper, detect_transfer_function, sanitize_image


@dataclass(frozen=True)
class ShotFrameEntry:
    shot_id: str
    frame_path: Path
    transfer_fn: Optional[str] = None


@dataclass(frozen=True)
class ShotRoutingResult:
    shot_order: list[str]
    assignments: np.ndarray
    distances: np.ndarray
    rows: list[dict[str, str]]
    output_path: Optional[Path]
    cluster_switches: int
    diagnostics: list[dict]


def parse_shot_manifest(path: Path) -> list[ShotFrameEntry]:
    """Parse shot manifest CSV.

    Accepted rows:
        frame_path
        shot_id,frame_path
        shot_id,frame_path,transfer_fn
    """
    if not path.exists():
        raise ValueError(f"Shot manifest not found: {path}")

    entries: list[ShotFrameEntry] = []
    base = path.parent
    auto_index = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        for line_no, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            row = next(csv.reader([raw], skipinitialspace=True))
            if len(row) not in {1, 2, 3}:
                raise ValueError(
                    f"Invalid shot manifest line {line_no}: expected 1-3 fields, got {len(row)}"
                )

            c0 = row[0].strip().lower()
            c1 = row[1].strip().lower() if len(row) >= 2 else ""
            c2 = row[2].strip().lower() if len(row) >= 3 else ""
            if c0 in {"shot", "shot_id"} and c1 in {"frame", "frame_path", "path"}:
                continue

            if len(row) == 1:
                auto_index += 1
                shot_id = f"shot_{auto_index:04d}"
                frame_path = Path(row[0].strip())
                tf = None
            else:
                shot_id = row[0].strip() or f"shot_{line_no:04d}"
                frame_path = Path(row[1].strip())
                tf = row[2].strip().lower() if len(row) == 3 and row[2].strip() else None

            if not frame_path.is_absolute():
                frame_path = (base / frame_path).resolve()
            entries.append(ShotFrameEntry(shot_id=shot_id, frame_path=frame_path, transfer_fn=tf))

    if not entries:
        raise ValueError("Shot manifest contains no valid rows")
    return entries


def _load_source_for_routing(
    path: Path,
    config: PipelineConfig,
    transfer_override: Optional[str],
) -> np.ndarray:
    from lutsmith.io.image import load_image

    img, meta = load_image(path)
    img = sanitize_image(img)
    tf = config.transfer_function
    if transfer_override:
        try:
            tf = TransferFunction(transfer_override.strip().lower())
        except ValueError as exc:
            raise ValueError(f"Invalid transfer_fn override '{transfer_override}' for {path}") from exc
    elif tf == TransferFunction.AUTO:
        tf = detect_transfer_function(img, meta)

    shaper_forward, _, _, _ = _resolve_shaper(tf, config.include_shaper)
    return shaper_forward(img)


def compute_shot_signatures(
    entries: list[ShotFrameEntry],
    config: PipelineConfig,
    progress_callback=None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[list[str], np.ndarray, list[dict]]:
    """Compute per-shot source appearance signatures (average across frames)."""
    grouped: dict[str, list[ShotFrameEntry]] = {}
    shot_order: list[str] = []
    for e in entries:
        if e.shot_id not in grouped:
            grouped[e.shot_id] = []
            shot_order.append(e.shot_id)
        grouped[e.shot_id].append(e)

    signatures = []
    diagnostics = []
    for i, shot_id in enumerate(shot_order):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")
        rows = grouped[shot_id]
        sigs = []
        for row in rows:
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Cancelled")
            src = _load_source_for_routing(row.frame_path, config, row.transfer_fn)
            sig = source_appearance_signature_from_image(src)
            sigs.append(sig)
        sig_avg = np.mean(np.asarray(sigs, dtype=np.float64), axis=0)
        signatures.append(sig_avg)
        diagnostics.append(
            {
                "shot_id": shot_id,
                "frame_count": len(rows),
                "frames": [str(r.frame_path) for r in rows],
            }
        )
        if progress_callback is not None:
            progress_callback((i + 1) / len(shot_order), f"shot signatures {i + 1}/{len(shot_order)}")

    return shot_order, np.vstack(signatures), diagnostics


def run_shot_routing(
    shot_entries: list[ShotFrameEntry],
    centroid_rows: list[dict],
    centroid_appearance: np.ndarray,
    config: PipelineConfig,
    output_path: Optional[Path] = None,
    temporal_window: int = 1,
    switch_penalty: float = 0.0,
    progress_callback=None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ShotRoutingResult:
    """Assign each shot to the nearest cluster centroid with optional temporal smoothing."""
    if len(shot_entries) == 0:
        raise ValueError("No shot entries provided")
    if len(centroid_rows) == 0:
        raise ValueError("No centroid rows provided")

    cent = np.asarray(centroid_appearance, dtype=np.float64)
    if cent.ndim != 2 or cent.shape[0] != len(centroid_rows):
        raise ValueError(
            "Centroid appearance matrix must have shape (num_centroids, signature_dim)"
        )

    if progress_callback is not None:
        progress_callback("preprocess", 0.0, "Computing shot signatures...")

    def on_sig_progress(fraction: float, message: str):
        if progress_callback is not None:
            progress_callback("preprocess", max(0.0, min(0.80 * fraction, 0.80)), message)

    shot_order, shot_signatures, shot_diag = compute_shot_signatures(
        shot_entries,
        config,
        progress_callback=on_sig_progress,
        cancel_check=cancel_check,
    )
    if cancel_check is not None and cancel_check():
        raise RuntimeError("Cancelled")

    med = np.median(cent, axis=0)
    mad = np.median(np.abs(cent - med), axis=0)
    scale = np.where(1.4826 * mad < 1e-8, 1.0, 1.4826 * mad)
    shot_n = (shot_signatures - med) / scale
    cent_n = (cent - med) / scale

    if progress_callback is not None:
        progress_callback("solving", 0.0, "Computing shot-to-cluster distances...")

    dists = np.sqrt(np.sum((shot_n[:, np.newaxis, :] - cent_n[np.newaxis, :, :]) ** 2, axis=2))
    assigned = temporal_route_assignments(
        dists,
        temporal_window=temporal_window,
        switch_penalty=switch_penalty,
    )
    if cancel_check is not None and cancel_check():
        raise RuntimeError("Cancelled")

    if progress_callback is not None:
        progress_callback("solving", 1.0, "Assignments complete")

    rows = []
    for i, shot_id in enumerate(shot_order):
        cid = int(assigned[i])
        centroid_row = centroid_rows[cid]
        sorted_d = np.sort(dists[i])
        margin = float(sorted_d[1] - sorted_d[0]) if len(sorted_d) > 1 else 0.0
        rows.append(
            {
                "shot_index": str(i + 1),
                "shot_id": shot_id,
                "cluster_id": centroid_row.get("cluster_id", str(cid)),
                "cluster_label": centroid_row.get("cluster_label", f"cluster_{cid + 1:02d}"),
                "lut_path": centroid_row.get("lut_path", ""),
                "distance": f"{float(dists[i, cid]):.8f}",
                "confidence_margin": f"{margin:.8f}",
                "frame_count": str(shot_diag[i].get("frame_count", 1)),
                "frames": ";".join(shot_diag[i].get("frames", [])),
            }
        )

    csv_output = None
    if output_path is not None:
        if progress_callback is not None:
            progress_callback("export", 0.0, "Writing routing CSV...")
        csv_output = write_shot_routing_csv(rows, output_path)
        if progress_callback is not None:
            progress_callback("export", 1.0, "Routing CSV written")

    switches = 0
    for i in range(1, len(rows)):
        if rows[i]["cluster_label"] != rows[i - 1]["cluster_label"]:
            switches += 1

    return ShotRoutingResult(
        shot_order=shot_order,
        assignments=assigned,
        distances=dists,
        rows=rows,
        output_path=csv_output,
        cluster_switches=switches,
        diagnostics=shot_diag,
    )


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    pad = window // 2
    out = np.empty_like(arr)
    for c in range(arr.shape[1]):
        x = arr[:, c]
        xp = np.pad(x, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / float(window)
        out[:, c] = np.convolve(xp, kernel, mode="valid")
    return out


def temporal_route_assignments(
    distances: np.ndarray,
    temporal_window: int = 1,
    switch_penalty: float = 0.0,
) -> np.ndarray:
    """Compute temporally robust cluster assignments.

    Args:
        distances: (num_shots, num_clusters) lower is better.
        temporal_window: Moving-average window on distance tracks.
        switch_penalty: Penalty added when assignment changes cluster between shots.
    """
    D = np.asarray(distances, dtype=np.float64)
    if temporal_window > 1:
        D = _moving_average(D, temporal_window)

    n, k = D.shape
    if n == 0:
        return np.array([], dtype=np.int64)
    if switch_penalty <= 0:
        return np.argmin(D, axis=1).astype(np.int64)

    # Viterbi on assignment graph with constant switch cost.
    cost = np.empty((n, k), dtype=np.float64)
    back = np.zeros((n, k), dtype=np.int64)
    cost[0] = D[0]
    for t in range(1, n):
        prev = cost[t - 1]
        for c in range(k):
            trans = prev + switch_penalty
            trans[c] = prev[c]
            j = int(np.argmin(trans))
            cost[t, c] = D[t, c] + trans[j]
            back[t, c] = j

    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmin(cost[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def write_shot_routing_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "shot_index",
        "shot_id",
        "cluster_id",
        "cluster_label",
        "lut_path",
        "distance",
        "confidence_margin",
        "frame_count",
        "frames",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
