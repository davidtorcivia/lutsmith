"""Manifest parsing for batch source/target pair workflows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    """Single batch manifest row."""

    source: Path
    target: Path
    weight: float = 1.0
    cluster: str | None = None
    transfer_fn: str | None = None
    normalization: str | None = None


def parse_pair_manifest(manifest: Path) -> list[ManifestEntry]:
    """Parse CSV manifest.

    Supports optional header row:
        source,target
        source,target,weight
        source,target,weight,cluster
        source,target,weight,cluster,transfer_fn,normalization

    Relative paths are resolved relative to the manifest directory.
    """
    if not manifest.exists():
        raise ValueError(f"Manifest not found: {manifest}")

    entries: list[ManifestEntry] = []
    base_dir = manifest.parent

    with manifest.open("r", encoding="utf-8", newline="") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            row = next(csv.reader([raw_line], skipinitialspace=True))
            if len(row) not in {2, 3, 4, 5, 6}:
                raise ValueError(
                    f"Invalid manifest line {line_no}: expected 2-6 CSV fields "
                    f"(source,target[,weight][,cluster][,transfer_fn][,normalization]), got {len(row)}"
                )

            c0 = row[0].strip().lower()
            c1 = row[1].strip().lower()
            c2 = row[2].strip().lower() if len(row) >= 3 else ""
            c3 = row[3].strip().lower() if len(row) >= 4 else ""
            c4 = row[4].strip().lower() if len(row) >= 5 else ""
            c5 = row[5].strip().lower() if len(row) >= 6 else ""
            is_header = (
                c0 in {"source", "src"}
                and c1 in {"target", "tgt"}
                and (len(row) < 3 or c2 in {"weight", "w"})
                and (len(row) < 4 or c3 in {"cluster", "scene", "group"})
                and (len(row) < 5 or c4 in {"transfer_fn", "transfer", "tf"})
                and (len(row) < 6 or c5 in {"normalization", "normalize", "norm"})
            )
            if is_header:
                continue

            source = Path(row[0].strip())
            target = Path(row[1].strip())
            weight = 1.0
            cluster = None
            transfer_fn = None
            normalization = None

            if len(row) >= 3 and row[2].strip():
                try:
                    weight = float(row[2].strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid weight on line {line_no}: '{row[2]}'"
                    ) from exc
                if weight < 0:
                    raise ValueError(f"Invalid weight on line {line_no}: must be >= 0")

            if len(row) >= 4:
                cluster_raw = row[3].strip()
                cluster = cluster_raw if cluster_raw else None

            if len(row) >= 5:
                tf_raw = row[4].strip()
                transfer_fn = tf_raw.lower() if tf_raw else None

            if len(row) >= 6:
                norm_raw = row[5].strip()
                normalization = norm_raw.lower() if norm_raw else None

            if not source.is_absolute():
                source = (base_dir / source).resolve()
            if not target.is_absolute():
                target = (base_dir / target).resolve()

            entries.append(
                ManifestEntry(
                    source=source,
                    target=target,
                    weight=weight,
                    cluster=cluster,
                    transfer_fn=transfer_fn,
                    normalization=normalization,
                )
            )

    if not entries:
        raise ValueError(
            "Manifest contains no valid pairs. Add lines like: "
            "source.png,target.png[,1.0][,scene_a][,log_c4][,rgb_affine]"
        )

    return entries
