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


def parse_pair_manifest(manifest: Path) -> list[ManifestEntry]:
    """Parse CSV manifest of source,target[,weight][,cluster] entries.

    Supports optional header row:
        source,target
        source,target,weight
        source,target,weight,cluster

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
            if len(row) not in {2, 3, 4}:
                raise ValueError(
                    f"Invalid manifest line {line_no}: expected 2-4 CSV fields "
                    f"(source,target[,weight][,cluster]), got {len(row)}"
                )

            c0 = row[0].strip().lower()
            c1 = row[1].strip().lower()
            c2 = row[2].strip().lower() if len(row) >= 3 else ""
            c3 = row[3].strip().lower() if len(row) >= 4 else ""
            is_header = (
                c0 in {"source", "src"}
                and c1 in {"target", "tgt"}
                and (len(row) < 3 or c2 in {"weight", "w"})
                and (len(row) < 4 or c3 in {"cluster", "scene", "group"})
            )
            if is_header:
                continue

            source = Path(row[0].strip())
            target = Path(row[1].strip())
            weight = 1.0
            cluster = None

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

            if not source.is_absolute():
                source = (base_dir / source).resolve()
            if not target.is_absolute():
                target = (base_dir / target).resolve()

            entries.append(ManifestEntry(source=source, target=target, weight=weight, cluster=cluster))

    if not entries:
        raise ValueError(
            "Manifest contains no valid pairs. Add lines like: source.png,target.png[,1.0][,scene_a]"
        )

    return entries

