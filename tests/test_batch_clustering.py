"""Tests for batch scene-clustering utilities."""

from __future__ import annotations

import numpy as np

from lutsmith.pipeline.batch_manifest import parse_pair_manifest
from lutsmith.pipeline.clustering import auto_cluster_features, kmeans_cluster_features


def test_manifest_parser_supports_cluster_column(tmp_path):
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    src.write_bytes(b"")
    tgt.write_bytes(b"")

    manifest = tmp_path / "pairs.csv"
    manifest.write_text(
        "source,target,weight,cluster\n"
        "src.png,tgt.png,1.5,scene_a\n",
        encoding="utf-8",
    )

    entries = parse_pair_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].weight == 1.5
    assert entries[0].cluster == "scene_a"


def test_kmeans_cluster_features_two_groups():
    rng = np.random.default_rng(123)
    g1 = rng.normal(loc=-2.0, scale=0.1, size=(8, 6))
    g2 = rng.normal(loc=2.0, scale=0.1, size=(8, 6))
    X = np.vstack([g1, g2])

    labels, diag = kmeans_cluster_features(X, k=2, random_seed=42)
    assert diag["k"] == 2
    assert len(set(labels.tolist())) == 2

    # First half and second half should mostly separate.
    first_majority = int(np.sum(labels[:8] == labels[0]))
    second_majority = int(np.sum(labels[8:] == labels[8]))
    assert first_majority >= 7
    assert second_majority >= 7


def test_auto_cluster_features_finds_two_clusters():
    rng = np.random.default_rng(321)
    g1 = rng.normal(loc=-1.5, scale=0.12, size=(10, 5))
    g2 = rng.normal(loc=1.5, scale=0.12, size=(10, 5))
    X = np.vstack([g1, g2])

    labels, diag = auto_cluster_features(X, max_clusters=4, random_seed=7)
    assert diag["k"] >= 2
    assert len(set(labels.tolist())) >= 2
