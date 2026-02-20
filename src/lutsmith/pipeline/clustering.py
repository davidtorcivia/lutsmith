"""Scene clustering utilities for batch LUT extraction."""

from __future__ import annotations

from typing import Callable, Optional, Sequence
from pathlib import Path

import numpy as np

from lutsmith.core.types import BinStatistics, PipelineConfig
from lutsmith.pipeline.preprocess import preprocess_pair
from lutsmith.pipeline.sampling import bin_and_aggregate


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return np.mean(x, axis=0)
    return np.sum(x * w[:, np.newaxis], axis=0) / w_sum


def pair_signature_from_bins(bins: Sequence[BinStatistics]) -> np.ndarray:
    """Build a compact transform signature from aggregated bin statistics."""
    if len(bins) == 0:
        return np.zeros(16, dtype=np.float64)

    counts = np.array([b.count for b in bins], dtype=np.float64)
    mean_in = np.stack([b.mean_input for b in bins], axis=0).astype(np.float64)
    mean_out = np.stack([b.mean_output for b in bins], axis=0).astype(np.float64)
    delta = mean_out - mean_in

    in_avg = _weighted_mean(mean_in, counts)
    out_avg = _weighted_mean(mean_out, counts)
    delta_avg = out_avg - in_avg

    abs_delta = _weighted_mean(np.abs(delta), counts)
    delta_std = np.sqrt(_weighted_mean((delta - delta_avg) ** 2, counts))

    # Channel-wise linear slope proxy (cov / var) on bin means.
    centered_in = mean_in - _weighted_mean(mean_in, counts)
    centered_out = mean_out - _weighted_mean(mean_out, counts)
    var_in = _weighted_mean(centered_in ** 2, counts)
    cov_io = _weighted_mean(centered_in * centered_out, counts)
    slope = cov_io / np.maximum(var_in, 1e-8)

    # Luma + saturation behavior helps separate scene/grade families.
    lum_in = (
        0.2126 * mean_in[:, 0]
        + 0.7152 * mean_in[:, 1]
        + 0.0722 * mean_in[:, 2]
    )
    lum_out = (
        0.2126 * mean_out[:, 0]
        + 0.7152 * mean_out[:, 1]
        + 0.0722 * mean_out[:, 2]
    )
    sat_in = np.max(mean_in, axis=1) - np.min(mean_in, axis=1)
    sat_out = np.max(mean_out, axis=1) - np.min(mean_out, axis=1)
    lum_delta = lum_out - lum_in
    sat_delta = sat_out - sat_in

    lum_stats = np.array(
        [
            float(np.average(lum_in, weights=counts)),
            float(np.sqrt(np.average((lum_in - np.average(lum_in, weights=counts)) ** 2, weights=counts))),
            float(np.average(lum_delta, weights=counts)),
            float(np.sqrt(np.average((lum_delta - np.average(lum_delta, weights=counts)) ** 2, weights=counts))),
        ],
        dtype=np.float64,
    )
    sat_stats = np.array(
        [
            float(np.average(sat_in, weights=counts)),
            float(np.average(sat_delta, weights=counts)),
        ],
        dtype=np.float64,
    )

    # 16D signature.
    signature = np.concatenate([delta_avg, abs_delta, delta_std, slope, lum_stats, sat_stats])
    return signature


def compute_pair_signatures(
    pair_paths: Sequence[tuple[Path | str, Path | str]],
    config: PipelineConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[np.ndarray, list[dict]]:
    """Compute signatures for each pair using preprocessing + bin aggregation."""
    signatures = []
    metadata = []
    for i, (source_path, target_path) in enumerate(pair_paths):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")

        source, target, meta, _ = preprocess_pair(source_path, target_path, config)
        bins, _ = bin_and_aggregate(source, target, config)
        if len(bins) < 10:
            raise ValueError(
                f"Pair {i + 1} has only {len(bins)} occupied bins; cannot cluster reliably."
            )
        sig = pair_signature_from_bins(bins)
        signatures.append(sig)
        metadata.append(
            {
                "index": i + 1,
                "source": str(source_path),
                "target": str(target_path),
                "occupied_bins": len(bins),
                "transfer_function": meta["transfer_function"].value,
            }
        )
        if progress_callback is not None:
            progress_callback((i + 1) / len(pair_paths), f"signatures {i + 1}/{len(pair_paths)}")

    return np.vstack(signatures), metadata


def _robust_standardize(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (X - med) / scale


def _init_kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(X)
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        probs = d2 / np.maximum(np.sum(d2), 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers[c] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centers[c]) ** 2, axis=1))
    return centers


def _kmeans_once(X: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 64) -> tuple[np.ndarray, np.ndarray, float]:
    centers = _init_kmeans_pp(X, k, rng)
    labels = np.zeros(len(X), dtype=np.int64)
    for _ in range(max_iter):
        dist = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centers[c] = np.mean(X[mask], axis=0)
            else:
                centers[c] = X[int(rng.integers(0, len(X)))]

    final_dist = np.sum((X - centers[labels]) ** 2, axis=1)
    inertia = float(np.sum(final_dist))
    return labels, centers, inertia


def kmeans_cluster_features(
    features: np.ndarray,
    k: int,
    n_init: int = 8,
    random_seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Cluster feature vectors with deterministic multi-start k-means."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(features):
        raise ValueError("k cannot exceed number of samples")

    X = _robust_standardize(np.asarray(features, dtype=np.float64))
    if k == 1:
        labels = np.zeros(len(X), dtype=np.int64)
        return labels, {"k": 1, "inertia": 0.0}

    rng = np.random.default_rng(random_seed)
    best = None
    for _ in range(max(n_init, 1)):
        run_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        labels, centers, inertia = _kmeans_once(X, k, run_rng)
        if best is None or inertia < best["inertia"]:
            best = {"labels": labels, "centers": centers, "inertia": inertia}

    cluster_sizes = [int(np.sum(best["labels"] == i)) for i in range(k)]
    return best["labels"], {
        "k": k,
        "inertia": float(best["inertia"]),
        "cluster_sizes": cluster_sizes,
    }


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    n = len(X)
    k = len(np.unique(labels))
    if k <= 1 or n <= 2:
        return -1.0

    # O(n^2), fine for typical batch counts.
    D = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2))
    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = float(np.mean(D[i, same])) if np.any(same) else 0.0
        b = np.inf
        for c in np.unique(labels):
            if c == labels[i]:
                continue
            mask = labels == c
            if np.any(mask):
                b = min(b, float(np.mean(D[i, mask])))
        if not np.isfinite(b):
            s = 0.0
        else:
            s = (b - a) / max(a, b, 1e-12)
        scores.append(s)
    return float(np.mean(scores))


def auto_cluster_features(
    features: np.ndarray,
    max_clusters: int = 6,
    random_seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Choose k automatically using silhouette score."""
    X = np.asarray(features, dtype=np.float64)
    n = len(X)
    if n <= 2:
        labels = np.zeros(n, dtype=np.int64)
        return labels, {"k": 1, "method": "auto", "reason": "too_few_pairs"}

    Xn = _robust_standardize(X)
    k_max = min(max_clusters, n)

    best = {"k": 1, "score": -1.0, "labels": np.zeros(n, dtype=np.int64), "diag": {"k": 1, "inertia": 0.0}}
    scores = []
    for k in range(2, k_max + 1):
        labels, diag = kmeans_cluster_features(X, k=k, random_seed=random_seed)
        score = _silhouette_score(Xn, labels)
        scores.append({"k": k, "score": score, "inertia": diag["inertia"]})
        if score > best["score"]:
            best = {"k": k, "score": score, "labels": labels, "diag": diag}

    # Conservative fallback: if separation is weak, prefer a single LUT.
    if best["score"] < 0.10:
        labels = np.zeros(n, dtype=np.int64)
        return labels, {
            "k": 1,
            "method": "auto",
            "reason": "low_silhouette",
            "best_silhouette": float(best["score"]),
            "scores": scores,
        }

    return best["labels"], {
        "k": int(best["k"]),
        "method": "auto",
        "silhouette": float(best["score"]),
        "scores": scores,
        "inertia": float(best["diag"]["inertia"]),
        "cluster_sizes": best["diag"].get("cluster_sizes", []),
    }
