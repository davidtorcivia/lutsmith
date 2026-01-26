"""Shared fixtures for ChromaForge tests."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from chromaforge.core.types import (
    InterpolationKernel,
    PipelineConfig,
    QualityMetrics,
    flat_index,
)


@pytest.fixture
def small_N():
    """Small LUT grid size for fast tests."""
    return 5


@pytest.fixture
def medium_N():
    """Medium LUT grid size for more thorough tests."""
    return 9


@pytest.fixture
def identity_lut_5():
    """5x5x5 identity LUT."""
    from chromaforge.core.lut import identity_lut
    return identity_lut(5)


@pytest.fixture
def identity_lut_9():
    """9x9x9 identity LUT."""
    from chromaforge.core.lut import identity_lut
    return identity_lut(9)


@pytest.fixture
def identity_lut_17():
    """17x17x17 identity LUT."""
    from chromaforge.core.lut import identity_lut
    return identity_lut(17)


@pytest.fixture
def random_colors():
    """Random (M, 3) float32 colors in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((500, 3), dtype=np.float32)


@pytest.fixture
def grid_corner_colors():
    """All 8 unit-cube corners as (8, 3) array."""
    corners = []
    for r in [0.0, 1.0]:
        for g in [0.0, 1.0]:
            for b in [0.0, 1.0]:
                corners.append([r, g, b])
    return np.array(corners, dtype=np.float32)


@pytest.fixture
def grid_node_colors(small_N):
    """All grid node colors for a small LUT as (N^3, 3) array."""
    N = small_N
    coords = np.linspace(0, 1, N, dtype=np.float32)
    rr, gg, bb = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=-1)


@pytest.fixture
def synthetic_pair_small():
    """Synthetic source/target pair: target = gain * source + offset.

    Returns (source, target, gain, offset).
    """
    rng = np.random.default_rng(123)
    M = 2000
    source = rng.random((M, 3), dtype=np.float32)

    gain = np.array([1.1, 0.9, 1.05], dtype=np.float32)
    offset = np.array([0.02, -0.01, 0.03], dtype=np.float32)
    target = np.clip(source * gain + offset, 0, 1).astype(np.float32)

    return source, target, gain, offset


@pytest.fixture
def tmp_cube_path(tmp_path):
    """Temporary .cube output path."""
    return tmp_path / "test_output.cube"


@pytest.fixture
def tmp_image_dir(tmp_path):
    """Temporary directory for test images."""
    d = tmp_path / "images"
    d.mkdir()
    return d


@pytest.fixture
def sample_image_pair(tmp_image_dir):
    """Create a pair of small PNG images on disk.

    Returns (source_path, target_path).
    """
    from chromaforge.io.image import save_image

    rng = np.random.default_rng(99)
    source = rng.random((64, 64, 3), dtype=np.float32)
    target = np.clip(source * 1.1 + 0.02, 0, 1).astype(np.float32)

    src_path = tmp_image_dir / "source.png"
    tgt_path = tmp_image_dir / "target.png"
    save_image(source, src_path, bit_depth=8)
    save_image(target, tgt_path, bit_depth=8)

    return src_path, tgt_path
