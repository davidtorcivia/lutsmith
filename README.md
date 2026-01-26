# ChromaForge

Image-derived 3D LUT generation tool. ChromaForge extracts color transformations from matched image pairs and generates industry-standard 3D LUTs compatible with DaVinci Resolve, ARRI cameras, and other post-production tools.

---

## Overview

ChromaForge solves the inverse color-grading problem: given a source image and its graded variant, it reverse-engineers the color grade as a transferable 3D lookup table. The core algorithm uses **regularized lattice regression** with Laplacian smoothing, producing high-quality LUTs even from noisy or sparsely-sampled image data.

### Two Workflows

**Image-Pair Extraction** -- Provide a source image and its graded version. ChromaForge extracts the color mapping via sparse regression and outputs a .cube LUT.

**Hald CLUT Identity Plate** -- Process a generated Hald identity image through your grading pipeline, then reconstruct the LUT directly. Provides guaranteed full-gamut coverage with no interpolation artifacts.

---

## Installation

Requires Python 3.10+.

```
pip install .
```

With GUI support:

```
pip install ".[gui]"
```

With all optional dependencies:

```
pip install ".[gui,oiio,ocio,dev]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy >= 1.24 | Array operations |
| scipy >= 1.10 | Sparse linear algebra, EDT, interpolation |
| numba >= 0.57 | JIT-compiled binning kernels |
| colour-science >= 0.4.2 | DeltaE 2000 metrics |
| typer >= 0.9 | CLI framework |
| rich >= 13.0 | Terminal formatting |
| imageio >= 2.31 | Image I/O fallback |

Optional:

| Package | Purpose |
|---------|---------|
| PySide6 >= 6.5 | GUI (install with `[gui]`) |
| OpenImageIO >= 2.4 | Professional image I/O (install with `[oiio]`) |
| PyOpenColorIO >= 2.2 | Color management (install with `[ocio]`) |

---

## CLI Usage

### Extract a LUT from an image pair

```
chromaforge extract source.png graded.png -o output.cube
```

Key options:

```
  -o, --output PATH       Output file path [default: output.cube]
  -s, --size INT           LUT grid size: 17, 33, or 65 [default: 33]
  -k, --kernel TEXT        Interpolation kernel: tetrahedral or trilinear [default: tetrahedral]
  --smoothness FLOAT       Smoothness regularization lambda_s [default: 0.1]
  --prior FLOAT            Identity prior strength lambda_r [default: 0.01]
  --loss TEXT              Loss function: l2 or huber [default: huber]
  --irls-iter INT          IRLS iterations for robust loss [default: 3]
  --bin-res INT            Bin resolution per axis [default: 64]
  --min-samples INT        Minimum samples per bin [default: 3]
  --refine                 Enable iterative refinement
  -f, --format TEXT        Output format: cube, aml, alf4 [default: cube]
  -v, --verbose            Enable debug logging
```

### Generate a Hald identity image

```
chromaforge hald-gen -o identity.tiff -l 8
```

This produces a 512x512 identity image encoding a 64^3 LUT. Process it through your color grading pipeline, then reconstruct:

### Reconstruct LUT from processed Hald image

```
chromaforge hald-recon processed.tiff -o output.cube -l 8
```

Options:

```
  -l, --level INT          Hald level (8 = 512x512, 64^3 LUT) [default: 8]
  -s, --target-size INT    Resample to target LUT size (0 = native) [default: 0]
```

### Validate an existing LUT

```
chromaforge validate source.png graded.png existing.cube
```

Applies the LUT to the source image and reports DeltaE 2000 metrics against the target.

---

## GUI Usage

Launch the graphical interface:

```
chromaforge-gui
```

The GUI provides three tabs:

- **Image Pair** -- Load source/target images, adjust parameters, run extraction, and inspect quality metrics and coverage maps
- **Hald CLUT** -- Generate identity images, load processed results, and reconstruct LUTs
- **Settings** -- Configure output directory, LUT title, and view I/O backend status

The interface uses a dark theme with amber/gold accents. Pipeline execution runs on a background thread with real-time progress updates and a diagnostic log.

---

## Mathematical Foundation

### Regularized Lattice Regression

For a LUT with N^3 grid points, ChromaForge minimizes:

```
J(L) = sum_i  alpha_i * rho(||Phi(c_i, L) - c_i_out||)   [Data Fidelity]
     + lambda_s * ||Laplacian(L)||^2                       [Smoothness]
     + lambda_r * sum_j  beta_j * ||L_j - L0_j||^2        [Prior]
```

This produces a sparse linear least-squares system solved via LSMR with an outer IRLS loop for Huber robust loss. The three RGB channels are solved independently using shared interpolation weights, enabling thread-parallel execution.

### Interpolation Kernels

ChromaForge supports both **trilinear** (8-corner cube) and **tetrahedral** (4-vertex, 6-way branching) interpolation. The kernel used during fitting must match the runtime application. Tetrahedral is the default, matching the Adobe .cube specification and producing smoother gradients.

### Indexing Convention

All LUT arrays use shape `(N, N, N, 3)` indexed as `lut[r, g, b, channel]`. The flat index convention is `flat = b * N * N + g * N + r` (R varies fastest), matching the .cube file specification. This convention is defined once in `core/types.py` and used consistently throughout.

---

## Project Structure

```
src/chromaforge/
    __init__.py              Package root (version)
    __main__.py              python -m chromaforge entry point
    config.py                Constants, limits, default parameters
    errors.py                Custom exception hierarchy
    core/
        types.py             Indexing convention, enums, dataclasses
        interpolation.py     Trilinear + tetrahedral kernels (scalar + vectorized)
        laplacian.py         3D discrete Laplacian (loop + vectorized)
        matrix.py            Sparse system construction (COO -> CSR)
        solver.py            LSMR + IRLS, per-channel parallel solving
        lut.py               LUT operations (identity, apply, clip, stats, health)
        distance.py          Distance-to-data via EDT, prior strength
    pipeline/
        preprocess.py        Image loading, sanitization, TF detection
        sampling.py          Pixel binning, Welford stats, weight computation
        solving.py           Matrix build + solve orchestration
        refinement.py        Optional iterative refit
        validation.py        DeltaE 2000, coverage, LUT health metrics
        runner.py            Full pipeline orchestration with progress
    hald/
        identity.py          Hald CLUT identity generation (vectorized)
        reconstruct.py       LUT reconstruction from processed Hald
        resample.py          LUT resampling via scipy map_coordinates
    io/
        image.py             OIIO + imageio auto-detect, validation
        cube.py              .cube read/write with strict parsing
        arri.py              ARRI Reference Tool CLI wrapper
    color/
        metrics.py           DeltaE 2000, sRGB-to-Lab, Oklab
        shaper.py            Shaper LUT generation, monotonicity enforcement
        spaces.py            Transfer functions (LogC3, LogC4, S-Log3, V-Log)
    _numba_kernels/
        binning.py           JIT-compiled pixel binning (Welford)
        interpolation.py     JIT-compiled LUT interpolation
    cli/
        app.py               Typer CLI with 4 commands
    gui/
        app.py               QApplication entry point
        main_window.py       Main window with tab layout
        workers.py           QThread pipeline/Hald workers
        styles/
            theme.py         Dark theme palette, typography
            qss.py           Qt stylesheets
        widgets/
            image_pair.py    Synchronized dual image viewer
            parameters.py    Parameter panel with collapsible Advanced
            progress.py      Multi-stage pipeline progress display
            metrics_view.py  Quality metrics with status indicators
            coverage.py      2D slice coverage visualization
            log_viewer.py    Timestamped diagnostic log
tests/
    conftest.py              Shared fixtures
    test_interpolation.py    Kernel correctness, weight sums, continuity
    test_laplacian.py        Symmetry, row sums, null space, node degrees
    test_matrix.py           Dimensions, sqrt scaling, system assembly
    test_solver.py           Identity/gain recovery, IRLS convergence
    test_binning.py          Count conservation, mean correctness, Numba/NumPy
    test_hald.py             Round-trip, known transform, resample fidelity
    test_cube_io.py          Round-trip, ordering, malformed rejection
    test_pipeline.py         Full pipeline, cancellation, progress
    test_validation.py       Identity dE=0, health metrics
    test_shaper.py           Monotonicity, round-trip, range
    test_security.py         Path traversal, NaN propagation, indexing
```

---

## Quality Metrics

After extraction, ChromaForge reports:

| Metric | Description | Good | Fair | Poor |
|--------|-------------|------|------|------|
| Mean dE2000 | Average perceptual error | < 1.0 | < 3.0 | > 5.0 |
| Median dE2000 | Median perceptual error | < 1.0 | < 3.0 | > 5.0 |
| P95 dE2000 | 95th percentile error | < 3.0 | < 5.0 | > 5.0 |
| Max dE2000 | Worst-case error | < 3.0 | < 5.0 | > 5.0 |
| Total Variation | LUT smoothness measure | -- | -- | -- |
| Neutral Monotonic | Gray-axis monotonicity | Yes | -- | No |
| Out-of-Gamut | Percentage of OOG nodes | < 1% | < 5% | > 5% |
| Coverage | Bin occupancy percentage | > 10% | > 5% | < 5% |

---

## Performance

- **Numba JIT** for pixel binning (~20-50x over pure Python)
- **NumPy fallback** if Numba is unavailable
- **Vectorized** interpolation weight computation (batch all M samples)
- **Pre-computed** shared components reused across R/G/B channels
- **Thread-parallel** per-channel solving (scipy releases GIL during BLAS)
- **Subsampled** DeltaE validation (10% of pixels by default)

Typical extraction time for an 8MP image pair with 33^3 LUT: under 15 seconds.

---

## Security

- Input path validation with extension whitelist
- Image dimension limits before memory allocation (16K per side, 100MP total)
- LUT size cap (129^3 maximum)
- NaN/Inf sanitization on all image input
- Division-by-zero guards throughout (epsilon terms)
- .cube parser: line count limits, finite float validation, size bounds
- No eval/exec; ARRI tool invocation uses list-form subprocess (no shell injection)

---

## Testing

```
pip install ".[dev]"
pytest
```

The test suite covers:

- **Math correctness**: interpolation weights sum to 1, Laplacian symmetry and null space, solver identity/gain recovery
- **Round-trip integrity**: .cube write/read, Hald identity/reconstruct, LUT resample
- **Robustness**: NaN propagation, malformed file rejection, edge cases
- **Security**: path validation, dimension limits, indexing consistency
- **Pipeline integration**: end-to-end extraction, cancellation, progress callbacks

---

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Adobe Cube | .cube | Primary format. R-fastest ordering. |
| ARRI Look (AML) | .aml | Requires ARRI Reference Tool. |
| ARRI Look (ALF4) | .alf4 | Requires ARRI Reference Tool. 28-char filename limit. |

---

## Transfer Functions

ChromaForge includes built-in support for common camera log encodings:

| Encoding | Camera System |
|----------|---------------|
| ARRI LogC3 (EI 800) | Alexa Classic |
| ARRI LogC4 | Alexa 35 |
| Sony S-Log3 | Venice, FX series |
| Panasonic V-Log | VariCam, S1H |

Auto-detection is attempted based on image statistics. Manual override is available via `--transfer-fn` (CLI) or the Transfer Function dropdown (GUI).

---

## License

MIT
