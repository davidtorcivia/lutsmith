# LutSmith

Image-derived 3D LUT generation tool. LutSmith extracts color transformations from matched image pairs and generates industry-standard 3D LUTs compatible with DaVinci Resolve, ARRI cameras, and other post-production tools.

---

## Overview

LutSmith solves the inverse color-grading problem: given a source image and its graded variant, it reverse-engineers the color grade as a transferable 3D lookup table. The core algorithm uses **regularized lattice regression** with Laplacian smoothing, producing high-quality LUTs even from noisy or sparsely-sampled image data.

### Two Workflows

**Image-Pair Extraction** -- Provide a source image and its graded version. LutSmith extracts the color mapping via sparse regression and outputs a .cube LUT.

**Batch Pair Extraction** -- Provide many matched source/target frame pairs (for example restored->original film frames). LutSmith fits one robust aggregate LUT from all pair-derived samples.

**Scene-Clustered Batch Extraction** -- Split matched pairs into scene/style clusters (manual labels or auto clustering), then fit one LUT per cluster plus an optional master LUT.

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
lutsmith extract source.png graded.png -o output.cube
```

Key options:

```
  -o, --output PATH               Output file path [default: output.cube]
  -s, --size INT                   LUT grid size: 17, 33, or 65 [default: 33]
  -k, --kernel TEXT                Interpolation: tetrahedral or trilinear [default: tetrahedral]
  --smoothness FLOAT               Smoothness lambda_s [default: 0.1]
  --prior FLOAT                    Prior strength lambda_r [default: 0.01]
  --loss TEXT                      Loss function: l2 or huber [default: huber]
  --irls-iter INT                  IRLS iterations [default: 3]
  --bin-res INT                    Bin resolution per axis [default: 64]
  --min-samples INT                Minimum samples per bin [default: 3]
  --refine                         Enable iterative refinement
  -f, --format TEXT                Output format: cube, aml, alf4 [default: cube]
  -v, --verbose                    Enable debug logging
```

Advanced solver options:

```
  --prior-model TEXT               Prior model [default: identity]
  --color-basis TEXT               Regularization color space [default: rgb]
  --chroma-ratio FLOAT             Chroma-to-luma smoothness ratio [default: 4.0]
  --laplacian-connectivity INT     Laplacian connectivity: 6, 18, 26 [default: 6]
```

### Extract an aggregate LUT from many matched pairs

Create a manifest CSV with one
`source,target[,weight][,cluster][,transfer_fn][,normalization]` pair per line:

```
# pairs.csv
source,target,weight,cluster,transfer_fn,normalization
restored/frame_0001.png,original/frame_0001.png,1.0,scene_a,log_c4,none
restored/frame_0002.png,original/frame_0002.png,1.0,scene_a,log_c4,luma_affine
restored/frame_0003.png,original/frame_0003.png,0.7,scene_b,auto,rgb_affine
```

Then run:

```
lutsmith extract-batch pairs.csv -o restoration_to_original.cube --prior-model baseline_residual
```

Useful robustness controls:

```
  --pair-balance TEXT            equal | by_bins | by_pixels [default: equal]
  --outlier-sigma FLOAT          Reject high-error pairs via median + sigma*MAD [default: 0.0]
  --min-pairs-after-outlier INT  Minimum pairs to keep after rejection [default: 3]
  --allow-mixed-transfer         Permit mixed transfer-function detections across pairs
```

`extract-batch` uses the same solver options as `extract`, applies pair-aware weighting, and fits from the union of all pair bins. This is more robust than averaging finished LUT cubes.

Per-pair manifest overrides:

```
  transfer_fn: auto | linear | log_c3 | log_c4 | slog3 | vlog | unknown
  normalization: none | luma_affine | rgb_affine
```

Normalization is optional and intended to reduce cross-source drift when frame sets come from mixed scans, transcodes, or exposure pipelines.

Scene-clustered extraction options:

```
  --cluster-mode TEXT            none | manual | auto [default: none]
  --cluster-count INT            Fixed cluster count in auto mode (0 = automatic)
  --max-clusters INT             Upper bound for automatic cluster search [default: 6]
  --export-master/--no-export-master
                                 Export aggregate LUT in addition to per-cluster LUTs
  --cluster-seed INT             Random seed for auto clustering [default: 42]
  --metrics-csv PATH             Optional CSV summary for master/cluster metrics
```

Examples:

```
# Manual clusters from manifest column 4
lutsmith extract-batch pairs.csv -o look.cube --cluster-mode manual

# Auto scene clustering + master LUT + per-cluster LUTs
lutsmith extract-batch pairs.csv -o look.cube --cluster-mode auto --cluster-count 0 --max-clusters 8
```

### Generate a Hald identity image

```
lutsmith hald-gen -o identity.tiff -l 8
```

This produces a 512x512 identity image encoding a 64^3 LUT. Process it through your color grading pipeline, then reconstruct:

### Reconstruct LUT from processed Hald image

```
lutsmith hald-recon processed.tiff -o output.cube -l 8
```

Options:

```
  -l, --level INT          Hald level (8 = 512x512, 64^3 LUT) [default: 8]
  -s, --target-size INT    Resample to target LUT size (0 = native) [default: 0]
```

### Validate an existing LUT

```
lutsmith validate source.png graded.png existing.cube
```

Applies the LUT to the source image and reports DeltaE 2000 metrics against the target.

---

## GUI Usage

Launch the graphical interface:

```
lutsmith-gui
```

The GUI provides four tabs:

- **Image Pair** -- Load source/target images, adjust parameters, run extraction, and inspect quality metrics and coverage maps
- **Batch** -- Load or generate a manifest template, run aggregate extraction, set per-pair overrides, enable manual/auto scene clustering, export master + per-cluster LUTs, optionally write a metrics CSV summary, and review a sortable batch summary table (dE/coverage/time) with one-click output-folder open
- **Hald CLUT** -- Generate identity images, load processed results, and reconstruct LUTs
- **Settings** -- Configure output directory, LUT title, and view I/O backend status

The interface uses a dark theme with amber/gold accents. Pipeline execution runs on background threads with real-time progress updates and diagnostic logs. All solver parameters including prior model, color basis, chroma ratio, and Laplacian connectivity are accessible from the collapsible Advanced panel on the Image Pair tab; the Batch tab reuses these solver settings and adds batch-specific controls (pair balancing, outlier rejection, clustering mode/count, master export).

---

## Mathematical Foundation

### Regularized Lattice Regression

For a LUT with N^3 grid points, LutSmith minimizes:

```
J(L) = sum_i  alpha_i * rho(||Phi(c_i, L) - c_i_out||)   [Data Fidelity]
     + lambda_s * ||Laplacian(L)||^2                       [Smoothness]
     + lambda_r * sum_j  beta_j * ||L_j - L0_j||^2        [Prior]
```

Where:
- `Phi(c, L)` interpolates the LUT at color `c` using the selected kernel
- `rho` is the loss function (L2 or Huber)
- `alpha_i` are per-sample weights from the binning stage
- `beta_j` are per-node prior strengths based on distance-to-data
- `L0` is the prior LUT (identity, or a fitted baseline transform)

This produces a sparse linear least-squares system solved via LSMR with an outer IRLS loop for Huber robust loss. The three output channels are solved independently using shared interpolation weights, enabling thread-parallel execution.

### Interpolation Kernels

LutSmith supports both **trilinear** (8-corner cube) and **tetrahedral** (4-vertex, 6-way branching) interpolation. The kernel used during fitting must match the runtime application. Tetrahedral is the default, matching the Adobe .cube specification and producing smoother gradients.

### Indexing Convention

All LUT arrays use shape `(N, N, N, 3)` indexed as `lut[r, g, b, channel]`. The flat index convention is `flat = b * N * N + g * N + r` (R varies fastest), matching the .cube file specification. This convention is defined once in `core/types.py` and used consistently throughout.

---

## Prior Models

The prior determines what the solver assumes about LUT nodes far from observed data. LutSmith supports three prior models, selectable via `--prior-model` (CLI) or the Prior Model dropdown (GUI).

### Identity Prior (default)

The classic approach: unseen LUT regions fall back to the identity transform (no change). Works well when the color grade is subtle or the image pair covers most of the color gamut.

### Baseline Residual

Instead of solving for the full LUT directly, LutSmith first fits a parametric baseline transform:

```
T0(x) = h(Mx + b)
```

Where:
- `M` is a 3x3 affine matrix (handles cross-channel coupling, saturation, white balance)
- `b` is a 3-vector bias (handles lift/offset)
- `h` is a set of per-channel monotone curves (handles gamma, contrast, toe/shoulder)

The fitting uses alternating optimization: fix the curves and solve for `M, b` via weighted IRLS least-squares, then fix `M, b` and fit `h` via binning followed by the Pool Adjacent Violators Algorithm (PAVA) for isotonic regression. This alternates for 3 iterations by default.

After fitting, a quality gate compares the baseline's weighted MSE against the identity prior. The baseline is used only if it achieves at least 5% lower MSE -- this prevents degenerate baselines on identity-like grades from degrading results.

When the baseline passes, the solver operates on residuals:

```
solve_target = output_rgb - T0(input_rgb)
prior_lut = zeros
```

The prior pulls unseen regions toward zero residual, meaning the final LUT `T0 + deltaL` gracefully extrapolates through the baseline in unmapped regions. This substantially improves extrapolation for strong color grades, creative looks, and film emulations where the identity prior would produce visible artifacts.

### Baseline + Multigrid

Same as baseline residual, but the residual solve uses a coarse-to-fine multigrid strategy. The solver starts at a coarse grid (default 17^3), solves for the residual, upsamples via trilinear interpolation, then solves again for the remaining residual at the next finer level. This repeats until reaching the target grid size.

At each level, smoothness is scaled by `multigrid_smoothness_scale^(levels_remaining - 1)`, so coarse levels use stronger smoothing (capturing the global trend) while the finest level uses the configured smoothness (capturing local detail).

Example schedules:
- N=33, coarse=17: [17, 33] (2 levels)
- N=65, coarse=17: [17, 33, 65] (3 levels)

Multigrid is most beneficial for large LUTs (65^3) with sparse data, where single-level solving may produce ringing or noise in uncovered regions.

---

## Opponent-Space Regularization

By default, the solver regularizes each RGB channel independently with the same smoothness weight. This can cause two issues:
1. Color fringing in sparsely-sampled regions (independent channel noise couples into visible chroma artifacts)
2. Over-smoothing of luminance detail when increasing smoothness to suppress chroma noise

Setting `--color-basis opponent` transforms the problem into an opponent color space before regularization:

```
Y  = (R + G + B) / sqrt(3)      luminance
C1 = (R - G)     / sqrt(2)      red-green chrominance
C2 = (R + G - 2B) / sqrt(6)     yellow-blue chrominance
```

The basis is orthonormal, so its inverse is its transpose and no information is lost in the transform. In this space, the solver applies:
- Standard smoothness (`lambda_s`) to the luminance channel Y
- Higher smoothness (`lambda_s * chroma_ratio`) to the chrominance channels C1, C2

The default `chroma_ratio` of 4.0 means chroma is smoothed 4x more aggressively than luma. This preserves luminance detail (contrast, tonal gradation) while suppressing chroma noise in sparse regions.

### Neutral-Aware Chroma Prior

When using opponent-space regularization, a neutral-aware boost is automatically applied to the chrominance prior. LUT nodes near the achromatic axis (R == G == B) receive a stronger prior pull toward zero chroma, discouraging spurious color shifts in grays and near-neutrals:

```
boost(node) = 1 + k * exp(-d^2 / (2 * sigma^2))
```

Where `d` is the perpendicular distance from the neutral axis in RGB space, `k` (default 3.0) controls the peak boost factor, and `sigma` (default 0.12) controls how quickly the boost falls off.

This is particularly useful for preserving neutral skin tones and preventing gray-axis color casts in regions with sparse sample coverage.

---

## Extended Laplacian Connectivity

The smoothness regularization uses a discrete 3D Laplacian to penalize differences between neighboring LUT nodes. The `--laplacian-connectivity` option controls the neighborhood definition:

| Connectivity | Neighbors | Description |
|-------------|-----------|-------------|
| 6 (default) | Face-adjacent | Standard cubic stencil. Fastest. |
| 18 | Face + edge-diagonal | 12 additional neighbors weighted by `1/sqrt(2)`. Less axis-aligned bias. |
| 26 | Face + edge + corner | 8 additional corner neighbors weighted by `1/sqrt(3)`. Most isotropic smoothing. |

The extended Laplacians are normalized so that an interior node's weighted degree remains 6.0, keeping the meaning of `lambda_s` consistent regardless of connectivity. Higher connectivity produces rounder, more isotropic smoothing at the cost of a denser sparse matrix (more non-zeros per row).

---

## Shadow Handling

Dark regions in images tend to have few distinct samples, causing noise and crush artifacts in the corresponding LUT nodes. LutSmith addresses this with shadow-aware Laplacian boosting.

When enabled (default: auto-detected), nodes in the shadow region receive up to 25x stronger smoothness regularization. The boost ramps smoothly from maximum at the deep shadow threshold to 1.0 at the shadow threshold, using Hermite smoothstep interpolation to avoid discontinuities:

```
deep_threshold = 0.08   (maximum boost below this luminance)
shadow_threshold = 0.25  (no boost above this luminance)
boost = 25.0             (maximum smoothness multiplier)
```

Auto mode estimates these thresholds from the input image statistics. Manual thresholds can be set via `--shadow-threshold` and `--deep-shadow-threshold` (CLI) or the Shadow controls (GUI).

---

## Pipeline Stages

LutSmith processes image pairs through six sequential stages:

### 1. Preprocess

Load source and target images (auto-detecting the I/O backend: OIIO if available, imageio as fallback). Sanitize pixel values (clamp to [0, 1], replace NaN/Inf). Auto-detect the camera transfer function from image statistics, and generate a 1D shaper LUT if the input is scene-linear.

### 2. Sample

Bin all pixel pairs into a 64^3 grid using Welford online accumulation (Numba-JIT compiled when available, ~20-50x faster than pure Python). Compute per-bin statistics (mean input/output, variance, count, spatial centroid). Detect spatial inconsistency indicating vignette or localized grading. Filter bins below the minimum sample count, compute adaptive per-sample weights, and flatten into regression sample arrays.

### 3. Solve

Build the sparse stacked system matrix (data fidelity + smoothness + prior) and solve via LSMR with IRLS outer loop. The three output channels are solved independently with shared interpolation weights and Laplacian. If a baseline prior model is selected, fit the baseline first, run the quality gate, and solve for residuals. If multigrid is selected, iterate through the coarse-to-fine schedule. Post-solve, clamp the LUT to [-0.5, 1.5].

### 4. Refine (optional)

Apply the current LUT to source samples, compute per-sample residuals against the target, identify high-residual outliers (above 2x P95), downweight them, and refit with slightly reduced smoothness. This can improve accuracy when the image pair contains localized artifacts (reflections, motion blur, text overlays) that the initial IRLS pass didn't fully suppress.

### 5. Validate

Compute DeltaE 2000 metrics on a subsample (10% by default): mean, median, P95, and max. Calculate total variation (smoothness measure), check neutral-axis monotonicity, compute out-of-gamut percentage, and measure bin coverage.

### 6. Export

Write the LUT to the selected format (.cube, .aml, .alf4). Optionally append a 1D shaper LUT for scene-linear inputs. Generate a coverage report.

---

## Quality Metrics

After extraction, LutSmith reports:

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
- **Baseline prior**: PAVA monotonicity, affine recovery, curve inversion, quality gate
- **Opponent space**: basis orthonormality, RGB roundtrip, neutral-axis zero-chroma, boost bounds
- **Extended Laplacian**: symmetry, row sums, null space, interior degree normalization for 18/26 connectivity
- **Prior boost**: scaling verification, identity (no-op) checks

---

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Adobe Cube | .cube | Primary format. R-fastest ordering. Optional 1D shaper. |
| ARRI Look (AML) | .aml | Requires ARRI Reference Tool. |
| ARRI Look (ALF4) | .alf4 | Requires ARRI Reference Tool. 28-char filename limit. |

---

## Transfer Functions

LutSmith includes built-in support for common camera log encodings:

| Encoding | Camera System |
|----------|---------------|
| ARRI LogC3 (EI 800) | Alexa Classic |
| ARRI LogC4 | Alexa 35 |
| Sony S-Log3 | Venice, FX series |
| Panasonic V-Log | VariCam, S1H |

Auto-detection is attempted based on image statistics. Manual override is available via `--transfer-fn` (CLI) or the Transfer Function dropdown (GUI).

When the input uses a log encoding, LutSmith generates a 1D shaper LUT (linearize-before, log-after) that is embedded in the .cube output. For scene-linear inputs, a log2-based shaper distributes precision across the dynamic range.

---

## Project Structure

```
src/lutsmith/
    __init__.py              Package root (version)
    __main__.py              python -m lutsmith entry point
    config.py                Constants, limits, default parameters
    errors.py                Custom exception hierarchy
    core/
        types.py             Indexing convention, enums, dataclasses
        interpolation.py     Trilinear + tetrahedral kernels (scalar + vectorized)
        laplacian.py         3D discrete Laplacian (6/18/26 connectivity)
        matrix.py            Sparse system construction (COO -> CSR)
        solver.py            LSMR + IRLS, per-channel parallel solving
        lut.py               LUT operations (identity, apply, clip, stats, health)
        distance.py          Distance-to-data via EDT, prior strength
        baseline.py          Baseline transform fitting (PAVA, alternating opt.)
        opponent.py          Opponent color space (orthonormal basis, neutral boost)
    pipeline/
        preprocess.py        Image loading, sanitization, TF detection
        sampling.py          Pixel binning, Welford stats, weight computation
        batch_manifest.py    CSV manifest parsing (source/target/weight/cluster/overrides)
        clustering.py        Pair-signature extraction + k-means scene clustering
        normalization.py     Per-pair normalization (none/luma_affine/rgb_affine)
        reporting.py         Batch metrics table generation + CSV export
        solving.py           Matrix build, baseline/multigrid orchestration
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
        app.py               Typer CLI with extraction, batch extraction, Hald, validation commands
    gui/
        app.py               QApplication entry point
        main_window.py       Main window with tab layout
        workers.py           QThread workers for image-pair, batch, and Hald workflows
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
    test_laplacian.py        Symmetry, row sums, null space, node degrees (6/18/26)
    test_matrix.py           Dimensions, sqrt scaling, system assembly, prior boost
    test_solver.py           Identity/gain recovery, IRLS, opponent mode, backward compat
    test_binning.py          Count conservation, mean correctness, Numba/NumPy
    test_hald.py             Round-trip, known transform, resample fidelity
    test_cube_io.py          Round-trip, ordering, malformed rejection
    test_pipeline.py         Full pipeline, cancellation, progress
    test_validation.py       Identity dE=0, health metrics
    test_shaper.py           Monotonicity, round-trip, range
    test_security.py         Path traversal, NaN propagation, indexing
    test_baseline.py         PAVA, affine recovery, curves, quality gate
    test_opponent.py         Basis orthonormality, roundtrip, neutral axis, boost
```

---

## Configuration Reference

### Default Parameter Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lut_size` | 33 | Grid nodes per axis (33^3 = 35,937 nodes) |
| `kernel` | tetrahedral | Interpolation kernel |
| `smoothness` | 0.1 | Laplacian regularization weight (lambda_s) |
| `prior_strength` | 0.01 | Prior regularization weight (lambda_r) |
| `robust_loss` | huber | Robust loss for outlier handling |
| `huber_delta` | 1.0 | Huber transition point |
| `irls_iterations` | 3 | IRLS outer loop iterations |
| `bin_resolution` | 64 | Bins per axis for pixel aggregation |
| `min_samples_per_bin` | 3 | Minimum count to include a bin |
| `prior_model` | identity | Prior model for unseen regions |
| `color_basis` | rgb | Regularization color space |
| `chroma_smoothness_ratio` | 4.0 | Chroma-to-luma smoothness (opponent mode) |
| `laplacian_connectivity` | 6 | Neighbor count per node |
| `multigrid_coarse_size` | 17 | Coarsest grid in multigrid schedule |
| `multigrid_smoothness_scale` | 2.0 | Smoothness scale factor per coarse level |
| `shadow_boost` | 25.0 | Laplacian multiplier for deep shadows |

### Safety Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| MAX_IMAGE_DIMENSION | 16,384 | Pixels per side |
| MAX_IMAGE_PIXELS | 100,000,000 | Total pixel count |
| MAX_CUBE_SIZE | 129 | Maximum LUT grid size |
| MAX_LSMR_ITERATIONS | 5,000 | Solver iteration cap |
| MAX_IRLS_ITERATIONS | 20 | IRLS iteration cap |
| LUT_CLAMP_RANGE | [-0.5, 1.5] | Post-solve value clamp |

---

## Practical Guidance

### Choosing a Prior Model

- **Identity** (default): Best for subtle grades, film print emulations, and situations where the image pair provides good gamut coverage. Lowest computational cost.
- **Baseline Residual**: Use for strong color grades (heavy lift/gamma/gain, cross-channel mixing, contrast curves). Provides much better extrapolation in unmapped regions. Adds a few seconds for baseline fitting.
- **Baseline + Multigrid**: Use for large LUTs (65^3) with sparse data. The coarse-to-fine approach reduces ringing and produces smoother results in unseen regions. Longer solve time due to multiple passes.

### Choosing a Color Basis

- **RGB**: Use for most situations. Independent per-channel regularization. Simplest and fastest.
- **Opponent**: Use when you see chroma fringing or color noise in sparsely-sampled regions (common with creative looks that shift only certain hue ranges). The anisotropic smoothing preserves luminance detail while suppressing chroma artifacts. The neutral-aware boost also helps prevent gray-axis color casts.

### Choosing Laplacian Connectivity

- **6** (default): Standard cubic stencil. Fast, predictable behavior. Fine for most use cases.
- **18**: Reduces axis-aligned smoothing bias. Consider if you notice grid-aligned artifacts in the LUT output.
- **26**: Most isotropic smoothing. Use for critical work where smoothness isotropy matters. Slightly higher memory and solve cost.

### Shadow Handling

Leave shadow auto-detection enabled for most inputs. Consider manual thresholds when:
- Working with high-key images (few shadow samples -- may want to lower the shadow threshold)
- Working with low-key/night footage (raise the shadow threshold to cover more of the tonal range)
- The auto-detected thresholds produce visible transitions (adjust thresholds to match the image content)

---

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-or-later).
