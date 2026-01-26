# ChromaForge: Technical Design Document

## Image-Derived 3D LUT Generation Tool
**Version:** 1.1  
**Status:** Implementation Specification (Post-Review)  
**Date:** January 2026

---

## 1. Project Summary

ChromaForge extracts color transformations from matched image pairs and generates industry-standard 3D LUTs. The tool solves the inverse problem: given a source image and its graded variant, reverse-engineer the color grade as a transferable lookup table.

### Primary Workflow (Image-Pair Extraction)
1. User provides source image + graded version of same image
2. ChromaForge extracts the color mapping via regularized regression
3. Outputs a 3D LUT usable in DaVinci Resolve, ARRI cameras, etc.

### Secondary Workflow (Identity Plate Mode)
1. User processes a Hald CLUT identity image through their grading pipeline
2. ChromaForge directly reconstructs the LUT from the processed identity
3. Provides guaranteed full-gamut coverage with no interpolation needed

> **Note:** Identity Plate Mode ships in Phase 1—it provides ground truth for testing exporters and immediate value to users while regression math is refined.

As a guiding rule when working on this project we must NEVER EVER use emojis (unicode and ASCII symbols are fine) and the final GUI must be beautiful and well crafted.

---

## 2. Mathematical Architecture

### 2.1 Core Approach: Regularized Lattice Regression

The core recommendation from both engineering reviews: **do not interpolate—regress.** Treat the LUT grid points as unknowns in a global optimization problem.

For a LUT with N³ grid points (e.g., 33³ = 35,937), we minimize:

```
J(L) = Σᵢ αᵢ · ρ(‖Φ(cᵢⁿ, L) - cᵢᵒᵘᵗ‖)     [Data Fidelity]
     + λₛ · ‖∇²L‖²                          [Smoothness]  
     + λᵣ · Σⱼ βⱼ · ‖Lⱼ - L⁰ⱼ‖²            [Prior/Identity Bias]
```

Where:
- **L** = vector of all LUT grid point values (N³ × 3 for RGB)
- **Φ(cᵢⁿ, L)** = interpolation of input color through current LUT
- **ρ** = robust loss function (Huber, requires IRLS—see §2.4)
- **∇²L** = discrete 3D Laplacian (smoothness constraint)
- **L⁰** = prior LUT (identity by default)
- **αᵢ** = per-sample weight (confidence, frequency)
- **βⱼ** = per-node prior strength (increases with distance from data)

### 2.2 Critical: Interpolation Kernel Must Match Runtime

**Problem:** Different applications use different 3D LUT interpolation methods. If you fit assuming trilinear but the runtime uses tetrahedral (or vice versa), you introduce systematic error.

**Adobe .cube spec is written around tetrahedral interpolation.** DaVinci Resolve defaults to trilinear but offers tetrahedral. Research shows tetrahedral produces smoother gradients and ~20% smaller LUTs can achieve equivalent quality.

**Solution:** Make interpolation kernel a first-class parameter:

| Export Target | Default Kernel | Rationale |
|---------------|----------------|-----------|
| .cube | **Tetrahedral** | Per Adobe spec; more accurate |
| DaVinci (explicit) | User choice | Resolve exposes this setting |
| ARRI cameras | Trilinear | Camera firmware typically uses trilinear |

**Implementation:**
- `interpolation_kernel` parameter: `tetrahedral` (default) or `trilinear`
- Fitting uses the specified kernel for weight computation
- Validation applies the same kernel used during fitting
- Document clearly: "This LUT was optimized for [X] interpolation"

### 2.3 The Linear System (Corrected)

The optimization reduces to solving a sparse linear least squares system. **Critical correction from review:** for proper LSQ formulation, rows must be scaled by **sqrt(weight)**, not weight.

**System structure:**
```
┌                    ┐   ┌   ┐     ┌                    ┐
│ sqrt(Wᵈ) · Aᵈ     │   │ L │     │ sqrt(Wᵈ) · bᵈ     │
│ sqrt(λₛ) · Aˢ     │ × │   │  =  │ 0                  │
│ sqrt(λᵣ·B) · I    │   │   │     │ sqrt(λᵣ·B) · L⁰   │
└                    ┘   └   ┘     └                    ┘
```

Where:
- **Wᵈ** = diagonal matrix of per-sample weights αᵢ
- **Aᵈ** = data constraint matrix (interpolation weights)
- **Aˢ** = smoothness constraint matrix (discrete 3D Laplacian)
- **B** = diagonal matrix of per-node prior strengths βⱼ
- **L⁰** = prior LUT values

**Per-channel solving:** Build matrix A with **N³ columns** (not 3×N³). Solve three times for R, G, B independently. This is simpler, parallelizable, and avoids artificial cross-channel coupling.

> **Important:** The input features are always the full (r,g,b) triplet for computing interpolation weights, even when solving for a single output channel. This naturally captures cross-channel effects (e.g., "add cyan to shadows").

### 2.4 Robust Loss Requires IRLS (Iteratively Reweighted Least Squares)

**Correction from review:** Plain `lsqr`/`lsmr` minimizes L2 (squared) error. To get Huber or other robust losses, you need an outer IRLS loop:

```python
def irls_solve(A, b, loss='huber', delta=1.0, max_iter=5, tol=1e-4):
    """Iteratively Reweighted Least Squares for robust regression."""
    x = lsmr(A, b)[0]  # Initial L2 solution
    
    for iteration in range(max_iter):
        residuals = A @ x - b
        
        # Huber weights: w_i = 1 if |r| < delta, else delta/|r|
        weights = np.where(
            np.abs(residuals) < delta,
            1.0,
            delta / (np.abs(residuals) + 1e-8)
        )
        
        # Reweight and solve
        W_sqrt = sparse.diags(np.sqrt(weights))
        x_new = lsmr(W_sqrt @ A, W_sqrt @ b)[0]
        
        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-8) < tol:
            break
        x = x_new
    
    return x
```

**Practical guidance:**
- 3-5 IRLS iterations typically sufficient
- Can push robustness entirely to binning/rejection phase if IRLS is too slow
- Be explicit in docs: "Robust fitting requires IRLS outer loop"

### 2.5 Why Lattice Regression Works

The formulation from Garcia & Gupta (2009, 2010) was developed for ICC color profile construction. Key insight: **construct the LUT while accounting for how it will be interpolated at runtime.**

This approach:
- Handles sparse sampling naturally (regularization fills gaps)
- Is noise-robust (robust loss + aggregation)
- Produces smooth output (Laplacian term)
- Degrades gracefully in unobserved regions (prior term)
- Handles interpolation and extrapolation simultaneously

---

## 3. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CHROMAFORGE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  PREPROCESS  │    │   SAMPLE     │    │    SOLVE     │              │
│  │              │───▶│              │───▶│              │              │
│  │ • Decode     │    │ • Bin pixels │    │ • Build A    │              │
│  │ • Sanitize   │    │ • Aggregate  │    │ • IRLS solve │              │
│  │ • Align      │    │ • Weight     │    │ • Extract L  │              │
│  │ • Shaper     │    │ • Detect     │    │              │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   VALIDATE   │    │   REFINE     │    │   EXPORT     │              │
│  │              │◀───│  (optional)  │◀───│              │              │
│  │ • ΔE metrics │    │ • Residuals  │    │ • .cube      │              │
│  │ • Coverage   │    │ • Reweight   │    │ • .aml/.alf4 │              │
│  │ • Health     │    │ • Refit      │    │ • .clf       │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage Details

### 4.1 Preprocessing

#### 4.1.1 Image Decoding
- Use **OpenImageIO (OIIO)** for image I/O—properly handles 10-bit, 16-bit, EXR, DPX
- Avoid OpenCV for decoding (mishandles bit depth and color profiles)
- Preserve full precision; work internally in float32

#### 4.1.2 Input Sanitization (Critical)

**NaN/Inf Poisoning:** A single NaN in the sparse matrix will propagate through the solver and corrupt the entire LUT.

```python
def sanitize_image(img):
    """Remove NaN/Inf values that would poison the solver."""
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, None)  # Clamp negative values
    return img
```

**Required for:** EXR files (commonly contain NaN/Inf), any floating-point input.

#### 4.1.3 Alignment (Optional)
- If images may have minor spatial shift, offer optional alignment pass
- Feature matching + RANSAC or ECC (Enhanced Correlation Coefficient)
- Make it optional and report alignment quality/confidence
- If alignment fails, warn user and proceed with direct pairing

#### 4.1.4 Shaper LUT Generation (Critical for HDR/Log)

**The Problem:** Log-encoded footage concentrates most data in a narrow range. A linear 3D LUT wastes grid resolution.

**Critical Distinction (from review):**

| Input Type | Shaper Action |
|------------|---------------|
| **Scene Linear** (EXR, 0.0-50.0+) | Generate log shaper to compress into [0,1] |
| **Already Log** (LogC3, S-Log3, etc.) | Use **identity** or simple range normalization |

> **Warning:** Applying a log shaper to already-log footage "double-logs" the data, crushing shadows.

**Implementation:**

```python
input_transfer_function: Literal['linear', 'log_c3', 'log_c4', 'slog3', 'vlog', 'unknown']

def get_shaper(input_tf: str, histogram: np.ndarray) -> Callable:
    if input_tf == 'linear':
        # Generate log-style shaper based on histogram
        return generate_adaptive_log_shaper(histogram)
    elif input_tf in ['log_c3', 'log_c4', 'slog3', 'vlog']:
        # Already log-encoded: identity or simple normalization
        return identity_shaper()
    else:
        # Unknown: use conservative fixed shaper or ask user
        return conservative_log_shaper()
```

**Shaper Monotonicity Enforcement:**

ARRI cameras reject non-monotonic shapers. **Enforce, don't just check:**

```python
from sklearn.isotonic import IsotonicRegression

def enforce_monotonic_shaper(shaper_values):
    """Ensure shaper curve is strictly monotonic."""
    ir = IsotonicRegression(increasing=True)
    return ir.fit_transform(np.arange(len(shaper_values)), shaper_values)
```

**Format Support for Shapers:**

Contrary to earlier draft, **.cube CAN support 1D+3D shaper combinations** in DaVinci Resolve. However, cross-application support is messy.

Recommendation:
- Default to **known shapers** for known encodings (LogC3/4, S-Log3, etc.) via OCIO definitions
- **Adaptive shaper** as advanced option only (makes LUT image-dependent, surprising to users)

### 4.2 Sampling

#### 4.2.1 Weighted Voxel Hashing (Binning)

**Do not process raw pixels in Python loops.** This will miss performance targets.

**Performance requirement:** 8M pixels must bin in <2 seconds.

```python
# WRONG: Pure Python loop (30-60 seconds for 8M pixels)
for pixel in pixels:
    bin_idx = compute_bin(pixel)
    bins[bin_idx].append(pixel)

# RIGHT: Vectorized NumPy + optional Numba JIT
@numba.jit(nopython=True, parallel=True)
def bin_pixels_fast(source_rgb, target_rgb, bin_resolution):
    """Numba-accelerated pixel binning."""
    # ... vectorized implementation
```

**Alternative:** Use `np.digitize` + `np.bincount` for pure-NumPy solution.

**Bin aggregation collects:**
- `sum_input_rgb`: Running sum of input colors
- `sum_output_rgb`: Running sum of output colors  
- `count`: Number of samples
- `sum_sq_output`: For variance computation (Welford's algorithm)
- `sum_xy_coords`: For spatial correlation (vignette detection)

#### 4.2.2 Outlier Handling (Revised)

**Correction from review:** With streaming Welford stats alone, you cannot actually remove per-pixel outliers within a bin—you only have aggregate statistics.

**Two options:**

**Option A (Lean, recommended for v1):**
- Use variance to **downweight bins**, not remove individual pixels
- Push robustness to IRLS at constraint level
- Document limitation: "Per-pixel outlier removal requires two-pass mode"

**Option B (More accurate, optional):**
- Two-pass approach: first pass computes mean/variance, second pass rejects outliers
- Or: reservoir sampling to keep subset of samples per bin

| Condition | Action |
|-----------|--------|
| Near black/white (< 0.02 or > 0.98) | Reduce weight (clipped = no info) |
| Low count (< threshold) | Discard (noise) |
| High variance | Reduce weight via confidence factor |

#### 4.2.3 Vignette/Spatial Inconsistency Detection

**Problem:** Vignettes cause the same input color to map differently based on spatial position (center vs corners).

**Solution:** Correlate variance with spatial coordinates:

```python
def detect_spatial_inconsistency(bin_data):
    """Check if high variance correlates with spatial position."""
    if bin_data.variance < threshold:
        return False
    
    # Compute correlation between output error and (x, y) position
    spatial_correlation = compute_spatial_correlation(
        bin_data.output_residuals,
        bin_data.pixel_positions
    )
    
    if spatial_correlation > spatial_threshold:
        return 'vignette_suspected'
    return 'local_adjustment_suspected'
```

**Action:** If spatial inconsistency detected, discard these bins to prevent muddy averaging.

#### 4.2.4 Contradiction Detection (Corrected Scoring)

**Problem with original formula:** `variance / mean_output_magnitude` explodes near black.

**Fixed formula:**

```python
def contradiction_score(variance, mean_output, epsilon=0.05):
    """Score indicating likelihood of local adjustment."""
    # Normalize by (mean + epsilon) to avoid dark-bin blow-up
    return variance / (np.linalg.norm(mean_output) + epsilon)
```

**Separate diagnostics:**
- **"Local adjustment suspicion"**: High variance, good alignment, no spatial correlation
- **"Possible misalignment"**: High variance, low alignment confidence, spatially structured residuals

#### 4.2.5 Sample Weighting

**Coverage-fair (default):** Each occupied bin contributes roughly equally.

```python
α_i = α_base * confidence(variance_i, count_i) * clip_penalty(input_i)

def confidence(variance, count, k=0.5):
    """Higher count and lower variance = more confidence."""
    return (1.0 / (1.0 + k * variance)) * min(1.0, count / min_count)
```

**Frequency-weighted:** Weight by pixel count. Better for matching specific image distribution.

### 4.3 Solving

#### 4.3.1 Matrix Construction (Corrected)

**Use COO format for construction, convert to CSR for solving:**

```python
from scipy.sparse import coo_matrix, csr_matrix

def build_system(samples, N, λ_s, λ_r, kernel='tetrahedral'):
    """Build sparse linear system for lattice regression."""
    rows, cols, vals = [], [], []
    b = []
    row_idx = 0
    
    # === Data fidelity rows ===
    for sample in samples:
        corners, weights = interpolation_weights(
            sample.input_rgb, N, kernel=kernel
        )
        sqrt_alpha = np.sqrt(sample.alpha)
        
        for corner, weight in zip(corners, weights):
            rows.append(row_idx)
            cols.append(corner)
            vals.append(sqrt_alpha * weight)
        
        b.append(sqrt_alpha * sample.output_channel)
        row_idx += 1
    
    # === Smoothness rows (Laplacian) ===
    sqrt_λ_s = np.sqrt(λ_s)
    for node in range(N**3):
        neighbors = get_neighbors_3d(node, N)
        rows.append(row_idx)
        cols.append(node)
        vals.append(sqrt_λ_s * len(neighbors))
        
        for neighbor in neighbors:
            rows.append(row_idx)
            cols.append(neighbor)
            vals.append(-sqrt_λ_s)
        
        b.append(0.0)
        row_idx += 1
    
    # === Prior rows ===
    sqrt_λ_r = np.sqrt(λ_r)
    for node in range(N**3):
        sqrt_beta = np.sqrt(prior_strength(node, samples))
        rows.append(row_idx)
        cols.append(node)
        vals.append(sqrt_λ_r * sqrt_beta)
        b.append(sqrt_λ_r * sqrt_beta * identity_value(node, N))
        row_idx += 1
    
    A = coo_matrix((vals, (rows, cols))).tocsr()
    return A, np.array(b)
```

#### 4.3.2 Distance-to-Data Computation

For prior strength βⱼ, compute per-node distance to nearest sample efficiently:

```python
def compute_distance_to_data(occupied_bins, N):
    """BFS/distance transform from occupied cells - O(N³)."""
    grid = np.full((N, N, N), np.inf)
    
    for bin_idx in occupied_bins:
        r, g, b = idx_to_rgb(bin_idx, N)
        grid[r, g, b] = 0
    
    from scipy.ndimage import distance_transform_edt
    distances = distance_transform_edt(grid > 0)
    
    return distances.flatten()
```

#### 4.3.3 Solver Selection

**Recommendation:** `scipy.sparse.linalg.lsmr` over `lsqr`

- Often converges faster on ill-conditioned systems
- Same interface as lsqr
- Better numerical stability for regularized problems

```python
from scipy.sparse.linalg import lsmr

x, istop, itn, normr, normar, norma, conda, normx = lsmr(
    A, b,
    damp=0.0,
    atol=1e-6,
    btol=1e-6,
    maxiter=1000
)
```

### 4.4 Refinement (Optional Iterative Loop)

```
1. Initial fit with conservative λ (more smoothing)
2. Apply LUT to source, compute residuals vs target
3. Identify bins with systematic large residuals
4. Downweight or drop those bins
5. Refit with slightly lower λ if coverage is good
6. Repeat 1-2 times
```

### 4.5 Validation & Quality Metrics

#### 4.5.1 Reconstruction Error

**Critical:** Apply LUT using the **same interpolation kernel** used during fitting.

- Mean ΔE₂₀₀₀ (perceptual, human-interpretable)
- 95th percentile ΔE
- Max ΔE with location

#### 4.5.2 Coverage Map

Compute per-node distance to nearest sample:
- **Green:** Dense data (distance < 2 grid steps)
- **Yellow:** Sparse data (2-5 grid steps)
- **Red:** Extrapolated (> 5 grid steps)

#### 4.5.3 LUT Health Metrics

**Total Variation (TV):** Sum of absolute differences between adjacent nodes. High TV = jagged LUT.

**Monotonicity on Neutral Axis:** Gray ramp should be monotonic. Reversals indicate problems.

**Out-of-Gamut Percentage:** Nodes with values outside [0,1].

**Kernel Sensitivity Test:** Optionally apply LUT with "wrong" kernel and report delta.

### 4.6 Export

#### 4.6.1 Format Support (Revised Priority)

| Format | Extension | Target | Priority | Notes |
|--------|-----------|--------|----------|-------|
| Adobe Cube | .cube | Universal | **P0** | Tetrahedral-optimized by default |
| ARRI Look File 2 | .aml | Alexa Mini/LF, Amira | **P0** | Via ARRI Reference Tool |
| ARRI Look File 4 | .alf4 | Alexa 35, 265 | **P1** | Via ARRI Reference Tool |
| ACES CLF | .clf | ACES ecosystem | **P2** | Use AMPAS reference impl |

#### 4.6.2 .cube Implementation

```
LUT_3D_SIZE 33
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0

# Optimized for tetrahedral interpolation
0.000000 0.000000 0.000000
0.031250 0.000000 0.000000
...
```

#### 4.6.3 ARRI Export Strategy (Revised)

**Strong recommendation from review:** Use ARRI's own tooling rather than DIY XML generation.

ARRI Reference Tool includes a **`look-builder` executable** that creates .aml/.alf4/.alf4c from .cube files.

**Implementation approach:**

```
P0: .cube export + document "use ARRI Reference Tool to convert"
P1: Optional integration with look-builder CLI if installed
P2: Native XML export only if standalone requirement emerges
```

**ALF2 (.aml) semantics:**
- Combines creative grade (CDL or 3D LUT) **with** a Display Render Transform to target space
- Operates LogC3 → target (e.g., Rec709)
- Filename limit: 28 characters including extension (camera filesystem constraint)

**ALF4 (.alf4) semantics:**
- Separates creative grade from DRT
- Creative Modification Transform (CMT) is **Log-to-Log** (LogC4 → LogC4)
- Camera applies DRT separately based on output setting
- More flexible for multiple deliverables

#### 4.6.4 CLF Implementation

Use AMPAS CLF reference implementation rather than building from scratch.

---

## 5. Identity Plate Mode (Hald CLUT Workflow)

### 5.1 Rationale

**Moved to Phase 1** per engineering review:
- Provides ground truth for testing .cube/.aml exporters
- Zero regression complexity—direct pixel-to-node mapping
- Immediate value to users (Hald workflow is popular)
- "If you can't reconstruct perfectly from identity, you can't trust regression"

### 5.2 How It Works

1. **Generate identity image:** Hald CLUT pattern
2. **User grades the identity:** Process through same pipeline as footage
3. **Direct reconstruction:** Each pixel position maps directly to a LUT node

### 5.3 Hald Size vs LUT Size (Important)

**Hald levels produce specific LUT sizes:**

| Hald Level | Image Size | LUT Size |
|------------|------------|----------|
| 8 | 512×512 | 64³ |
| 12 | 1728×1728 | 144³ |

**Problem:** Common LUT sizes (33³, 65³) don't have exact Hald representations.

**Solution:** Reconstruct at native Hald resolution (e.g., 64³), then **resample** to target size:

```python
def resample_lut(lut_64: np.ndarray, target_size: int = 33) -> np.ndarray:
    """Resample 64³ LUT to target size using trilinear interpolation."""
    from scipy.ndimage import map_coordinates
    # ... implementation using same kernel as fitting
```

**Alternative:** Support "strip identity" patterns that can represent arbitrary N³.

### 5.4 Implementation

```python
def generate_hald_identity(level=8):
    """Generate Hald CLUT identity image."""
    size = level ** 3
    img = np.zeros((size, size, 3), dtype=np.float32)
    cube_size = level * level
    
    for y in range(size):
        for x in range(size):
            r = (x % cube_size) / (cube_size - 1)
            g = ((x // cube_size) + (y % level) * level) / (cube_size - 1)
            b = (y // level) / (cube_size - 1)
            img[y, x] = [r, g, b]
    
    return img

def reconstruct_from_hald(processed_img, level=8):
    """Directly reconstruct LUT from processed Hald image."""
    size = level ** 3
    cube_size = level * level
    lut = np.zeros((cube_size, cube_size, cube_size, 3))
    
    for y in range(size):
        for x in range(size):
            r_idx = x % cube_size
            g_idx = (x // cube_size) + (y % level) * level
            b_idx = y // level
            lut[r_idx, g_idx, b_idx] = processed_img[y, x]
    
    return lut
```

### 5.5 User Guidance

- **16-bit TIFF** workflow to preserve precision
- **No JPEG** for Hald images (compression destroys accuracy)
- Spatial effects (vignettes, blurs) cannot be captured
- Resolution mismatch warning if processed image size differs from identity

---

## 6. Color Space Considerations

### 6.1 Solving Space

**Solve in RGB (the LUT's operating space), validate in perceptual space.**

Rationale:
- The LUT operates in RGB—solving there avoids gamut boundary issues
- Perceptual spaces for metrics, not the model
- Avoids artifacts from non-linear transforms at gamut edges

### 6.2 Perceptual Metrics

Use Oklab or CIELAB for:
- ΔE computation in validation
- User-facing quality reports
- Contradiction detection (with proper normalization)

### 6.3 Input/Output Color Space Declaration

LUT must declare its operating domain:

```yaml
input_space: log_c4
output_space: log_c4  # Log-to-log for ALF4
# or
input_space: log_c3
output_space: rec709  # For ALF2
```

Use OCIO for color space conversions where possible.

---

## 7. Technology Stack

### 7.1 Core Dependencies

| Component | Library | Rationale |
|-----------|---------|-----------|
| Image I/O | OpenImageIO | Proper 10/16-bit, EXR, DPX handling |
| Sparse Linear Algebra | scipy.sparse | lsmr solver, COO/CSR matrices |
| Array Operations | NumPy | Vectorized processing |
| Acceleration | Numba | JIT for binning loops |
| Color Conversions | colour-science | Oklab, LogC transforms |
| Color Management | OpenColorIO | OCIO configs, LUT utilities |
| CLI | Typer | Modern, less boilerplate than Click |

### 7.2 Optional Dependencies

| Component | Library | Use Case |
|-----------|---------|----------|
| GUI | Qt/PySide6 | Optional desktop interface |
| GPU Acceleration | CuPy | Large image binning |
| Isotonic Regression | scikit-learn | Shaper monotonicity enforcement |
| CLF Export | ampas-clf | ACES CLF format |

### 7.3 Lightweight Alternative

If OIIO is too heavy to bundle:
- **imageio v3** wraps FreeImage/GDI+, lighter weight
- Less robust for DIT workflows but acceptable for simpler use cases

---

## 8. User-Facing Parameters

### 8.1 Essential Controls

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lut_size` | int | 33 | Output resolution (17, 33, 65) |
| `smoothness` | float | 0.1 | λₛ regularization strength |
| `format` | enum | cube | Output format |
| `interpolation_kernel` | enum | tetrahedral | Fitting/validation kernel |

### 8.2 Advanced Controls

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_transfer_function` | enum | auto | linear/log_c3/log_c4/slog3/vlog |
| `bin_resolution` | int | 64 | Internal sampling grid |
| `min_samples_per_bin` | int | 3 | Discard threshold |
| `prior_strength` | float | 0.01 | λᵣ for identity prior |
| `robust_loss` | enum | huber | Loss function (l2/huber/none) |
| `irls_iterations` | int | 3 | Outer loop iterations |
| `alignment` | enum | none | none/auto |

### 8.3 Output Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_shaper` | bool | auto | Generate 1D shaper |
| `coverage_report` | bool | true | Output visualization |
| `metrics_json` | bool | true | Quality metrics file |

---

## 9. Validation & Testing Strategy

### 9.1 Unit Tests

**Identity Test:** Source → Source should produce identity LUT.

**Gain Test:** Source → Source×1.5 should produce linear scaling.

**Offset Test:** Source → Source+0.1 should produce constant offset.

### 9.2 Sparse Coverage Tests (New)

**Critical from review:** Must test under-constrained cases.

**Sparse Region Test:**
1. Sample only midtones + skin tones (small RGB region)
2. Apply known transform
3. Fit LUT
4. Verify: in-region error low, out-of-region near identity, health metrics bounded

**Kernel Mismatch Test:**
1. Fit assuming tetrahedral
2. Apply with trilinear
3. Measure error delta
4. Use to justify interpolation parameter in UI

### 9.3 Integration Tests

**Round-Trip:** Generate LUT, apply to source, compare to target (same kernel).

**Format Compliance:**
- Load .cube in DaVinci Resolve
- Convert .cube to .aml via ARRI Reference Tool, load in camera simulator
- Verify no parsing errors

### 9.4 Regression Tests

Test corpus: simple corrections, typical film looks, extreme grades, challenging inputs (low coverage, high noise, vignettes).

---

## 10. Performance Targets

| Operation | Target (33³) | Target (65³) |
|-----------|--------------|--------------|
| Preprocessing | < 1s | < 2s |
| Sampling/Binning | < 2s | < 2s |
| Matrix Construction | < 1s | < 5s |
| Solve (incl. IRLS) | < 5s | < 45s |
| **Total** | **< 10s** | **< 60s** |

**Critical:** Binning must be Numba-accelerated or vectorized NumPy to hit <2s for 8M pixels.

---

## 11. Known Limitations

1. **Local adjustments cannot be captured.** Power windows, masks, vignettes → averaged or warned.
2. **Coverage determines quality.** Single shot → reliable only for similar content.
3. **Extrapolation is a guess.** Unseen colors have no ground truth.
4. **Interpolation kernel matters.** LUT optimized for tetrahedral may show artifacts with trilinear.
5. **This is not film emulation.** No halation, grain, spatial effects.
6. **Identity plate mode for precision.** When accuracy is critical, use Hald workflow.

---

## 12. Implementation Roadmap (Revised)

### Phase 1: Core + Identity Mode (4-6 weeks)
- [ ] Image I/O with OIIO + sanitization
- [ ] **Identity plate mode (Hald CLUT)** ← Moved up
- [ ] Basic binning (Numba-accelerated)
- [ ] Sparse matrix construction (COO→CSR)
- [ ] lsmr solver integration
- [ ] .cube export
- [ ] CLI interface
- [ ] Identity/gain/offset/sparse tests

### Phase 2: Regression Quality (4-6 weeks)
- [ ] IRLS outer loop for Huber loss
- [ ] Tetrahedral interpolation weights
- [ ] Outlier rejection and contradiction detection
- [ ] Spatial inconsistency (vignette) detection
- [ ] Iterative refinement loop
- [ ] Coverage and quality metrics
- [ ] Multi-image input support

### Phase 3: Format & Integration (4-6 weeks)
- [ ] Shaper generation with monotonicity enforcement
- [ ] ARRI Reference Tool integration (look-builder)
- [ ] .clf export via AMPAS reference
- [ ] OCIO integration
- [ ] Optional GUI

### Phase 4: Polish (2-4 weeks)
- [ ] Performance profiling and optimization
- [ ] Comprehensive test suite
- [ ] Documentation and tutorials
- [ ] Packaging and distribution

---

## 13. References

### Academic
- Garcia, E. & Gupta, M. (2009). "Building Accurate and Smooth ICC Profiles by Lattice Regression."
- Garcia, E. & Gupta, M. (2010). "Optimized Construction of ICC Profiles by Lattice Regression."
- Lin, H.T. et al. (2012). "Nonuniform Lattice Regression for Modeling the Camera Imaging Pipeline."
- Holland, P.W. & Welsch, R.E. (1977). "Robust Regression Using Iteratively Reweighted Least-Squares."
- Vandenberg, J. & Andriani, S. (2018). "A Survey on 3D-LUT Performance in 10-bit and 12-bit HDR BT.2100 PQ."

### Standards & Specifications
- Adobe Cube LUT Specification 1.0
- AMPAS Common LUT Format (CLF) Specification S-2014-006
- ARRI Look File 2/4 Workflow Guidelines
- ARRI "How to Create an ARRI Look File 4" (2025)

### Tools & Libraries
- OpenColorIO: https://opencolorio.readthedocs.io/
- colour-science: https://colour.readthedocs.io/
- AMPAS CLF Reference: https://github.com/ampas/CLF
- ARRI Reference Tool: https://www.arri.com/en/learn-help/learn-help-camera-system/tools/arri-reference-tool
- LUTc: https://github.com/jedypod/lutc
