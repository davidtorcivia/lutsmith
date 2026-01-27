"""Default configuration, constants, and limits for ChromaForge."""

# --- Security limits ---
MAX_IMAGE_DIMENSION = 16384  # 16K pixels per side
MAX_IMAGE_PIXELS = 100_000_000  # 100 megapixels
MAX_CUBE_SIZE = 129  # Maximum LUT grid size (129^3 ~ 2.1M nodes)
MAX_CUBE_FILE_LINES = MAX_CUBE_SIZE ** 3 + 200  # Safety limit for .cube parsing
MAX_LSMR_ITERATIONS = 5000
MAX_IRLS_ITERATIONS = 20

# --- Allowed file extensions ---
IMAGE_EXTENSIONS = frozenset({
    ".exr", ".tiff", ".tif", ".png", ".jpg", ".jpeg",
    ".dpx", ".hdr", ".bmp",
})
LUT_EXTENSIONS = frozenset({".cube"})

# --- Default pipeline parameters ---
DEFAULT_LUT_SIZE = 33
DEFAULT_SMOOTHNESS = 0.1  # lambda_s
DEFAULT_PRIOR_STRENGTH = 0.01  # lambda_r
DEFAULT_BIN_RESOLUTION = 64
DEFAULT_MIN_SAMPLES_PER_BIN = 3
DEFAULT_ROBUST_LOSS = "huber"
DEFAULT_HUBER_DELTA = 1.0
DEFAULT_IRLS_ITERATIONS = 3
DEFAULT_INTERPOLATION_KERNEL = "tetrahedral"
DEFAULT_FORMAT = "cube"

# --- Numerical safety ---
EPSILON = 1e-8  # Division-by-zero guard
CLIP_PENALTY_LOW = 0.05  # Near-black threshold
CLIP_PENALTY_HIGH = 0.98  # Near-white threshold
LUT_CLAMP_MIN = -0.5  # Post-solve LUT clamp range
LUT_CLAMP_MAX = 1.5

# --- Solver defaults ---
LSMR_ATOL = 1e-6
LSMR_BTOL = 1e-6
LSMR_DEFAULT_MAXITER = 1000
IRLS_CONVERGENCE_TOL = 1e-4

# --- Hald defaults ---
DEFAULT_HALD_LEVEL = 8  # 512x512 image, 64^3 LUT

# --- Validation ---
VALIDATION_SUBSAMPLE_FRACTION = 0.1  # Use 10% of samples for DeltaE
COVERAGE_DENSE_THRESHOLD = 2  # Grid steps
COVERAGE_SPARSE_THRESHOLD = 5

# --- Shadow smoothness boost ---
DEFAULT_SHADOW_SMOOTH_BOOST = 25.0  # Laplacian weight multiplier for deep-shadow nodes

# --- Prior strength falloff ---
PRIOR_DISTANCE_SCALE = 3.0  # Distance at which prior strength is ~63% of max
