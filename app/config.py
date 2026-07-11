"""
Central configuration for paths, upload limits, CORS, and model integrity checks.

Security-sensitive constants live here so they are defined once and reviewed
in one place, instead of being scattered/hardcoded across app/api.py,
app/main.py, and src/pipeline.py.
"""
import hashlib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# =============================
# UPLOAD / OUTPUT DIRECTORIES
# =============================
UPLOAD_IMAGE_DIR = BASE_DIR / "uploads" / "images"
UPLOAD_VIDEO_DIR = BASE_DIR / "uploads" / "videos"
OUTPUT_VIDEO_DIR = BASE_DIR / "outputs" / "videos"
OUTPUT_IMAGE_DIR = BASE_DIR / "app" / "static" / "results"
SAMPLE_VIDEO_DIR = BASE_DIR / "tracker"

for _d in (UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR, OUTPUT_VIDEO_DIR, OUTPUT_IMAGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# =============================
# MODEL / TRACKER PATHS
# =============================
MODEL_DIR = BASE_DIR / "models"

# Car detector ships in models/ alongside the damage and team models, so all
# three weights live in one place. best_f1_detect.pt is a byte-identical copy
# of the old yolo_model_robflow/.../best.pt training output; loading it from
# models/ means a fresh clone or Docker build no longer depends on the
# gitignored training-scratch tree being present.
CAR_MODEL_PATH = MODEL_DIR / "best_f1_detect.pt"
DAMAGE_MODEL_PATH = MODEL_DIR / "best_carDD.pt"
TEAM_MODEL_PATH = MODEL_DIR / "f1_team_classifier.pkl"
TRACKER_CONFIG_PATH = BASE_DIR / "tracker" / "bytetrack.yaml"

# The local-path pipeline form may run files from these whitelisted roots.
LOCAL_VIDEO_ROOTS = (UPLOAD_VIDEO_DIR, SAMPLE_VIDEO_DIR)
DEFAULT_LOCAL_VIDEO_PATH = "tracker/tracker2.mp4"
SAMPLE_VIDEO_PATH = BASE_DIR / DEFAULT_LOCAL_VIDEO_PATH
SAMPLE_IMAGE_PATH = (
    BASE_DIR
    / "splited_dataset"
    / "val"
    / "valid_images"
    / "ferrari_f1_car_0014.jpg"
)

# =============================
# UPLOAD VALIDATION
# =============================
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024      # 20 MB
MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024     # 500 MB

# =============================
# CORS
# =============================
# Comma-separated list, override with the ALLOWED_ORIGINS env var in deployment.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if origin.strip()
]

# =============================
# MODEL INTEGRITY (anti-tamper)
# =============================
# Pinned SHA-256 of trusted model files. Verified before any pickle-based
# model (e.g. FastAI's load_learner) is deserialized, since a swapped .pkl
# file can execute arbitrary code on load.
TRUSTED_MODEL_HASHES = {
    "models/f1_team_classifier.pkl": (
        "d6594c0c1d5c7804ed65e20ce228eb07dbdf6b2829a244105d08577a53b244af"
    ),
}


class ModelIntegrityError(RuntimeError):
    """Raised when a model file's hash doesn't match the pinned trusted value."""


def verify_model_integrity(model_path: Path) -> None:
    """
    Verify a model file's SHA-256 hash against TRUSTED_MODEL_HASHES before it
    is deserialized. Required for any pickle-based model (e.g. FastAI's
    load_learner), since loading a tampered pickle can execute arbitrary code.

    Files not present in TRUSTED_MODEL_HASHES are rejected rather than
    silently allowed, so a new model must be explicitly pinned here.
    """
    model_path = Path(model_path).resolve()
    try:
        rel_key = str(model_path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        rel_key = model_path.name

    expected = TRUSTED_MODEL_HASHES.get(rel_key)
    if expected is None:
        raise ModelIntegrityError(
            f"No pinned hash for model '{rel_key}'. Refusing to load an "
            "unverified pickle file. Add its SHA-256 to TRUSTED_MODEL_HASHES "
            "in app/config.py after manually verifying the file."
        )

    digest = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()

    if actual != expected:
        raise ModelIntegrityError(
            f"Integrity check failed for model '{rel_key}': expected hash "
            f"{expected}, got {actual}. The file may have been tampered "
            "with or corrupted; refusing to load."
        )
