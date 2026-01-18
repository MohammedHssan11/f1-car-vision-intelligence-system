import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from pipeline import run_full_pipeline  # 👈 pipeline.py

__all__ = ["run_full_pipeline"]
