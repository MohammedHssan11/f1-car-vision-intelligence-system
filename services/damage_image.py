import sys
from pathlib import Path

# =============================
# ADD src TO PYTHON PATH
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# =============================
# IMPORT REAL LOGIC FILE
# =============================
from detection import detect_damage_image  # 👈 detection.py

__all__ = ["detect_damage_image"]
