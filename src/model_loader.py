"""
Central, thread-safe, lazy model cache.

Every model (car detector, damage detector, FastAI team classifier) is
imported and constructed only the first time it is actually requested, not
at module-import time. This means starting the API, or hitting an
image-only endpoint, never pays the cost of loading the tracker/team
classifier stack unless that request path actually needs it.
"""
from __future__ import annotations

import importlib
import pathlib
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from app.config import (
    CAR_MODEL_PATH,
    DAMAGE_MODEL_PATH,
    TEAM_MODEL_PATH,
    TRACKER_CONFIG_PATH,
    verify_model_integrity,
)

_PLUM_SUBMODULES = (
    "alias",
    "dispatcher",
    "function",
    "method",
    "overload",
    "parametric",
    "promotion",
    "resolver",
    "signature",
    "type",
    "util",
)

_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
_status: dict[str, dict[str, Any]] = {
    "car_detector": {"path": str(CAR_MODEL_PATH), "loaded": False, "load_seconds": None},
    "damage_detector": {"path": str(DAMAGE_MODEL_PATH), "loaded": False, "load_seconds": None},
    "team_classifier": {"path": str(TEAM_MODEL_PATH), "loaded": False, "load_seconds": None},
}
_plum_patched = False


def _require_file(path: Path, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_once(name: str, builder: Callable[[], Any]) -> Any:
    """Return the cached instance for `name`, building it on first call only."""
    instance = _cache.get(name)
    if instance is not None:
        return instance

    with _cache_lock:
        instance = _cache.get(name)
        if instance is not None:
            return instance

        print(f"[MODELS] Loading '{name}'...")
        start = time.perf_counter()
        instance = builder()
        elapsed = time.perf_counter() - start

        _cache[name] = instance
        _status[name]["loaded"] = True
        _status[name]["load_seconds"] = round(elapsed, 3)
        print(f"[MODELS] Loaded '{name}' in {elapsed:.2f}s")
        return instance


def get_car_model():
    """YOLO car detector, used only by the full video pipeline."""

    def _build():
        from ultralytics import YOLO

        _require_file(CAR_MODEL_PATH, "Car detector model")
        return YOLO(str(CAR_MODEL_PATH))

    return _load_once("car_detector", _build)


def get_damage_model():
    """YOLO damage detector, shared by the image endpoint and the video pipeline."""

    def _build():
        from ultralytics import YOLO

        _require_file(DAMAGE_MODEL_PATH, "Damage detector model")
        return YOLO(str(DAMAGE_MODEL_PATH))

    return _load_once("damage_detector", _build)


def _patch_plum_modules() -> None:
    """
    FastAI's pickled learner references plum's private submodules under
    their public names. Patch sys.modules once, only right before the
    pickle is actually loaded (not at import time for every process).
    """
    global _plum_patched
    if _plum_patched:
        return
    for name in _PLUM_SUBMODULES:
        try:
            sys.modules.setdefault(
                f"plum.{name}", importlib.import_module(f"plum._{name}")
            )
        except ImportError:
            pass
    _plum_patched = True


def get_team_model():
    """FastAI team classifier. Only needed by the full video pipeline."""

    def _build():
        if sys.version_info < (3, 11):
            raise RuntimeError(
                "Python 3.11 or newer is required to load the bundled "
                "FastAI team classifier pickle."
            )
        _require_file(TEAM_MODEL_PATH, "Team classifier model")

        # team_model_path is a pickle (FastAI export); verify its hash before
        # deserializing since load_learner() executes arbitrary code embedded
        # in a tampered pickle.
        verify_model_integrity(TEAM_MODEL_PATH)

        if sys.platform.startswith("win"):
            pathlib.PosixPath = pathlib.WindowsPath
        _patch_plum_modules()

        from fastai.vision.all import load_learner

        return load_learner(TEAM_MODEL_PATH)

    return _load_once("team_classifier", _build)


def get_tracker_config_path() -> Path:
    _require_file(TRACKER_CONFIG_PATH, "ByteTrack tracker config")
    return TRACKER_CONFIG_PATH


def get_model_status() -> dict[str, dict[str, Any]]:
    """Snapshot of which models are loaded, and how long each took, for /api/info."""
    with _cache_lock:
        return {name: dict(info) for name, info in _status.items()}


def get_device_info() -> dict[str, Any]:
    """CPU/GPU info for /api/info. Torch is already a dependency of ultralytics."""
    import torch

    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "cuda_device_name": None,
        "cuda_device_count": 0,
    }
    if cuda_available:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info
