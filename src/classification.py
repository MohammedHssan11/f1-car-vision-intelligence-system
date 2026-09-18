"""Train the F1 team/livery classifier and export it to the runtime path.

This module deliberately has no workstation-specific paths. Training data is
provided through ``--dataset-dir`` (or ``TRAINING_DATASET_DIR``), while the
default model output is the same ``models/f1_team_classifier.pkl`` file that
the FastAPI service loads.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import TEAM_MODEL_PATH


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_dataset_images(dataset_dir: Path) -> list[Path]:
    """Fail safely on corrupt input instead of deleting a user's dataset."""
    from fastai.vision.all import get_image_files
    from PIL import Image, UnidentifiedImageError

    corrupt: list[Path] = []
    for image_path in get_image_files(dataset_dir):
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            corrupt.append(image_path)
    return corrupt


def train_classifier(
    dataset_dir: Path,
    output_path: Path = TEAM_MODEL_PATH,
    *,
    epochs: int = 5,
    valid_pct: float = 0.2,
    seed: int = 42,
) -> Path:
    """Train ResNet34 and atomically replace the configured runtime export."""
    from fastai.vision.all import (
        ImageDataLoaders,
        Resize,
        accuracy,
        aug_transforms,
        resnet34,
        vision_learner,
    )

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Training dataset directory not found: {dataset_dir}")
    if not 0 < valid_pct < 1:
        raise ValueError("valid_pct must be between 0 and 1")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    corrupt = _verify_dataset_images(dataset_dir)
    if corrupt:
        examples = "\n".join(f"  - {path}" for path in corrupt[:10])
        remaining = "" if len(corrupt) <= 10 else f"\n  ... and {len(corrupt) - 10} more"
        raise RuntimeError(
            "Training data contains corrupt images. Remove or replace them "
            "manually; the training command will not delete source data.\n"
            f"{examples}{remaining}"
        )

    data_loaders = ImageDataLoaders.from_folder(
        dataset_dir,
        valid_pct=valid_pct,
        seed=seed,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(),
        num_workers=0,
    )
    learner = vision_learner(data_loaders, resnet34, metrics=[accuracy])
    learner.fine_tune(epochs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f".tmp{output_path.suffix}")
    try:
        learner.export(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=os.environ.get("TRAINING_DATASET_DIR"),
        help="Folder containing one subfolder per team (or set TRAINING_DATASET_DIR).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TEAM_MODEL_PATH,
        help=f"Runtime model export path (default: {TEAM_MODEL_PATH}).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--valid-pct", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dataset_dir is None:
        raise SystemExit(
            "Provide --dataset-dir <folder> or set TRAINING_DATASET_DIR. "
            "No machine-specific training path is assumed."
        )

    exported_path = train_classifier(
        args.dataset_dir,
        args.output,
        epochs=args.epochs,
        valid_pct=args.valid_pct,
        seed=args.seed,
    )
    digest = _sha256(exported_path)
    print(f"Classifier exported to: {exported_path}")
    print(f"SHA-256: {digest}")
    print(
        "Before deploying the new model, review it and replace the "
        "models/f1_team_classifier.pkl hash in app/config.py with this value."
    )


if __name__ == "__main__":
    main()
