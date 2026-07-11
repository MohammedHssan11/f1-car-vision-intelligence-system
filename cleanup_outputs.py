"""
Clean generated uploads and outputs without removing their parent folders.

Usage:
    py -3.11 cleanup_outputs.py
    py -3.11 cleanup_outputs.py --yes
"""
from __future__ import annotations

import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CLEAN_ROOTS = (
    BASE_DIR / "uploads" / "images",
    BASE_DIR / "uploads" / "videos",
    BASE_DIR / "app" / "static" / "results",
    BASE_DIR / "outputs" / "videos",
)
KEEP_NAMES = {".gitkeep", ".gitignore"}


def _is_under(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def iter_generated_files() -> list[Path]:
    files: list[Path] = []
    for root in CLEAN_ROOTS:
        resolved_root = root.resolve()
        if not resolved_root.exists():
            continue

        for path in resolved_root.rglob("*"):
            resolved_path = path.resolve()
            if (
                path.is_file()
                and path.name not in KEEP_NAMES
                and _is_under(resolved_path, resolved_root)
            ):
                files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean generated project media files.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag, only prints what would be removed.",
    )
    args = parser.parse_args()

    files = iter_generated_files()
    action = "Deleting" if args.yes else "Would delete"
    total_bytes = sum(path.stat().st_size for path in files)

    for path in files:
        print(f"{action}: {path.relative_to(BASE_DIR)}")
        if args.yes:
            path.unlink()

    print(f"{len(files)} files, {total_bytes} bytes")
    if not args.yes:
        print("Dry run only. Re-run with --yes to delete these files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
