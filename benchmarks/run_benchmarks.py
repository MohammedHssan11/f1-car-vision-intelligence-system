from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import analyze_damage_image
from src.model_loader import get_car_model
from src.pipeline import CONF_TH, predict_team, run_full_pipeline

BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifest.json"
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 model quality benchmarks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Run image benchmarks only. Useful for quick local iteration.",
    )
    args = parser.parse_args()

    results = run_benchmarks(args.manifest, args.results_dir, skip_video=args.skip_video)
    json_path = args.results_dir / "benchmark_results.json"
    markdown_path = BENCHMARK_DIR / "benchmark_results.md"

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(results, json_path), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def run_benchmarks(manifest_path: Path, results_dir: Path, skip_video: bool = False) -> dict:
    manifest_path = manifest_path.resolve()
    results_dir = results_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_output_dir = results_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    sample_results = []
    started_at = utc_now()
    start_time = time.perf_counter()

    for sample in manifest["samples"]:
        if skip_video and sample["type"] == "video":
            continue

        if sample["type"] == "image":
            sample_results.append(run_image_sample(sample, image_output_dir))
        elif sample["type"] == "video":
            sample_results.append(run_video_sample(sample))
        else:
            sample_results.append({
                "id": sample["id"],
                "type": sample["type"],
                "status": "failed",
                "error": f"Unsupported sample type: {sample['type']}",
                "expected": sample.get("expected", {}),
                "observed": {},
                "metrics": {},
                "qualitative_observations": [],
            })

    return {
        "run_metadata": {
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": round(time.perf_counter() - start_time, 3),
            "manifest_path": relative_to_project(manifest_path),
            "results_dir": relative_to_project(results_dir),
            "python": sys.version.split()[0],
        },
        "manifest": {
            "version": manifest.get("version"),
            "description": manifest.get("description"),
        },
        "overall_metrics": compute_overall_metrics(sample_results),
        "samples": sample_results,
        "weak_classes_and_retraining_needs": build_retraining_notes(sample_results),
    }


def run_image_sample(sample: dict, image_output_dir: Path) -> dict:
    sample_path = resolve_sample_path(sample["path"])
    expected = sample.get("expected", {})
    started = time.perf_counter()

    if not sample_path.is_file():
        return failed_sample(sample, f"Sample file not found: {sample['path']}")

    image = cv2.imread(str(sample_path))
    if image is None:
        return failed_sample(sample, f"OpenCV could not read image: {sample['path']}")

    car_observed = run_car_detection(image)
    team_observed = run_team_classification(image, expected.get("team_label"))
    damage_observed = analyze_damage_image(sample_path, image_output_dir)

    metrics = {}
    if expected.get("team_label") is not None:
        metrics["team_label_available"] = True
        metrics["team_label_match"] = team_observed["matches_expected"]
    else:
        metrics["team_label_available"] = False

    metrics["car_box_ground_truth_available"] = expected.get("car_boxes") is not None
    metrics["damage_ground_truth_available"] = expected.get("damage_labels") is not None

    return {
        "id": sample["id"],
        "type": "image",
        "status": "completed",
        "path": sample["path"],
        "evaluation_scope": sample.get("evaluation_scope", []),
        "expected": expected,
        "expected_notes": sample.get("expected_notes", ""),
        "observed": {
            "car_detection": car_observed,
            "team_classification": team_observed,
            "damage_detection": summarize_damage_for_report(damage_observed),
        },
        "metrics": metrics,
        "qualitative_observations": image_qualitative_notes(expected, damage_observed),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def run_video_sample(sample: dict) -> dict:
    sample_path = resolve_sample_path(sample["path"])
    expected = sample.get("expected", {})
    started = time.perf_counter()

    if not sample_path.is_file():
        return failed_sample(sample, f"Sample file not found: {sample['path']}")

    output_name = f"benchmark_{sample['id']}.mp4"
    progress_events = []

    def progress_callback(frames_processed: int, total_frames: int | None) -> None:
        progress_events.append({
            "frames_processed": frames_processed,
            "total_frames": total_frames,
        })

    summary = run_full_pipeline(
        video_path=sample_path,
        output_name=output_name,
        progress_callback=progress_callback,
    )

    metrics = {
        "track_ground_truth_available": expected.get("tracked_cars") is not None,
        "event_ground_truth_available": (
            expected.get("collisions") is not None or expected.get("overtakes") is not None
        ),
        "damage_ground_truth_available": expected.get("damage_labels") is not None,
        "team_ground_truth_available": expected.get("team_labels") is not None,
    }

    return {
        "id": sample["id"],
        "type": "video",
        "status": "completed",
        "path": sample["path"],
        "evaluation_scope": sample.get("evaluation_scope", []),
        "expected": expected,
        "expected_notes": sample.get("expected_notes", ""),
        "observed": {
            "pipeline_summary": summary,
            "progress_events_recorded": len(progress_events),
            "last_progress_event": progress_events[-1] if progress_events else None,
        },
        "metrics": metrics,
        "qualitative_observations": video_qualitative_notes(expected, summary),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def run_car_detection(image) -> dict:
    car_model = get_car_model()
    results = car_model(image, conf=CONF_TH, verbose=False)
    detections = []
    class_counts = {}

    for result in results:
        if result.boxes is None:
            continue
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_name = car_model.names[int(cls)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            detections.append({
                "bbox": [int(v) for v in box],
                "class_name": class_name,
                "confidence": float(conf),
            })

    return {
        "detection_count": len(detections),
        "class_counts": dict(sorted(class_counts.items())),
        "max_confidence": round(max((item["confidence"] for item in detections), default=0.0), 4),
        "detections": detections,
    }


def run_team_classification(image, expected_label: str | None) -> dict:
    predicted_label = predict_team(image)
    normalized_predicted = normalize_label(predicted_label)
    normalized_expected = normalize_label(expected_label) if expected_label else None

    return {
        "predicted_label": predicted_label,
        "expected_label": expected_label,
        "matches_expected": labels_match(normalized_expected, normalized_predicted),
        "note": "Runtime wrapper does not expose confidence; TEAM_CONF_TH may return UNKNOWN.",
    }


def summarize_damage_for_report(summary: dict) -> dict:
    return {
        "raw_detections": summary["raw_detections"],
        "valid_detections": summary["valid_detections"],
        "raw_detection_counts": summary["raw_detection_counts"],
        "valid_detection_counts": summary["valid_detection_counts"],
        "glass_missing_global": summary["glass_missing_global"],
        "output_path": relative_to_project(Path(summary["output_path"])),
    }


def compute_overall_metrics(samples: list[dict]) -> dict:
    completed = [sample for sample in samples if sample["status"] == "completed"]
    image_samples = [sample for sample in completed if sample["type"] == "image"]
    video_samples = [sample for sample in completed if sample["type"] == "video"]
    team_labeled = [
        sample for sample in image_samples
        if sample["metrics"].get("team_label_available")
    ]
    team_correct = [
        sample for sample in team_labeled
        if sample["metrics"].get("team_label_match")
    ]

    return {
        "total_samples": len(samples),
        "completed_samples": len(completed),
        "failed_samples": len(samples) - len(completed),
        "image_samples": len(image_samples),
        "video_samples": len(video_samples),
        "team_labeled_image_samples": len(team_labeled),
        "team_correct_image_samples": len(team_correct),
        "team_accuracy": (
            round(len(team_correct) / len(team_labeled), 4) if team_labeled else None
        ),
        "car_detection_accuracy": None,
        "damage_detection_accuracy": None,
        "event_detection_accuracy": None,
        "accuracy_notes": (
            "Only team image labels are available in this manifest. "
            "Car boxes, damage labels, and video events are measured qualitatively only."
        ),
    }


def build_retraining_notes(samples: list[dict]) -> list[str]:
    notes = []

    for sample in samples:
        if sample["status"] != "completed":
            notes.append(f"{sample['id']}: failed benchmark sample; inspect error before tuning models.")
            continue

        if sample["type"] == "image":
            team = sample["observed"]["team_classification"]
            if team["expected_label"] and not team["matches_expected"]:
                notes.append(
                    f"{sample['id']}: team classifier predicted {team['predicted_label']} "
                    f"for expected {team['expected_label']}."
                )

            car_count = sample["observed"]["car_detection"]["detection_count"]
            if car_count == 0:
                notes.append(f"{sample['id']}: car detector found 0 cars; inspect sample quality or detector coverage.")

        if sample["type"] == "video":
            summary = sample["observed"]["pipeline_summary"]
            if summary.get("team_counts", {}).get("UNKNOWN", 0) > 0:
                notes.append(
                    f"{sample['id']}: video pipeline produced UNKNOWN team labels; inspect crop quality or add team data."
                )
            if not summary.get("damage_type_counts"):
                notes.append(
                    f"{sample['id']}: video sample produced no assigned damage; add labeled damage video before tuning thresholds."
                )

    notes.append("Add box-level car labels before reporting car detection precision/recall.")
    notes.append("Add damage labels before reporting damage precision/recall or per-class F1.")
    notes.append("Add event labels before reporting collision/overtake accuracy.")
    return notes


def image_qualitative_notes(expected: dict, damage_summary: dict) -> list[str]:
    notes = []
    if expected.get("car_boxes") is None:
        notes.append("Car boxes are not labeled; car detection count is measured but not scored.")
    if expected.get("damage_labels") is None:
        notes.append("Damage labels are not available; damage detections are measured but not scored.")
    if damage_summary["valid_detections"] == 0:
        notes.append("No valid damage detections after post-processing.")
    return notes


def video_qualitative_notes(expected: dict, summary: dict) -> list[str]:
    notes = []
    if expected.get("tracked_cars") is None:
        notes.append("Tracked car count has no ground truth label; value is qualitative.")
    if expected.get("collisions") is None or expected.get("overtakes") is None:
        notes.append("Collision/overtake counts have no event labels; values are qualitative.")
    if expected.get("damage_labels") is None:
        notes.append("Damage counts have no frame-level labels; values are qualitative.")
    notes.append(
        f"Measured {summary.get('cars_tracked')} tracks, {summary.get('total_collisions')} collisions, "
        f"and {summary.get('total_overtakes')} overtakes."
    )
    return notes


def render_markdown(results: dict, json_path: Path) -> str:
    metrics = results["overall_metrics"]
    lines = [
        "# Phase 4 Benchmark Results",
        "",
        f"Generated: `{results['run_metadata']['completed_at']}`",
        f"Manifest: `{results['run_metadata']['manifest_path']}`",
        f"Machine-readable results: `{relative_to_project(json_path)}`",
        "",
        "## Evidence Boundary",
        "",
        metrics["accuracy_notes"],
        "",
        "Expected labels and measured observations are listed separately. `null` expected values mean no ground truth label was available.",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "| ------ | ----- |",
    ]

    for key, value in metrics.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend([
        "",
        "## Sample Results",
        "",
        "| Sample | Type | Expected | Observed | Metrics |",
        "| ------ | ---- | -------- | -------- | ------- |",
    ])

    for sample in results["samples"]:
        lines.append(
            "| `{id}` | `{type}` | `{expected}` | `{observed}` | `{metrics}` |".format(
                id=sample["id"],
                type=sample["type"],
                expected=compact_json(sample.get("expected", {})),
                observed=compact_json(sample.get("observed", {})),
                metrics=compact_json(sample.get("metrics", {})),
            )
        )

    lines.extend([
        "",
        "## Qualitative Observations",
        "",
    ])

    for sample in results["samples"]:
        lines.append(f"### `{sample['id']}`")
        for note in sample.get("qualitative_observations", []):
            lines.append(f"- {note}")
        if not sample.get("qualitative_observations"):
            lines.append("- No qualitative notes.")
        lines.append("")

    lines.extend([
        "## Weak Classes and Retraining Needs",
        "",
    ])

    for note in results["weak_classes_and_retraining_needs"]:
        lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def failed_sample(sample: dict, error: str) -> dict:
    return {
        "id": sample["id"],
        "type": sample["type"],
        "status": "failed",
        "path": sample.get("path"),
        "evaluation_scope": sample.get("evaluation_scope", []),
        "expected": sample.get("expected", {}),
        "expected_notes": sample.get("expected_notes", ""),
        "observed": {},
        "metrics": {},
        "qualitative_observations": [error],
        "error": error,
    }


def resolve_sample_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_to_project(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def normalize_label(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def labels_match(expected: str, predicted: str) -> bool:
    return bool(expected and predicted and (expected == predicted or expected in predicted))


def compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).replace("|", "\\|")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
