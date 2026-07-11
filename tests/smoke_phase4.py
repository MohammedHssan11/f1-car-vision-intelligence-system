from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from benchmarks.run_benchmarks import run_benchmarks


class Phase4SmokeTests(unittest.TestCase):
    def test_benchmark_manifest_exists_and_references_existing_samples(self) -> None:
        manifest_path = BASE_DIR / "benchmarks" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(manifest["samples"]), 4)
        for sample in manifest["samples"]:
            sample_path = BASE_DIR / sample["path"]
            self.assertTrue(sample_path.is_file(), sample["path"])
            self.assertIn("expected", sample)
            self.assertIn("expected_notes", sample)

    def test_benchmark_outputs_exist_after_runner(self) -> None:
        json_path = BASE_DIR / "benchmarks" / "results" / "benchmark_results.json"
        markdown_path = BASE_DIR / "benchmarks" / "benchmark_results.md"

        self.assertTrue(json_path.is_file(), "Run py -3.11 benchmarks\\run_benchmarks.py first")
        self.assertTrue(markdown_path.is_file(), "Run py -3.11 benchmarks\\run_benchmarks.py first")

        results = json.loads(json_path.read_text(encoding="utf-8"))
        metrics = results["overall_metrics"]

        self.assertEqual(metrics["total_samples"], 4)
        self.assertIn("accuracy_notes", metrics)
        self.assertIn("samples", results)
        self.assertIn("weak_classes_and_retraining_needs", results)

        for sample in results["samples"]:
            self.assertIn("expected", sample)
            self.assertIn("observed", sample)
            self.assertIn("metrics", sample)

    def test_benchmark_runner_executes_image_only_path(self) -> None:
        manifest_path = BASE_DIR / "benchmarks" / "manifest.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_benchmarks(
                manifest_path=manifest_path,
                results_dir=Path(temp_dir),
                skip_video=True,
            )

        metrics = results["overall_metrics"]
        self.assertEqual(metrics["total_samples"], 3)
        self.assertEqual(metrics["completed_samples"], 3)
        self.assertEqual(metrics["failed_samples"], 0)
        self.assertEqual(metrics["image_samples"], 3)
        self.assertEqual(metrics["video_samples"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
