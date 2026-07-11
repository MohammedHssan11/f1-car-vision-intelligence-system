from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.jobs import video_jobs
from app.main import app

client = TestClient(app)


def _wait_for_job(job_id: str, timeout_seconds: int = 120) -> dict:
    deadline = time.time() + timeout_seconds
    last_status = None
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        last_status = job["status"]
        if last_status in {"completed", "failed"}:
            return job
        time.sleep(1)
    raise AssertionError(f"Timed out waiting for job {job_id}; last status={last_status}")


class Phase3SmokeTests(unittest.TestCase):
    def test_background_demo_video_job_flow(self) -> None:
        response = client.post("/jobs/demo/video", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        job_url = response.headers["location"]
        job_id = job_url.rsplit("/", 1)[-1]

        status_response = client.get(f"/api/jobs/{job_id}")
        self.assertEqual(status_response.status_code, 200)
        self.assertIn(status_response.json()["status"], {"queued", "running", "completed"})

        job = _wait_for_job(job_id)

        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"], 100)
        self.assertIn("/outputs/videos/", job["output_video_url"])
        self.assertTrue((BASE_DIR / "outputs" / "videos" / job["output_name"]).is_file())

        result_response = client.get(job["result_url"])
        self.assertEqual(result_response.status_code, 200)
        self.assertIn("POST-RACE DEBRIEF", result_response.text)
        self.assertIn("Download Video", result_response.text)

    def test_bad_path_job_submission_is_friendly(self) -> None:
        response = client.post(
            "/jobs/pipeline/run",
            data={"video_path": "tracker/missing.mp4"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("Video not found", response.text)

    def test_failed_job_state_is_visible(self) -> None:
        job = video_jobs.submit(
            video_path=BASE_DIR / "tracker" / "missing.mp4",
            source_label="tracker/missing.mp4",
            output_stem="missing_video",
        )
        completed = _wait_for_job(job["job_id"], timeout_seconds=30)

        self.assertEqual(completed["status"], "failed")
        self.assertIn("failed", completed["message"].lower())
        self.assertTrue(completed["error"])

        result_response = client.get(completed["result_url"])
        self.assertEqual(result_response.status_code, 500)
        self.assertIn("Pipeline Job", result_response.text)
        self.assertIn(completed["error"], result_response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
