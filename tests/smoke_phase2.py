from __future__ import annotations

from pathlib import Path
import sys
import unittest

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.main import app

client = TestClient(app)


class Phase2SmokeTests(unittest.TestCase):
    def test_home_exposes_demo_actions(self) -> None:
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/demo/image", response.text)
        self.assertIn("/jobs/demo/video", response.text)

    def test_demo_image_flow(self) -> None:
        response = client.post("/demo/image")

        self.assertEqual(response.status_code, 200)
        self.assertIn("ANALYSIS COMPLETE", response.text)
        self.assertIn("Download Annotated Image", response.text)
        self.assertIn("/static/results/", response.text)

    def test_demo_video_flow(self) -> None:
        response = client.post("/demo/video")

        self.assertEqual(response.status_code, 200)
        self.assertIn("POST-RACE DEBRIEF", response.text)
        self.assertIn("Damage Types", response.text)
        self.assertIn("Download Video", response.text)
        self.assertIn("demo_tracker2.mp4", response.text)
        self.assertTrue((BASE_DIR / "outputs" / "videos" / "demo_tracker2.mp4").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
