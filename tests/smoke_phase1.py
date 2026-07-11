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


class Phase1SmokeTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_rejects_fake_image_content(self) -> None:
        response = client.post(
            "/damage/image",
            files={"file": ("fake.jpg", b"not an image", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("not a readable image", response.text)

    def test_rejects_fake_video_content(self) -> None:
        response = client.post(
            "/pipeline/video",
            files={"file": ("fake.mp4", b"not a video", "video/mp4")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("not readable", response.text)

    def test_image_upload_smoke(self) -> None:
        sample = next((BASE_DIR / "splited_dataset").rglob("*.jpg"))

        with sample.open("rb") as image_file:
            response = client.post(
                "/damage/image",
                files={"file": (sample.name, image_file, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("ANALYSIS COMPLETE", response.text)

    def test_local_video_pipeline_smoke(self) -> None:
        response = client.post(
            "/pipeline/run",
            data={"video_path": "tracker/tracker2.mp4"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("POST-RACE DEBRIEF", response.text)
        self.assertTrue((BASE_DIR / "outputs" / "videos" / "result.mp4").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
