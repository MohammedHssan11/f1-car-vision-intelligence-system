"""Fast unit tests for model trust, tracking continuity, and collision timing."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import config
from app.api import templates
from src.pipeline import _browser_transcode_command, _find_label_rect, _team_code
from src.tracking import CarState
from starlette.requests import Request
from tracking_memory.collision_detector import detect_collisions
from tracking_memory.damage_state import DamageState
from tracking_memory.tracker_core import TrackIdentityResolver, update_cars


def _detection(raw_id: int, bbox: list[int]) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "id": raw_id,
        "bbox": bbox,
        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
    }


class ModelIntegrityTests(unittest.TestCase):
    def test_integrity_rejects_a_changed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "candidate.pt"
            artifact.write_bytes(b"trusted-model")
            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            key = artifact.name
            original = config.TRUSTED_MODEL_HASHES.get(key)
            config.TRUSTED_MODEL_HASHES[key] = expected
            try:
                config.verify_model_integrity(artifact)
                artifact.write_bytes(b"changed-model")
                with self.assertRaises(config.ModelIntegrityError):
                    config.verify_model_integrity(artifact)
            finally:
                if original is None:
                    config.TRUSTED_MODEL_HASHES.pop(key, None)
                else:
                    config.TRUSTED_MODEL_HASHES[key] = original


class TemplateRenderingTests(unittest.TestCase):
    def test_pipeline_result_is_server_rendered_not_raw_jinja(self) -> None:
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
        response = templates.TemplateResponse(
            request,
            "pipeline_result.html",
            {
                "summary": {
                    "cars_tracked": 7,
                    "track_reassociations": 1,
                    "total_collisions": 0,
                    "total_overtakes": 0,
                    "frames_processed": 60,
                    "damage_type_counts": {},
                },
                "source_label": "sample.mp4",
                "output_video_name": "result.mp4",
                "video_path": "/outputs/videos/result.mp4",
                "video_download_url": "/outputs/videos/result.mp4",
            },
        )
        page = response.body.decode("utf-8")
        self.assertNotIn("{{", page)
        self.assertNotIn("{%", page)
        self.assertIn(">7<", page)


class VisualDeliveryTests(unittest.TestCase):
    def test_browser_transcode_uses_h264_aac_and_faststart(self) -> None:
        command = _browser_transcode_command(
            "ffmpeg",
            Path("intermediate.mp4"),
            Path("source.mp4"),
            Path("published.mp4"),
        )
        self.assertIn("libx264", command)
        self.assertIn("aac", command)
        self.assertIn("+faststart", command)
        self.assertIn("1:a?", command)

    def test_compact_team_tags_and_label_layout(self) -> None:
        self.assertEqual(_team_code("Ferrari F1 car"), "FER")
        self.assertEqual(_team_code("UNKNOWN"), "")
        first_label = _find_label_rect(100, 100, [30, 30, 50, 50], 20, 10, [])
        self.assertIsNotNone(first_label)
        second_label = _find_label_rect(
            100, 100, [30, 30, 50, 50], 20, 10, [first_label]
        )
        self.assertIsNotNone(second_label)
        self.assertNotEqual(first_label, second_label)


class TrackingContinuityTests(unittest.TestCase):
    def test_short_id_switch_reuses_the_existing_car_state(self) -> None:
        cars: dict[int, CarState] = {}
        resolver = TrackIdentityResolver(max_gap_frames=5)

        update_cars(cars, [_detection(10, [0, 0, 20, 20])], 1, 30, identity_resolver=resolver)
        update_cars(cars, [_detection(10, [10, 0, 30, 20])], 2, 30, identity_resolver=resolver)
        resolved = update_cars(
            cars,
            [_detection(42, [20, 0, 40, 20])],
            3,
            30,
            identity_resolver=resolver,
        )

        self.assertEqual(resolved[0]["id"], 10)
        self.assertEqual(resolved[0]["raw_id"], 42)
        self.assertEqual(set(cars), {10})
        self.assertEqual(cars[10].raw_track_ids, {10, 42})
        self.assertEqual(resolver.reassociation_count, 1)

    def test_kinematics_accounts_for_dropped_frames(self) -> None:
        car = CarState(1)
        car.update((0, 0), [0, 0, 20, 20], frame_id=1, fps=10)
        car.update((20, 0), [10, 0, 30, 20], frame_id=3, fps=10)

        # Twenty pixels over two frames at 10 FPS is 100 px/s, not 200 px/s.
        self.assertEqual(car.speed_history[-1], 100.0)

    def test_best_same_frame_reassociation_wins_over_detection_order(self) -> None:
        cars: dict[int, CarState] = {}
        resolver = TrackIdentityResolver(max_gap_frames=5)
        update_cars(cars, [_detection(10, [0, 0, 20, 20])], 1, 30, identity_resolver=resolver)
        update_cars(cars, [_detection(10, [10, 0, 30, 20])], 2, 30, identity_resolver=resolver)

        # Both new raw tracks could claim car 10. The closer second detection
        # must win even though it appears later in the detector output.
        resolved = update_cars(
            cars,
            [
                _detection(42, [23, 0, 43, 20]),
                _detection(43, [20, 0, 40, 20]),
            ],
            3,
            30,
            identity_resolver=resolver,
        )

        by_raw_id = {item["raw_id"]: item["id"] for item in resolved}
        self.assertEqual(by_raw_id[43], 10)
        self.assertNotEqual(by_raw_id[42], 10)


class DamageAndCollisionTests(unittest.TestCase):
    def test_damage_observations_are_unique_per_frame_and_severity_is_temporal(self) -> None:
        damage = DamageState()
        self.assertTrue(damage.update("dent", 10))
        self.assertFalse(damage.update("dent", 10))
        self.assertTrue(damage.update("dent", 45))

        self.assertEqual(damage.observation_count("dent"), 2)
        self.assertEqual(damage.persistence_frames("dent"), 36)
        self.assertEqual(damage.severity("dent"), "HIGH")

    def test_collision_requires_confirmed_damage_and_recent_braking(self) -> None:
        car = CarState(1)
        # Build a high-speed then sharp-stop sequence at 30 FPS.
        car.update((0, 0), [0, 0, 20, 20], 1, 30)
        car.update((20, 0), [10, 0, 30, 20], 2, 30)
        car.update((40, 0), [30, 0, 50, 20], 3, 30)
        car.damage.update("dent", 3)
        car.update((40, 0), [30, 0, 50, 20], 4, 30)
        car.damage.update("dent", 4)

        events = detect_collisions(
            {1: car},
            4,
            decel_th=-1500,
            min_preimpact_speed=250,
            damage_lookback_frames=10,
            min_damage_observations=2,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["damage_type"], "dent")
        self.assertLess(events[0]["peak_deceleration"], -1500)
        self.assertEqual(
            detect_collisions(
                {1: car},
                4,
                decel_th=-1500,
                min_preimpact_speed=250,
                damage_lookback_frames=10,
                min_damage_observations=2,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
