"""Per-run track state and conservative recovery from ByteTrack ID switches."""
from __future__ import annotations

from math import hypot, log
from typing import Any

from tracking_memory.car_state import CarState
from tracking_memory.utils import iou


class TrackIdentityResolver:
    """Map short-lived ByteTrack IDs to stable, per-video car identities.

    ByteTrack is deliberately responsible for frame-to-frame association. This
    small second layer only acts when ByteTrack emits a *new* ID shortly after
    an existing identity disappeared. It is deliberately conservative: an
    ambiguous match remains a new identity, which is safer than merging two
    different race cars and contaminating their damage/event histories.
    """

    def __init__(
        self,
        *,
        max_gap_frames: int = 30,
        max_match_score: float = 1.4,
        ambiguity_margin: float = 0.25,
    ) -> None:
        if max_gap_frames < 1:
            raise ValueError("max_gap_frames must be at least 1")
        self.max_gap_frames = max_gap_frames
        self.max_match_score = max_match_score
        self.ambiguity_margin = ambiguity_margin
        self.raw_to_stable: dict[int, int] = {}
        self.reassociation_count = 0

    @staticmethod
    def _center(bbox: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _diagonal(bbox: list[int] | tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        return max(1.0, hypot(x2 - x1, y2 - y1))

    @staticmethod
    def _area(bbox: list[int] | tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        return max(1.0, (x2 - x1) * (y2 - y1))

    def _match_score(self, car: CarState, bbox: list[int], frame_id: int) -> float:
        """Lower is better; combines motion prediction, overlap, and scale."""
        assert car.last_bbox is not None
        assert car.last_position is not None
        assert car.last_seen is not None

        gap = frame_id - car.last_seen
        predicted_center = (
            car.last_position[0] + car.velocity_px_per_frame[0] * gap,
            car.last_position[1] + car.velocity_px_per_frame[1] * gap,
        )
        candidate_center = self._center(bbox)
        scale = max(self._diagonal(car.last_bbox), self._diagonal(bbox))
        prediction_error = hypot(
            candidate_center[0] - predicted_center[0],
            candidate_center[1] - predicted_center[1],
        ) / scale
        overlap_penalty = 1.0 - iou(car.last_bbox, bbox)
        scale_penalty = abs(log(self._area(bbox) / self._area(car.last_bbox)))
        return prediction_error + 0.35 * overlap_penalty + 0.20 * scale_penalty

    @staticmethod
    def _next_stable_id(cars: dict[int, CarState], requested_id: int) -> int:
        if requested_id not in cars:
            return requested_id
        return max(cars, default=0) + 1

    def resolve(
        self,
        cars: dict[int, CarState],
        detections: list[dict[str, Any]],
        frame_id: int,
    ) -> list[dict[str, Any]]:
        """Return detections with stable ``id`` and their original ``raw_id``."""
        resolved: list[dict[str, Any] | None] = [None] * len(detections)
        used_stable_ids: set[int] = set()
        unmatched_indices: list[int] = []

        # Existing ByteTrack IDs keep their established logical identity.
        for index, detection in enumerate(detections):
            raw_id = int(detection["id"])
            stable_id = self.raw_to_stable.get(raw_id)
            if stable_id is None or stable_id in used_stable_ids:
                unmatched_indices.append(index)
                continue
            item = dict(detection)
            item["raw_id"] = raw_id
            item["id"] = stable_id
            resolved[index] = item
            used_stable_ids.add(stable_id)

        # Evaluate every new raw ID against identities that vanished only
        # recently. Scores are sorted globally so two newcomers cannot claim
        # the same old car in one frame.
        proposals: list[tuple[float, int, int]] = []
        for index in unmatched_indices:
            bbox = detections[index]["bbox"]
            for stable_id, car in cars.items():
                if stable_id in used_stable_ids or car.last_seen is None:
                    continue
                gap = frame_id - car.last_seen
                if not 0 < gap <= self.max_gap_frames or car.last_bbox is None:
                    continue
                proposals.append((self._match_score(car, bbox, frame_id), index, stable_id))

        proposals.sort()
        viable_by_detection: dict[int, list[tuple[float, int]]] = {}
        for score, index, stable_id in proposals:
            if score <= self.max_match_score:
                viable_by_detection.setdefault(index, []).append((score, stable_id))

        selected_matches: dict[int, int] = {}
        intended_matches: list[tuple[float, int, int]] = []
        for index, choices in viable_by_detection.items():
            best_score, best_id = choices[0]
            second_score = choices[1][0] if len(choices) > 1 else None
            unambiguous = (
                second_score is None
                or second_score - best_score >= self.ambiguity_margin
            )
            if unambiguous:
                intended_matches.append((best_score, index, best_id))

        # Resolve competing claims globally by score. Without this step, the
        # order YOLO returns boxes could let a weak match consume the identity
        # that a much stronger candidate needs in the same frame.
        for _, index, stable_id in sorted(intended_matches):
            if stable_id not in used_stable_ids:
                selected_matches[index] = stable_id
                used_stable_ids.add(stable_id)
                self.reassociation_count += 1

        for index in unmatched_indices:
            raw_id = int(detections[index]["id"])
            selected_id: int | None = selected_matches.get(index)

            if selected_id is None:
                selected_id = self._next_stable_id(cars, raw_id)

            self.raw_to_stable[raw_id] = selected_id
            item = dict(detections[index])
            item["raw_id"] = raw_id
            item["id"] = selected_id
            resolved[index] = item
            used_stable_ids.add(selected_id)

        return [item for item in resolved if item is not None]


def update_cars(
    cars: dict[int, CarState],
    detections: list[dict[str, Any]],
    frame_id: int,
    fps: float,
    *,
    identity_resolver: TrackIdentityResolver | None = None,
) -> list[dict[str, Any]]:
    """Update caller-owned car state and return detections with stable IDs."""
    if identity_resolver is not None:
        resolved_detections = identity_resolver.resolve(cars, detections, frame_id)
    else:
        resolved_detections = [dict(detection, raw_id=detection["id"]) for detection in detections]

    for det in resolved_detections:
        track_id = int(det["id"])
        center = det["center"]
        bbox = det["bbox"]

        if track_id not in cars:
            cars[track_id] = CarState(track_id)

        car = cars[track_id]
        car.raw_track_ids.add(int(det["raw_id"]))
        car.update(center, bbox, frame_id, fps=fps)

    return resolved_detections
