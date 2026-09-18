"""Temporal, damage-confirmed impact detection.

The system only has monocular broadcast video, so this module reports a
``collision`` as a *probable visual impact*, not a telemetry-grade fact. A
single-frame detector artefact or one noisy motion measurement is never enough
to create an event.
"""
from __future__ import annotations

from math import hypot
from typing import Any


def _nearby_car_ids(cars: dict[int, Any], subject: Any, frame_idx: int) -> list[int]:
    """Find currently visible cars close enough to be plausible counterparts."""
    if subject.last_bbox is None:
        return []
    sx1, sy1, sx2, sy2 = subject.last_bbox
    subject_center = ((sx1 + sx2) / 2, (sy1 + sy2) / 2)
    subject_diagonal = max(1.0, hypot(sx2 - sx1, sy2 - sy1))
    nearby: list[int] = []

    for candidate in cars.values():
        if candidate.id == subject.id or candidate.last_seen != frame_idx:
            continue
        if candidate.last_bbox is None:
            continue
        x1, y1, x2, y2 = candidate.last_bbox
        candidate_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        candidate_diagonal = max(1.0, hypot(x2 - x1, y2 - y1))
        separation = hypot(
            subject_center[0] - candidate_center[0],
            subject_center[1] - candidate_center[1],
        )
        if separation <= 1.25 * max(subject_diagonal, candidate_diagonal):
            nearby.append(candidate.id)

    return sorted(nearby)


def detect_collisions(
    cars: dict[int, Any],
    frame_idx: int,
    *,
    decel_th: float = -1500.0,
    min_preimpact_speed: float = 250.0,
    damage_lookback_frames: int = 30,
    min_damage_observations: int = 2,
    cooldown: int = 45,
) -> list[dict[str, Any]]:
    """Return new, conservatively-confirmed collision events.

    Damage detection normally runs on a stride, so the first braking signal
    rarely occurs on the exact frame on which damage is confirmed. An event is
    emitted only when a damage class has been seen on distinct inference frames
    and a strong deceleration occurred within the preceding temporal window.
    """
    if min_damage_observations < 1:
        raise ValueError("min_damage_observations must be at least 1")

    collision_events: list[dict[str, Any]] = []
    for car in cars.values():
        # Stale boxes cannot yield a meaningful current event.
        if car.last_seen != frame_idx:
            continue

        reported_damage = getattr(car, "reported_collision_damage", set())
        car.reported_collision_damage = reported_damage
        earliest_frame = frame_idx - damage_lookback_frames

        for damage_type, first_seen in car.damage.first_seen.items():
            last_seen = car.damage.last_seen.get(damage_type)
            observations = car.damage.observation_count(damage_type)
            if (
                damage_type in reported_damage
                or last_seen != frame_idx
                or observations < min_damage_observations
            ):
                continue

            # Confirmation happens on this inference frame; inspect movement
            # since the damage first appeared, including braking just before
            # it. This bridges DAMAGE_EVERY_N_FRAMES safely.
            window_start = max(earliest_frame, first_seen - damage_lookback_frames)
            acceleration_window = [
                acceleration
                for sample_frame, acceleration in car.acceleration_samples
                if window_start <= sample_frame <= frame_idx
            ]
            speed_window = [
                speed
                for sample_frame, speed in car.speed_samples
                if window_start <= sample_frame <= frame_idx
            ]
            peak_deceleration = min(acceleration_window, default=0.0)
            preimpact_speed = max(speed_window, default=0.0)

            if peak_deceleration > decel_th or preimpact_speed < min_preimpact_speed:
                continue
            if (
                car.last_collision_frame is not None
                and frame_idx - car.last_collision_frame < cooldown
            ):
                continue

            nearby_cars = _nearby_car_ids(cars, car, frame_idx)
            event = {
                "frame": frame_idx,
                "car_id": car.id,
                "damage_type": damage_type,
                "damage_first_seen": first_seen,
                "damage_observations": observations,
                "preimpact_speed": round(preimpact_speed, 1),
                "peak_deceleration": round(peak_deceleration, 1),
                "nearby_car_ids": nearby_cars,
                "confidence": "high" if nearby_cars else "medium",
            }
            collision_events.append(event)
            car.collision_frames.append(frame_idx)
            car.last_collision_frame = frame_idx
            reported_damage.add(damage_type)
            # A car can have multiple damage types, but one collision per car
            # per frame is the clearest debrief output.
            break

    return collision_events
