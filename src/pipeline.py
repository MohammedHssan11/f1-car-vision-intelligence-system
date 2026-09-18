import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import cv2

# =============================
# PATH SETUP
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# =============================
# IMPORTS
# =============================
# Models are pulled from the shared lazy cache (src/model_loader.py) instead
# of being constructed at import time. This keeps `import src.pipeline` cheap
# and means the heavy car/team/damage stack is only built the first time a
# video is actually processed — image-only requests never pay for it.
from app.config import OUTPUT_VIDEO_DIR
from src.detection import validate_damage_class
from src.model_loader import (
    get_car_model,
    get_damage_model,
    get_device_info,
    get_team_model,
    get_tracker_config_path,
)
from tracking_memory.tracker_core import TrackIdentityResolver, update_cars
from tracking_memory.damage_assigner import assign_damage
from tracking_memory.collision_detector import detect_collisions
from tracking_memory.overtake_detector import detect_overtakes


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# =============================
# CONFIG (GLOBAL, env-overridable)
# =============================
# Tuned for a responsive local demo. Damage detection is the expensive
# per-frame step (one YOLO forward pass per tracked car), so it runs on a
# stride rather than every frame; MAX_FRAMES caps worst-case runtime on a
# long upload.
CONF_TH = _env_float("PIPELINE_CONF_TH", 0.4)
DAMAGE_CONF_TH = _env_float("PIPELINE_DAMAGE_CONF_TH", 0.4)
TEAM_CONF_TH = _env_float("PIPELINE_TEAM_CONF_TH", 0.6)
DAMAGE_EVERY_N_FRAMES = max(1, _env_int("DAMAGE_EVERY_N_FRAMES", 10))
MAX_FRAMES = _env_int("MAX_FRAMES", 3000)  # safety limit
TRACK_REASSOCIATION_MAX_GAP = max(1, _env_int("TRACK_REASSOCIATION_MAX_GAP", 30))
TRACK_REASSOCIATION_MAX_SCORE = max(
    0.1, _env_float("TRACK_REASSOCIATION_MAX_SCORE", 1.4)
)
TRACK_ACTIVE_MAX_AGE = max(0, _env_int("TRACK_ACTIVE_MAX_AGE", 5))
COLLISION_DECEL_THRESHOLD = _env_float("COLLISION_DECEL_THRESHOLD", -1500.0)
COLLISION_MIN_PREIMPACT_SPEED = max(
    0.0, _env_float("COLLISION_MIN_PREIMPACT_SPEED", 250.0)
)
COLLISION_DAMAGE_LOOKBACK_FRAMES = max(
    DAMAGE_EVERY_N_FRAMES * 2,
    _env_int("COLLISION_DAMAGE_LOOKBACK_FRAMES", DAMAGE_EVERY_N_FRAMES * 3),
)
COLLISION_MIN_DAMAGE_OBSERVATIONS = max(
    1, _env_int("COLLISION_MIN_DAMAGE_OBSERVATIONS", 2)
)
INTERMEDIATE_VIDEO_CODEC = os.environ.get("PIPELINE_VIDEO_CODEC", "mp4v")
H264_CRF = min(35, max(18, _env_int("PIPELINE_H264_CRF", 23)))
SHOW_SPEED_OVERLAY = os.environ.get("PIPELINE_SHOW_SPEED", "false").lower() in {
    "1", "true", "yes", "on",
}

TEAM_CODES = {
    "AlphaTauri F1 car": "AT",
    "Ferrari F1 car": "FER",
    "McLaren F1 car": "MCL",
    "Mercedes F1 car": "MER",
    "Racing Point F1 car": "RP",
    "Red Bull Racing F1 car": "RBR",
    "Renault F1 car": "REN",
    "Williams F1 car": "WIL",
}


def _team_code(team_name: str | None) -> str:
    """Return a compact overlay tag instead of a livery-model class name."""
    if not team_name or team_name == "UNKNOWN":
        return ""
    return TEAM_CODES.get(team_name, team_name.replace(" F1 car", "")[:3].upper())


def _rectangles_overlap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _find_label_rect(
    frame_width: int,
    frame_height: int,
    bbox: list[int],
    label_width: int,
    label_height: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Find an uncluttered compact-label position near one detection box."""
    x1, y1, x2, y2 = bbox
    padding = 4
    candidates = (
        (x1, y1 - label_height - padding),
        (x2 - label_width, y1 - label_height - padding),
        (x1, y2 + padding),
        (x2 - label_width, y2 + padding),
        (x2 + padding, y1),
        (x1 - label_width - padding, y1),
    )
    checked: set[tuple[int, int, int, int]] = set()
    for candidate_x, candidate_y in candidates:
        x = max(0, min(candidate_x, frame_width - label_width))
        y = max(0, min(candidate_y, frame_height - label_height))
        candidate = (x, y, x + label_width, y + label_height)
        if candidate in checked:
            continue
        checked.add(candidate)
        if not any(_rectangles_overlap(candidate, other) for other in occupied):
            return candidate
    return None


def _draw_car_overlays(frame, cars, frame_idx: int) -> None:
    """Draw compact non-overlapping car tags and only essential annotations."""
    frame_height, frame_width = frame.shape[:2]
    occupied_labels: list[tuple[int, int, int, int]] = []
    visible_cars = sorted(
        (
            car
            for car in cars.values()
            if car.last_bbox is not None
            and frame_idx - car.last_seen <= TRACK_ACTIVE_MAX_AGE
        ),
        key=lambda car: (car.last_bbox[1], car.last_bbox[0], car.id),
    )

    for car in visible_cars:
        x1, y1, x2, y2 = car.last_bbox
        is_new_collision = car.last_collision_frame == frame_idx
        is_recent_overtake = (
            hasattr(car, "last_overtake_frame")
            and frame_idx - car.last_overtake_frame < 10
        )
        box_color = (0, 70, 255) if is_new_collision else (255, 220, 0) if is_recent_overtake else (0, 220, 90)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        label = f"#{car.id}"
        team_code = _team_code(car.team)
        if team_code:
            label += f" | {team_code}"
        if car.damage.types:
            label += f" | {next(iter(car.damage.types)).upper()}"
        if is_new_collision:
            label += " | IMPACT"
        elif is_recent_overtake:
            label += " | PASS"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.42, 1)
        label_rect = _find_label_rect(
            frame_width,
            frame_height,
            car.last_bbox,
            text_width + 10,
            text_height + baseline + 8,
            occupied_labels,
        )
        if label_rect is not None:
            left, top, right, bottom = label_rect
            # A solid, dark tag remains legible over bright track paint while
            # the thin colored top line preserves event/detection meaning.
            cv2.rectangle(frame, (left, top), (right, bottom), (12, 18, 24), -1)
            cv2.rectangle(frame, (left, top), (right, top + 2), box_color, -1)
            cv2.putText(
                frame,
                label,
                (left + 5, bottom - baseline - 4),
                font,
                0.42,
                (245, 248, 250),
                1,
                cv2.LINE_AA,
            )
            occupied_labels.append(label_rect)

        if SHOW_SPEED_OVERLAY:
            speed_label = f"{int(car.smoothed_speed)} px/s"
            cv2.putText(
                frame,
                speed_label,
                (x1, min(frame_height - 6, y2 + 16)),
                font,
                0.38,
                (255, 220, 0),
                1,
                cv2.LINE_AA,
            )


def _browser_transcode_command(
    ffmpeg_path: str,
    intermediate_path: Path,
    source_video_path: Path,
    encoded_path: Path,
) -> list[str]:
    """Build a widely supported H.264/AAC MP4 command for HTML5 video."""
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(intermediate_path),
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(H264_CRF),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(encoded_path),
    ]


def _publish_browser_video(
    intermediate_path: Path, source_video_path: Path, output_path: Path
) -> str:
    """Transcode the OpenCV intermediate into a browser-compatible MP4."""
    requested_binary = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    ffmpeg_path = shutil.which(requested_binary)
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to publish browser-compatible H.264 video. "
            "Install ffmpeg or set FFMPEG_BINARY to its executable path."
        )

    encoded_path = output_path.with_name(f".{output_path.stem}.browser.mp4")
    encoded_path.unlink(missing_ok=True)
    command = _browser_transcode_command(
        ffmpeg_path, intermediate_path, source_video_path, encoded_path
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not encoded_path.is_file() or encoded_path.stat().st_size == 0:
        encoded_path.unlink(missing_ok=True)
        details = result.stderr.strip() or "ffmpeg produced no output"
        raise RuntimeError(f"Browser-video transcoding failed: {details}")

    os.replace(encoded_path, output_path)
    intermediate_path.unlink(missing_ok=True)
    return "h264/aac"

# =============================
# TEAM PREDICTION
# =============================
def predict_team(crop):
    try:
        from fastai.vision.all import PILImage

        team_model = get_team_model()
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = PILImage.create(crop)

        pred_class, pred_idx, probs = team_model.predict(img)
        conf = probs[pred_idx].item()

        if conf < TEAM_CONF_TH:
            return "UNKNOWN"

        return str(pred_class)

    except Exception:
        return "UNKNOWN"

# =============================
# MAIN PIPELINE
# =============================
def run_full_pipeline(video_path, output_name="result.mp4", progress_callback=None):
    """
    Run full F1 analysis pipeline

    Args:
        video_path (str | Path): input video
        output_name (str): output video file name
        progress_callback (callable | None): optional callback receiving
            (frames_processed, total_frames) during processing

    Returns:
        dict: summary
    """

    # =============================
    # PER-RUN TRACK STATE
    # =============================
    # Local to this call, not a module global, so concurrent pipeline runs
    # (blocking routes can be served in parallel) never share or corrupt each
    # other's tracking state.
    cars = {}
    identity_resolver = TrackIdentityResolver(
        max_gap_frames=TRACK_REASSOCIATION_MAX_GAP,
        max_match_score=TRACK_REASSOCIATION_MAX_SCORE,
    )

    # =============================
    # LOAD MODELS (lazy, cached)
    # =============================
    # First video request in a process pays the load cost here; subsequent
    # requests reuse the cached instances. Loading is timed so the log makes
    # startup-vs-processing cost obvious.
    load_start = time.perf_counter()
    car_model = get_car_model()
    damage_model = get_damage_model()
    get_team_model()  # warm the team classifier before the frame loop
    tracker_config_path = get_tracker_config_path()
    load_seconds = time.perf_counter() - load_start
    # YOLO reports .device as cpu until its first inference, so report the
    # actual compute device torch will use instead of the pre-warm value.
    compute_device = get_device_info()["device"]
    print(
        f"[TIMING] models ready in {load_seconds:.2f}s "
        f"(device={compute_device})"
    )

    # =============================
    # OUTPUT PATH (IMPORTANT)
    # =============================
    output_dir = OUTPUT_VIDEO_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Pipeline output_name must use the .mp4 extension")
    intermediate_path = output_path.with_name(
        f".{output_path.stem}.intermediate.mp4"
    )
    intermediate_path.unlink(missing_ok=True)
    print(f"[PIPELINE] Saving output to: {output_path}")

    process_start = time.perf_counter()

    # =============================
    # VIDEO IO
    # =============================
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = min(raw_total_frames, MAX_FRAMES) if raw_total_frames > 0 else None
    if progress_callback:
        progress_callback(0, total_frames)

    # OpenCV writes an intermediate because its bundled MPEG-4 encoder is not
    # consistently supported by Chromium. ffmpeg then publishes H.264/AAC
    # with moov metadata at the front of the file, which starts reliably in
    # the browser and preserves any source audio.
    if len(INTERMEDIATE_VIDEO_CODEC) != 4:
        raise ValueError("PIPELINE_VIDEO_CODEC must be a four-character OpenCV codec")
    fourcc = cv2.VideoWriter_fourcc(*INTERMEDIATE_VIDEO_CODEC)
    out = cv2.VideoWriter(str(intermediate_path), fourcc, fps, (w, h))

    if not out.isOpened():
        raise RuntimeError("Failed to open output video writer")

    # =============================
    # METRICS
    # =============================
    frame_idx = 0
    total_collisions = 0
    total_overtakes = 0

    # =============================
    # MAIN LOOP
    # =============================
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx > MAX_FRAMES:
            break

        # =============================
        # CAR DETECTION + TRACKING
        # =============================
        results = car_model.track(
            frame,
            persist=True,
            tracker=str(tracker_config_path),
            conf=CONF_TH,
            verbose=False
        )

        detections = []

        for r in results:
            if r.boxes.id is None:
                continue

            for box, tid, score in zip(r.boxes.xyxy, r.boxes.id, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(tid)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "confidence": float(score),
                })

        # =============================
        # UPDATE TRACK MEMORY
        # =============================
        resolved_detections = update_cars(
            cars,
            detections,
            frame_idx,
            fps,
            identity_resolver=identity_resolver,
        )

        # Classify after IDs are resolved so a ByteTrack ID switch does not
        # trigger a fresh, potentially conflicting team label.
        for detection in resolved_detections:
            car = cars[detection["id"]]
            if car.team is not None:
                continue
            x1, y1, x2, y2 = detection["bbox"]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                car.set_team(predict_team(crop))

        # =============================
        # DAMAGE DETECTION (batched)
        # =============================
        # All eligible car crops for this stride frame are collected and run
        # through the damage model in a single batched forward pass, instead
        # of one YOLO call per car. On GPU this collapses N per-car launches
        # into one, which is where most of the per-frame time was going.
        damage_detections = []

        if frame_idx % DAMAGE_EVERY_N_FRAMES == 0:
            crops = []
            crop_offsets = []
            for car in cars.values():
                if car.last_bbox is None:
                    continue
                if frame_idx - car.last_seen > TRACK_ACTIVE_MAX_AGE:
                    continue

                x1, y1, x2, y2 = car.last_bbox
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crops.append(crop)
                crop_offsets.append((x1, y1))

            if crops:
                dmg_results = damage_model(
                    crops,
                    conf=DAMAGE_CONF_TH,
                    verbose=False
                )

                for (x1, y1), crop, r in zip(crop_offsets, crops, dmg_results):
                    if r.boxes is None:
                        continue

                    crop_h, crop_w = crop.shape[:2]
                    for box, cls, score in zip(
                        r.boxes.xyxy, r.boxes.cls, r.boxes.conf
                    ):
                        dx1, dy1, dx2, dy2 = map(int, box)
                        raw_class = damage_model.names[int(cls)]

                        # Same geometry validation the image endpoint applies
                        # (src/detection.py), run against the car crop's own
                        # dimensions so the image and video paths agree on what
                        # counts as damage instead of the video path trusting
                        # every raw YOLO box.
                        damage_type = validate_damage_class(
                            raw_class, float(score),
                            dx1, dy1, dx2, dy2,
                            crop_h, crop_w,
                        )
                        if damage_type is None:
                            continue

                        damage_detections.append({
                            "bbox": [x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2],
                            "type": damage_type,
                        })

            assign_damage(cars, damage_detections, frame_idx)

        # =============================
        # COLLISION + OVERTAKE
        # =============================
        collision_events = detect_collisions(
            cars,
            frame_idx,
            decel_th=COLLISION_DECEL_THRESHOLD,
            min_preimpact_speed=COLLISION_MIN_PREIMPACT_SPEED,
            damage_lookback_frames=COLLISION_DAMAGE_LOOKBACK_FRAMES,
            min_damage_observations=COLLISION_MIN_DAMAGE_OBSERVATIONS,
        )
        overtake_events = detect_overtakes(cars, frame_idx)

        total_collisions += len(collision_events)
        total_overtakes += len(overtake_events)

        # =============================
        # DRAW OUTPUT
        # =============================
        _draw_car_overlays(frame, cars, frame_idx)

        out.write(frame)
        if progress_callback:
            progress_callback(frame_idx, total_frames)

    # =============================
    # CLEANUP
    # =============================
    cap.release()
    out.release()
    browser_codec = _publish_browser_video(
        intermediate_path, Path(video_path), output_path
    )

    process_seconds = time.perf_counter() - process_start
    fps_processed = frame_idx / process_seconds if process_seconds > 0 else 0.0
    output_size_mb = (
        output_path.stat().st_size / (1024 * 1024) if output_path.is_file() else 0.0
    )
    print("Video processing finished")
    print(
        f"[TIMING] processed {frame_idx} frames in {process_seconds:.2f}s "
        f"({fps_processed:.1f} fps) | output {output_size_mb:.1f} MB "
        f"codec={browser_codec}"
    )

    if progress_callback:
        progress_callback(frame_idx, total_frames or frame_idx)

    damage_type_counts = {}
    damage_by_car = []
    team_counts = {}

    for car in cars.values():
        team_name = car.team or "UNKNOWN"
        team_counts[team_name] = team_counts.get(team_name, 0) + 1

        car_damage = dict(sorted(car.damage.types.items()))
        if car_damage:
            damage_by_car.append({
                "id": car.id,
                "team": team_name,
                "types": car_damage,
            })

        for damage_type, count in car_damage.items():
            damage_type_counts[damage_type] = damage_type_counts.get(damage_type, 0) + count

    return {
        "cars_tracked": len(cars),
        "track_reassociations": identity_resolver.reassociation_count,
        "total_overtakes": total_overtakes,
        "total_collisions": total_collisions,
        "damage_type_counts": dict(sorted(damage_type_counts.items())),
        "damage_by_car": sorted(damage_by_car, key=lambda item: item["id"]),
        "team_counts": dict(sorted(team_counts.items())),
        "frames_processed": frame_idx,
        "source_video": str(video_path),
        "output_video": str(output_path),
        "output_video_name": output_name,
        "output_codec": browser_codec,
        "processing_seconds": round(process_seconds, 2),
        "processing_fps": round(fps_processed, 1),
        "output_size_mb": round(output_size_mb, 2),
    }
