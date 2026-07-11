import os
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
from tracking_memory.tracker_core import update_cars
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
DAMAGE_EVERY_N_FRAMES = _env_int("DAMAGE_EVERY_N_FRAMES", 10)
MAX_FRAMES = _env_int("MAX_FRAMES", 3000)  # safety limit

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

    # mp4v (MPEG-4 Part 2) is the default: it is bundled with OpenCV on all
    # platforms and plays in-browser from an .mp4 container without extra
    # codecs. H.264 (avc1) yields much smaller files but needs a system
    # openh264/ffmpeg build that isn't guaranteed here, so it stays opt-in
    # via PIPELINE_VIDEO_CODEC to avoid silently producing an empty file.
    default_codec = "mp4v" if output_path.suffix.lower() == ".mp4" else "XVID"
    codec = os.environ.get("PIPELINE_VIDEO_CODEC", default_codec)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

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

            for box, tid in zip(r.boxes.xyxy, r.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(tid)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy)
                })

                # TEAM CLASSIFICATION (ONCE)
                if track_id in cars and cars[track_id].team is None:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cars[track_id].set_team(predict_team(crop))

        # =============================
        # UPDATE TRACK MEMORY
        # =============================
        update_cars(cars, detections, frame_idx, fps)

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
                if frame_idx - car.last_seen > 5:
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
        collision_events = detect_collisions(cars, frame_idx)
        overtake_events = detect_overtakes(cars, frame_idx)

        total_collisions += len(collision_events)
        total_overtakes += len(overtake_events)

        # =============================
        # DRAW OUTPUT
        # =============================
        for car in cars.values():
            if car.last_bbox is None:
                continue

            x1, y1, x2, y2 = car.last_bbox

            label = f"ID {car.id}"
            if car.team:
                label += f" | {car.team}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Speed: {int(car.smoothed_speed)} px/s",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            if car.last_collision_frame == frame_idx:
                cv2.putText(frame, "COLLISION!",
                            (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 3)

            if hasattr(car, "last_overtake_frame"):
                if frame_idx - car.last_overtake_frame < 10:
                    cv2.putText(frame, "OVERTAKE!",
                                (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 0), 3)

            y = y1 - 60
            for dmg in car.damage.types:
                sev = car.damage.severity(dmg)
                cv2.putText(frame,
                            f"{dmg} ({sev})",
                            (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
                y -= 18

        out.write(frame)
        if progress_callback:
            progress_callback(frame_idx, total_frames)

    # =============================
    # CLEANUP
    # =============================
    cap.release()
    out.release()

    process_seconds = time.perf_counter() - process_start
    fps_processed = frame_idx / process_seconds if process_seconds > 0 else 0.0
    output_size_mb = (
        output_path.stat().st_size / (1024 * 1024) if output_path.is_file() else 0.0
    )
    print("Video processing finished")
    print(
        f"[TIMING] processed {frame_idx} frames in {process_seconds:.2f}s "
        f"({fps_processed:.1f} fps) | output {output_size_mb:.1f} MB "
        f"codec={codec}"
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
        "total_overtakes": total_overtakes,
        "total_collisions": total_collisions,
        "damage_type_counts": dict(sorted(damage_type_counts.items())),
        "damage_by_car": sorted(damage_by_car, key=lambda item: item["id"]),
        "team_counts": dict(sorted(team_counts.items())),
        "frames_processed": frame_idx,
        "source_video": str(video_path),
        "output_video": str(output_path),
        "output_video_name": output_name,
        "processing_seconds": round(process_seconds, 2),
        "processing_fps": round(fps_processed, 1),
        "output_size_mb": round(output_size_mb, 2),
    }
