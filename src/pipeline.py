import sys
from pathlib import Path
import cv2

# =============================
# PATH SETUP
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# =============================
# IMPORTS
# =============================
from ultralytics import YOLO
from fastai.vision.all import load_learner, PILImage

from tracking_memory.tracker_core import update_cars, cars
from tracking_memory.damage_assigner import assign_damage
from tracking_memory.collision_detector import detect_collisions
from tracking_memory.overtake_detector import detect_overtakes

# =============================
# CONFIG (GLOBAL)
# =============================
CONF_TH = 0.4
DAMAGE_CONF_TH = 0.4
TEAM_CONF_TH = 0.6
DAMAGE_EVERY_N_FRAMES = 5
MAX_FRAMES = 5000  # safety limit

car_yolo_path = (
    BASE_DIR
    / "yolo_model_robflow"
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)

damage_yolo_path = Path(
    r"C:\TERM 7\computer vision\final project\data\CarDD_YOLO\runs\detect\train2\weights\best.pt"
)

team_model_path = BASE_DIR / "models" / "f1_team_classifier.pkl"

# =============================
# LOAD MODELS (ONCE)
# =============================
print("📦 Loading models...")
car_model = YOLO(str(car_yolo_path))
damage_model = YOLO(str(damage_yolo_path))
team_model = load_learner(team_model_path)

# =============================
# TEAM PREDICTION
# =============================
def predict_team(crop):
    try:
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
def run_full_pipeline(video_path, output_name="result.avi"):
    """
    Run full F1 analysis pipeline

    Args:
        video_path (str | Path): input video
        output_name (str): output video file name

    Returns:
        dict: summary
    """

    # =============================
    # RESET GLOBAL STATE
    # =============================
    cars.clear()

    # =============================
    # OUTPUT PATH (IMPORTANT)
    # =============================
    output_dir = BASE_DIR / "outputs" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name
    print(f"[PIPELINE] Saving output to: {output_path}")

    # =============================
    # VIDEO IO
    # =============================
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("❌ Failed to open input video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not out.isOpened():
        raise RuntimeError("❌ Failed to open output video writer")

    # =============================
    # METRICS
    # =============================
    frame_idx = 0
    total_collisions = 0
    total_overtakes = 0

    # =============================
    # MAIN LOOP
    # =============================
    print("🚀 Processing video...")
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
            tracker="trackers/bytetrack.yaml",
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
        update_cars(detections, frame_idx, fps)

        # =============================
        # DAMAGE DETECTION
        # =============================
        damage_detections = []

        if frame_idx % DAMAGE_EVERY_N_FRAMES == 0:
            for car in cars.values():
                if car.last_bbox is None:
                    continue
                if frame_idx - car.last_seen > 5:
                    continue

                x1, y1, x2, y2 = car.last_bbox
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                dmg_results = damage_model(
                    crop,
                    conf=DAMAGE_CONF_TH,
                    verbose=False
                )

                for r in dmg_results:
                    for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                        dx1, dy1, dx2, dy2 = map(int, box)
                        damage_detections.append({
                            "bbox": [x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2],
                            "type": damage_model.names[int(cls)]
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

    # =============================
    # CLEANUP
    # =============================
    cap.release()
    out.release()

    print("✅ Video processing finished")

    return {
        "cars_tracked": len(cars),
        "total_overtakes": total_overtakes,
        "total_collisions": total_collisions,
        "output_video": str(output_path)
    }
