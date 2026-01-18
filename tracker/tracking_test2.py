import sys
from pathlib import Path

# =============================
# PATH SETUP
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# =============================
# IMPORTS
# =============================
import cv2
from ultralytics import YOLO
from fastai.vision.all import load_learner, PILImage

from tracking_memory.tracker_core import update_cars, cars
from tracking_memory.damage_assigner import assign_damage

# =============================
# CONFIG
# =============================
video_path = BASE_DIR / "tracker" / "tracker2.mp4"

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
output_path = BASE_DIR / "output_tracking.avi"

CONF_TH = 0.4
DAMAGE_CONF_TH = 0.4
TEAM_CONF_TH = 0.6

# =============================
# LOAD MODELS
# =============================
print("📦 Loading Car YOLO model...")
car_model = YOLO(str(car_yolo_path))

print("📦 Loading Damage YOLO model...")
damage_model = YOLO(str(damage_yolo_path))

print("📦 Loading FastAI team classifier...")
team_model = load_learner(team_model_path)

# =============================
# TEAM PREDICTION
# =============================
def predict_team(crop):
    try:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = PILImage.create(crop)

        pred_class, pred_idx, probs = team_model.predict(img)
        confidence = probs[pred_idx].item()

        if confidence < TEAM_CONF_TH:
            return "UNKNOWN"

        return str(pred_class)

    except Exception:
        return "UNKNOWN"

# =============================
# VIDEO IO
# =============================
cap = cv2.VideoCapture(str(video_path))
assert cap.isOpened(), "❌ Failed to open video"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
assert out.isOpened(), "❌ Failed to open VideoWriter"

# =============================
# MAIN LOOP
# =============================
frame_idx = 0
print("🚀 Tracking started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # =============================
    # CAR TRACKING
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
    update_cars(detections, frame_idx)

    # =============================
    # DAMAGE DETECTION
    # =============================
    damage_detections = []

    for car in cars.values():

        if car.last_bbox is None:
            continue

        if frame_idx - car.last_seen > 5:
            continue

        x1, y1, x2, y2 = car.last_bbox
        car_crop = frame[y1:y2, x1:x2]

        if car_crop.size == 0:
            continue

        damage_results = damage_model(
            car_crop,
            conf=DAMAGE_CONF_TH,
            verbose=False
        )

        for r in damage_results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                dx1, dy1, dx2, dy2 = map(int, box)
                damage_detections.append({
                    "bbox": [x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2],
                    "type": damage_model.names[int(cls)]
                })

    # =============================
    # ASSIGN DAMAGE
    # =============================
    assign_damage(cars, damage_detections, frame_idx)

    # =============================
    # DRAW OUTPUT
    # =============================
    for car in cars.values():

        if car.last_bbox is None:
            continue

        if frame_idx - car.last_seen > 5:
            continue

        x1, y1, x2, y2 = car.last_bbox

        label = f"ID {car.id}"
        if car.team:
            label += f" | {car.team}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        y = y1 - 30
        for dmg in car.damage.types:
            sev = car.damage.severity(dmg)
            cv2.putText(
                frame,
                f"{dmg} ({sev})",
                (x1, max(y, 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
            y -= 18

    # =============================
    # WRITE & SHOW
    # =============================
    out.write(frame)
    cv2.imshow("F1 Tracking + Damage Reasoning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =============================
# CLEANUP
# =============================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ DONE")
print(f"🚗 Total tracked cars: {len(cars)}")
