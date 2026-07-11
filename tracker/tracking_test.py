import sys
import pathlib
import importlib
from pathlib import Path
import cv2
from ultralytics import YOLO
from fastai.vision.all import load_learner, PILImage

# ===== COLOR TERMINAL =====
from colorama import Fore, Style, init
init(autoreset=True)

# =============================
# PATH SETUP
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from tracking_memory.tracker_core import update_cars
from tracking_memory.damage_assigner import assign_damage
from tracking_memory.collision_detector import detect_collisions
from tracking_memory.overtake_detector import detect_overtakes
from app.config import (
    CAR_MODEL_PATH,
    DAMAGE_MODEL_PATH,
    TEAM_MODEL_PATH,
    TRACKER_CONFIG_PATH,
    verify_model_integrity,
)

# Per-run tracking state for this standalone script (update_cars now takes it
# as an argument instead of using a module-level global).
cars = {}

# =============================
# CONFIG
# =============================
video_path = BASE_DIR / "tracker" / "tracker2.mp4"

car_yolo_path = CAR_MODEL_PATH
damage_yolo_path = DAMAGE_MODEL_PATH
team_model_path = TEAM_MODEL_PATH
tracker_config_path = TRACKER_CONFIG_PATH
output_path = BASE_DIR / "output_tracking_final.mp4"

CONF_TH = 0.4
DAMAGE_CONF_TH = 0.4
TEAM_CONF_TH = 0.6
DAMAGE_EVERY_N_FRAMES = 5

# =============================
# LOAD MODELS
# =============================
print(Fore.CYAN + "📦 Loading Car YOLO model...")
car_model = YOLO(str(car_yolo_path))

print(Fore.CYAN + "📦 Loading Damage YOLO model...")
damage_model = YOLO(str(damage_yolo_path))

print(Fore.CYAN + "📦 Loading FastAI team classifier...")
verify_model_integrity(team_model_path)
if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11 or newer is required to load the team classifier.")
if sys.platform.startswith("win"):
    pathlib.PosixPath = pathlib.WindowsPath
for _plum_name in (
    "alias", "dispatcher", "function", "method", "overload", "parametric",
    "promotion", "resolver", "signature", "type", "util",
):
    try:
        sys.modules.setdefault(
            f"plum.{_plum_name}",
            importlib.import_module(f"plum._{_plum_name}"),
        )
    except ImportError:
        pass
team_model = load_learner(team_model_path)

# =============================
# UTILITIES
# =============================
SEVERITY_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

def predict_team(crop):
    try:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = PILImage.create(crop)
        pred, idx, probs = team_model.predict(img)
        if probs[idx].item() < TEAM_CONF_TH:
            return "UNKNOWN"
        return str(pred)
    except:
        return "UNKNOWN"

def draw_label(img, text, x, y, fg=(255,255,255), bg=(0,0,0)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x, y-h-8), (x+w+6, y), bg, -1)
    cv2.putText(img, text, (x+3, y-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 2)

def get_primary_damage(car):
    if not car.damage.types:
        return None, None
    best_dmg, best_sev, best_score = None, None, -1
    for dmg in car.damage.types:
        sev = car.damage.severity(dmg)
        score = SEVERITY_ORDER.get(sev, 0)
        if score > best_score:
            best_score = score
            best_dmg = dmg
            best_sev = sev
    return best_dmg, best_sev

# =============================
# VIDEO IO
# =============================
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError("Failed to open video")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps == 0:
    fps = 30

out = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

print(Fore.GREEN + f"🎥 Output video: {output_path}")
print(Fore.GREEN + f"🎞 FPS: {fps}")

# =============================
# MAIN LOOP
# =============================
frame_idx = 0
print(Fore.GREEN + "Tracking started...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

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

            detections.append({
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "center": ((x1+x2)//2, (y1+y2)//2)
            })

            if track_id in cars and cars[track_id].team is None:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    cars[track_id].set_team(predict_team(crop))

    update_cars(cars, detections, frame_idx, fps)

    # ===== DAMAGE =====
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

            dmg_results = damage_model(crop, conf=DAMAGE_CONF_TH, verbose=False)
            for r in dmg_results:
                if r.boxes is None:
                    continue

                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    dx1, dy1, dx2, dy2 = map(int, box)
                    damage_detections.append({
                        "bbox": [x1+dx1, y1+dy1, x1+dx2, y1+dy2],
                        "type": damage_model.names[int(cls)]
                    })

        assign_damage(cars, damage_detections, frame_idx)

    # ===== EVENTS =====
    for ev in detect_overtakes(cars, frame_idx):
        print(
            Fore.YELLOW +
            f"🏁 OVERTAKE | Frame {ev['frame']} | "
            f"Car {ev['overtaker']} -> Car {ev['overtaken']}"
        )

    detect_collisions(cars, frame_idx)

    # ===== DRAW =====
    for car in cars.values():
        if car.last_bbox is None:
            continue

        x1, y1, x2, y2 = car.last_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        header = f"ID {car.id}"
        if car.team:
            header += f" | {car.team}"
        draw_label(frame, header, x1, y1-8, bg=(0,120,0))

        draw_label(frame, f"{int(car.smoothed_speed)} px/s",
                   x1, y2+22, fg=(255,255,0))

        dmg, sev = get_primary_damage(car)
        if dmg:
            bg = (0,0,255) if sev == "HIGH" else (0,165,255)
            draw_label(frame, f"{dmg.upper()} ({sev})", x1, y1-30, bg=bg)

    out.write(frame)
    cv2.imshow("F1 Tracking – Final", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =============================
# SUMMARY
# =============================
print(Style.BRIGHT + Fore.GREEN + "\nDONE")
print(Fore.CYAN + f"🚗 Total tracked cars: {len(cars)}")

for car in cars.values():
    print(
        Fore.WHITE +
        f"Car {car.id} | Frames: {len(car.speed_history)} | "
        f"Avg speed: {int(car.smoothed_speed)} px/s | "
        f"Path: {int(car.path_length)} px | "
        f"Team: {car.team if car.team else 'UNKNOWN'} | "
        f"Damage: {list(car.damage.types.keys())}"
    )

cap.release()
out.release()
cv2.destroyAllWindows()
