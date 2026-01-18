import cv2
from ultralytics import YOLO
import numpy as np
import csv
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
VIDEO_PATH = r"C:\Users\mh978\Downloads\car accident right after the crash damage inspection - dobroslawja (720p, h264).mp4"
OUTPUT_PATH = "damage_timeline_output.mp4"
TIMELINE_CSV = "damage_timeline.csv"
TIMELINE_PLOT = "damage_timeline.png"

CAR_MODEL_PATH = "yolov8n.pt"
DAMAGE_MODEL_PATH = r"C:\TERM 7\computer vision\final project\data\CarDD_YOLO\runs\detect\train2\weights\best.pt"

CAR_CONF = 0.4
DAMAGE_CONF = 0.2
IOU_TH = 0.5

# =========================
# LOAD MODELS
# =========================
print("📦 Loading models...")
car_model = YOLO(CAR_MODEL_PATH)
damage_model = YOLO(DAMAGE_MODEL_PATH)

# =========================
# VIDEO SETUP
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

# =========================
# MEMORY (SINGLE CAR)
# =========================
car_box = None
damage_memory = []        # all detected damage boxes
damage_count = 0

timeline_frames = []
timeline_damage = []

frame_idx = 0

# =========================
# IOU FUNCTION
# =========================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(1, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    boxBArea = max(1, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # -------------------------
    # CAR DETECTION (LARGEST ONLY)
    # -------------------------
    car_results = car_model(frame, conf=CAR_CONF, verbose=False)
    vehicle_boxes = []

    for r in car_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = car_model.names[int(box.cls[0])]
            if cls in ["car", "truck", "bus"]:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                area = (x2-x1)*(y2-y1)
                vehicle_boxes.append((area, (x1,y1,x2,y2)))

    if vehicle_boxes:
        _, car_box = max(vehicle_boxes, key=lambda x: x[0])

    if car_box is None:
        continue

    x1,y1,x2,y2 = car_box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # -------------------------
    # DAMAGE DETECTION
    # -------------------------
    dmg_results = damage_model(crop, conf=DAMAGE_CONF, verbose=False)

    for r in dmg_results:
        if r.boxes is None:
            continue
        for dbox in r.boxes:
            dx1,dy1,dx2,dy2 = map(int, dbox.xyxy[0])
            fx1,fy1 = dx1+x1, dy1+y1
            fx2,fy2 = dx2+x1, dy2+y1
            new_box = (fx1,fy1,fx2,fy2)

            is_new = True
            for old in damage_memory:
                if compute_iou(new_box, old) > 0.5:
                    is_new = False
                    break

            if is_new:
                damage_memory.append(new_box)
                damage_count += 1
                cv2.rectangle(frame,(fx1,fy1),(fx2,fy2),(0,0,255),2)

    # -------------------------
    # TIMELINE LOGGING
    # -------------------------
    timeline_frames.append(frame_idx)
    timeline_damage.append(damage_count)

    # -------------------------
    # DRAW OVERLAY
    # -------------------------
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(frame,f"Damage Count: {damage_count}",
                (30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    # timeline bar
    bar_x = int((damage_count / max(1, max(timeline_damage))) * 300)
    cv2.rectangle(frame,(30,60),(30+bar_x,80),(255,0,0),-1)

    out.write(frame)

# =========================
# CLEANUP
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

# =========================
# SAVE CSV
# =========================
with open(TIMELINE_CSV,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame","Damage_Count"])
    for f_id, d in zip(timeline_frames, timeline_damage):
        writer.writerow([f_id,d])

# =========================
# PLOT TIMELINE
# =========================
plt.figure(figsize=(10,5))
plt.plot(timeline_frames, timeline_damage)
plt.xlabel("Frame")
plt.ylabel("Total Damage Regions")
plt.title("Damage Accumulation Over Time")
plt.grid()
plt.savefig(TIMELINE_PLOT)
plt.close()

print("✅ Finished Successfully")
print(f"🧾 CSV saved: {TIMELINE_CSV}")
print(f"📈 Timeline plot saved: {TIMELINE_PLOT}")
print(f"🚗 Final Damage Count: {damage_count}")
