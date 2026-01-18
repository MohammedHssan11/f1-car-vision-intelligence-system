# 🏎️ F1 Car Vision Intelligence System

An end-to-end **computer vision system** for race car analysis and **vehicle damage detection** using video and image-based inference.

The system converts raw race footage into structured intelligence using **detection, tracking, fine-tuned deep learning models, and temporal reasoning** — without relying on official telemetry.

---

## 🚀 What This System Does

### 🎥 Video Analysis
- Detects and tracks multiple race cars (persistent IDs)
- Classifies teams using car livery
- Estimates speed and acceleration
- Detects **collisions** and **overtakes**
- Tracks and counts **car damage over time**
- Produces fully annotated output videos

### 🖼️ Image Damage Detection
- Detects damage from a **single image**
- Supports multiple damage types
- Estimates damage severity
- Outputs annotated images

---

## 🧠 How It Works (High Level)

- Uses **stateful tracking** instead of frame-by-frame decisions
- Maintains a **temporal state per car**
- Events are detected using **multiple signals combined over time**
- Designed for **real-world race footage**, not clean datasets

---

## 🏷️ Dataset & Model Training

- A **custom dataset of Formula-style race cars** was collected
- Cars and damage regions were **manually annotated** in YOLO format
- **YOLOv8 was fine-tuned** on this dataset to reliably detect:
  - Formula race cars
  - Visual damage regions
- Fine-tuning improves robustness to:
  - Broadcast camera motion
  - Small, fast-moving objects
  - Team livery variations

This allows the system to go **beyond off-the-shelf models**.

---

## 🔬 Detected Events

| Event | Detection Logic |
|------|----------------|
| Collision | Sudden deceleration + new damage |
| Overtake | Temporal rank change (path-based) |
| Damage | YOLO detection + temporal validation |

---

## 📊 Damage Severity

Damage severity is inferred from how long it persists:

- **LOW** → < 10 frames  
- **MEDIUM** → 10–30 frames  
- **HIGH** → > 30 frames  

---

## 🧰 Technologies Used

- **YOLOv8** — fine-tuned detection
- **ByteTrack** — multi-object tracking
- **FastAI (ResNet34)** — team classification
- **OpenCV** — video processing
- **FastAPI** — backend API
- **Jinja2** — web interface

---

## ⚙️ Performance Considerations

- Damage detection runs every *N* frames
- Cropped inference instead of full-frame
- Lightweight models for near real-time performance

---

## ⚠️ Limitations

- Pixel-based speed estimation (no telemetry)
- Broadcast camera motion introduces noise
- Very small damages may be missed

These are mitigated using temporal smoothing and multi-signal logic.

---

## 🔮 Future Improvements

- World-coordinate speed estimation
- SAM-based damage refinement
- Multi-camera fusion
- Telemetry integration
- Event analytics dashboard

---

## 🏁 Summary

This project demonstrates how **computer vision, tracking, and temporal reasoning** can be combined to build a realistic race analysis system from raw video.

It reflects real-world engineering trade-offs and serves as a strong foundation for **sports analytics and automotive vision research**.

---

## 🏷️ Tags

`computer-vision` `yolo` `object-tracking` `deep-learning`  
`video-analysis` `damage-detection` `sports-analytics`
