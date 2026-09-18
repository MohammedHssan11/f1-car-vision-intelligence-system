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

**Car detection** — a custom Formula-style race-car dataset was collected and
**annotated on Roboflow** (YOLO format), then used to **fine-tune YOLOv8** for
robustness to broadcast camera motion, small/fast-moving objects, and team
livery variation.

**Damage detection** — YOLOv8 was **fine-tuned on the public CarDD dataset**
(7 classes: scratch, dent, glass shatter, lamp broken, tire flat, deformation,
other damage).

**Team classification** — a **ResNet34** (FastAI) classifier was fine-tuned on
per-team race-car images across 8 teams.

Combining a fine-tuned detector, a benchmark damage model, and a livery
classifier lets the system go **beyond off-the-shelf models**.

---

## 📈 Model Performance

Measured on held-out data (methodology in [`benchmarks/`](benchmarks/)).

**Car detector — YOLOv8 (`f1_car`)** · 101-image held-out validation set

| Metric | Score |
|--------|-------|
| mAP@0.5 | **0.983** |
| mAP@0.5:0.95 | **0.937** |
| Precision | 0.970 |
| Recall | 0.937 |

**Team classifier — FastAI ResNet34** · 477 validation images, 8 teams

| Metric | Score |
|--------|-------|
| Overall accuracy | **0.973** (464 / 477) |
| Perfect (1.00) | Ferrari · Renault · Williams |
| Weakest | AlphaTauri 0.83 — confused with Red Bull (near-identical sister-team livery) |

> ℹ️ Team accuracy is *indicative*: validation labels are derived from filenames and may overlap the training split. Damage-detector per-class mAP is pending re-evaluation on the CarDD test set.

---

## 🔬 Detected Events

| Event | Detection Logic |
|------|----------------|
| Collision | Probable impact: strong recent deceleration + damage confirmed on two inference frames |
| Overtake | Temporal rank change (path-based) |
| Damage | YOLO detection + temporal validation |

---

## 📊 Damage Severity

Damage severity is inferred from its elapsed persistence in the video (not the
number of sparse damage-inference calls):

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
- Overtakes are heuristic until track coordinates and labelled event footage are available
- Very small damages may be missed

Collision results are conservative visual-impact candidates, not official race
steward decisions. These limits are mitigated using temporal smoothing,
stable-ID recovery, and multi-signal logic.

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
