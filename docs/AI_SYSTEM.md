# AI Systems & Pipeline Reconstruction

This document maps the current computer vision stack to the code and model files in the repository.

## AI/LLM/RAG Presence Verification

* **Large Language Models:** Not present in the codebase.
* **Vector Databases / Embeddings:** Not present in the active app.
* **RAG / Prompting Systems:** Not present in the active app.

## Computer Vision Inference Stack

The application combines YOLOv8 detection/tracking, a YOLO damage detector, and a FastAI ResNet34 team classifier.

```mermaid
graph LR
    A[Input image or video frame] --> B[YOLOv8 car detector and ByteTrack]
    A --> C[YOLOv8 damage detector]
    B --> D[Car crop]
    D --> E[FastAI ResNet34 team classifier]
    B & C & E --> F[tracking_memory state and event logic]
```

## YOLOv8 Car Detector and Tracker

* **Framework:** Ultralytics YOLOv8 / PyTorch.
* **Runtime Weight Path:** `yolo_model_robflow/runs/detect/train/weights/best.pt`.
* **Duplicate Available Weight:** `models/best_f1_detect.pt` exists and has the same size as the runtime `best.pt`, but the current code loads the `yolo_model_robflow/.../best.pt` path.
* **Dataset Config:** `data.yaml`.
* **Class Labels:** `['f1_car']`.
* **Confidence Threshold:** `CONF_TH = 0.4`.
* **Tracker Config:** `tracker/bytetrack.yaml`, configured by `app.config.TRACKER_CONFIG_PATH`.

## YOLOv8 Damage Detector

* **Framework:** Ultralytics YOLOv8 / PyTorch.
* **Runtime Path in Code:** `models/best_carDD.pt`, configured by `app.config.DAMAGE_MODEL_PATH`.
* **Image Endpoint Threshold:** `BASE_CONF = 0.45` in `src/detection.py`.
* **Video Pipeline Threshold:** `DAMAGE_CONF_TH = 0.4` in `src/pipeline.py`.
* **Damage Classes Used by Logic:** `tire_flat`, `glass_shatter`, `dent`, `scratch`, plus derived `glass_missing` from geometry heuristics.

Damage post-processing rules in `src/detection.py`:

* Ignore boxes with height below 3% of image height.
* Keep `tire_flat` only when centered in the lower 25% of the image; very large tire boxes become `glass_shatter`.
* Convert large upper-frame `glass_shatter` regions to `glass_missing`.
* Convert low-confidence or low-position `glass_shatter` detections to `scratch`.
* Convert aggregate glass area over 18% of the image to global `glass_missing`.

## FastAI Team Classifier

* **Framework:** FastAI with a ResNet34 backbone.
* **Runtime Model Path:** `models/f1_team_classifier.pkl`.
* **Integrity Control:** `app.config.verify_model_integrity()` checks the model SHA-256 before `load_learner()` runs.
* **Pinned SHA-256:** `d6594c0c1d5c7804ed65e20ce228eb07dbdf6b2829a244105d08577a53b244af`.
* **Inference Threshold:** `TEAM_CONF_TH = 0.6`; lower-confidence predictions become `"UNKNOWN"`.
* **Training Script Mismatch:** `src/classification.py` exports to the project root `f1_team_classifier.pkl`, while runtime loads `models/f1_team_classifier.pkl`.

## Video Processing Pipeline Trace

```mermaid
flowchart TD
    A[Input video] --> B[cv2.VideoCapture]
    B --> C[car_model.track]
    C --> D[Update tracking_memory.cars]
    D --> E[Classify team once per track ID]
    D --> F{Frame index divisible by 5}
    F -- Yes --> G[Crop active cars and run damage YOLO]
    G --> H[Assign damage by IoU > 0.3]
    F -- No --> I[Skip damage inference]
    H --> J[Detect collisions]
    I --> J
    J --> K[Detect overtakes]
    K --> L[Draw overlays]
    L --> M[cv2.VideoWriter output AVI]
```

## Event Logic

| Event | Current Logic |
| ----- | ------------- |
| Collision | Latest acceleration below `-300` and new damage first seen on the same frame; 15-frame cooldown. |
| Overtake | Active cars sorted by `path_length`; rank improvement after cooldown creates an event. |
| Damage Severity | `LOW` for up to 10 observed frames, `MEDIUM` for more than 10, `HIGH` for more than 30. |

## Current Runtime Notes

* `src/pipeline.py` checks that the car model, damage model, team model, and tracker config exist before loading models.
* Pipeline outputs are requested as `.mp4` by the API and written with an MP4 codec when the output suffix is `.mp4`.
* The remaining model maintenance mismatch is in `src/classification.py`, which exports to the project root while runtime loads `models/f1_team_classifier.pkl`.

## Phase 4 Benchmark Harness

The benchmark harness lives under `benchmarks/`:

* `benchmarks/manifest.json` references existing image/video samples without duplicating data.
* `benchmarks/run_benchmarks.py` runs the current runtime models against those samples.
* `benchmarks/benchmark_results.md` is the human-readable output.
* `benchmarks/results/benchmark_results.json` is the machine-readable output.

Current evidence boundary:

* Team labels are available for selected image samples through dataset filenames.
* Car box labels are not available in the manifest, so car detection counts are measured but not scored.
* Damage labels are not available in the manifest, so damage detections are measured but not scored.
* Video track/event labels are not available, so tracked cars, collisions, overtakes, and damage counts are qualitative pipeline observations.
