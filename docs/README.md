# F1 Car Vision Intelligence System Documentation

This documentation suite was checked against the current codebase in this workspace. It describes the app as implemented after the runtime path cleanup.

## Project Overview

The system is a local FastAPI computer vision app for Formula 1-style race footage and vehicle damage images. It can:

* Upload a single car/damage image and return an annotated result image.
* Upload a video and run car detection, tracking, team classification, damage detection, collision detection, and overtake detection.
* Run curated demo image/video actions without manually selecting files.
* Queue video-processing jobs and poll job progress while processing continues in the background.
* Run a Phase 4 benchmark manifest for measured image/video model behavior.
* Render a browser UI with Jinja2 templates.
* Store uploads and generated outputs on the local filesystem.

## Repository Architecture Map

```text
.
|-- app/
|   |-- main.py                  # FastAPI app, CORS, static mounts, /api root
|   |-- api.py                   # Routes, upload validation, safe path handling
|   |-- config.py                # Shared paths, CORS, upload limits, model hash checks
|   |-- static/                  # Static assets and image results
|   `-- templates/               # Jinja2 HTML pages
|-- services/
|   |-- damage_image.py          # Wrapper for src/detection.py
|   `-- full_pipeline.py         # Wrapper for src/pipeline.py
|-- src/
|   |-- classification.py        # FastAI training script
|   |-- detection.py             # Single-image damage detector and heuristics
|   |-- tracking.py              # CarState class and motion state
|   `-- pipeline.py              # Video analysis loop
|-- tracking_memory/
|   |-- tracker_core.py          # Global cars dictionary
|   |-- car_state.py             # CarState import bridge
|   |-- damage_state.py          # Damage history and severity
|   |-- damage_assigner.py       # IoU damage-to-car assignment
|   |-- collision_detector.py    # Collision event logic
|   |-- overtake_detector.py     # Rank/path-length overtake logic
|   `-- utils.py                 # IoU helper
|-- tracker/
|   |-- bytetrack.yaml           # ByteTrack config present in repo
|   |-- tracking_test.py         # GUI tracking harness
|   `-- tracking_test2.py        # Headless tracking harness
|-- models/
|   |-- best_carDD.pt            # Runtime damage detector
|   |-- best_f1_detect.pt        # Present, but current code loads yolo_model_robflow/.../best.pt
|   `-- f1_team_classifier.pkl   # Runtime team classifier
|-- yolo_model_robflow/
|   `-- runs/detect/train/weights/best.pt  # Runtime car detector
|-- uploads/
|-- outputs/
|-- docs/
|-- requirements.txt
`-- data.yaml
```

## Current Alignment Notes

The main runtime path mismatches have been fixed:

* Model and tracker paths are centralized in `app/config.py`.
* `src/detection.py` and `src/pipeline.py` use `models/best_carDD.pt` for damage inference.
* `src/pipeline.py` uses `tracker/bytetrack.yaml`.
* `/pipeline/run` accepts whitelisted local videos under `tracker/` and `uploads/videos/`.
* `/demo/image` and `/demo/video` provide one-click demo paths for the UI.
* `/jobs/pipeline/run`, `/jobs/pipeline/video`, and `/jobs/demo/video` queue background video jobs.
* `/api/jobs/{job_id}` exposes queued/running/completed/failed state for the polling UI.
* Pipeline outputs are generated as `.mp4`, and templates use the web-safe `video_path`.
* Result screens include download links for generated media.
* `benchmarks/run_benchmarks.py` writes evidence-bound benchmark JSON and Markdown results.
* Remaining maintenance note: `src/classification.py` exports to a root-level `f1_team_classifier.pkl`, while runtime loads `models/f1_team_classifier.pkl`.

## Documentation Index

1. [ARCHITECTURE.md](ARCHITECTURE.md): System architecture and data flow.
2. [DATABASE.md](DATABASE.md): In-memory state and file-based outputs.
3. [API_REFERENCE.md](API_REFERENCE.md): FastAPI routes, upload policies, and errors.
4. [FRONTEND.md](FRONTEND.md): Template screens and UI/backend mismatches.
5. [BACKEND.md](BACKEND.md): File-by-file backend analysis.
6. [AI_SYSTEM.md](AI_SYSTEM.md): Models, thresholds, paths, and CV pipeline.
7. [SECURITY_AUDIT.md](SECURITY_AUDIT.md): Current mitigations and open risks.
8. [DEPLOYMENT.md](DEPLOYMENT.md): Local setup and run blockers.
9. [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md): Training, tuning, and cleanup notes.

Phase 4 benchmark files live under `benchmarks/` at the project root.

The root `PROJECT_REVERSE_ENGINEERING_REPORT.md` is a higher-level audit report. The files in `docs/` are the more maintainable source docs.
