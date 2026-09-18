# System Deployment Guide

This guide describes how the current codebase is expected to run locally after the runtime path cleanup.

## Environment Prerequisites

### Operating System Support

* **Docker (any OS):** The recommended path - the CPU-only image below runs identically on Windows, Linux, and macOS. See "Docker" below.
* **Windows Native:** Supported for local (non-container) runs.
* **Linux/macOS Native:** Use the Docker image; native runs rely on relative paths resolved from the project root.

### Hardware Dependencies

* **CPU:** Quad-core x86_64 or better.
* **GPU:** NVIDIA CUDA GPU recommended for YOLO inference.
* **Memory:** 16 GB RAM minimum is recommended because the app loads multiple deep learning models.

### Software Requirements

* Python `>= 3.11` for the bundled FastAI classifier pickle
* PyTorch/TorchVision matching the local CUDA or CPU setup
* FastAPI/Uvicorn stack
* Ultralytics YOLOv8
* FastAI
* OpenCV

Install the listed packages with:

```powershell
pip install -r requirements.txt
```

`requirements.txt` has been cleaned so it contains installable Python package requirements only. Python itself still needs to be installed separately; use Python 3.11 or newer.

## Docker (recommended for a portable, one-command run)

The project ships a CPU-only container so it runs on any machine with Docker -
no Python, CUDA, or NVIDIA toolkit required on the host. The Linux container
also sidesteps the Windows-only path caveat above.

```bash
# From the project root:
docker compose up --build

# App:  http://localhost:8010
#       http://localhost:8010/page/damage
#       http://localhost:8010/page/pipeline
#       http://localhost:8010/docs
```

Stop with `Ctrl+C`, or `docker compose down`.

**What's in the image (`Dockerfile`):**

* Base `python:3.11-slim` (3.11 is mandatory for the FastAI pickle).
* CPU-only `torch==2.6.0` / `torchvision==0.21.0` installed from PyTorch's
  CPU wheel index *before* `requirements.txt`, so the default CUDA wheel is
  never pulled; this keeps the image at a few GB instead of ~8 GB.
* System libs `libgl1` + `libglib2.0-0` for OpenCV.
* App code, the three runtime model files, `tracker/` assets, and the single
  demo sample image. The ~800 MB training dataset, generated media, caches,
  and git history are trimmed by `.dockerignore`.

**Configuration** is via environment variables (see `.env.example`; copy to
`.env` and `docker compose` picks it up automatically):

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `ALLOWED_ORIGINS` | `http://localhost:8010,http://127.0.0.1:8010` | CORS allow-list. |
| `F1_APP_PORT` | `8010` | Host port for the Docker service; avoids a collision with other local apps on 8000. |
| `DAMAGE_EVERY_N_FRAMES` | `10` | Damage-inference stride. |
| `MAX_FRAMES` | `3000` | Per-video frame cap. |
| `PIPELINE_VIDEO_CODEC` | `mp4v` | Temporary OpenCV FourCC; ffmpeg publishes the final H.264/AAC browser video. |
| `PIPELINE_H264_CRF` | `23` | Final H.264 quality/size trade-off. |
| `PIPELINE_SHOW_SPEED` | `false` | Enables optional per-car speed text in the annotated video. |
| `TRACK_REASSOCIATION_MAX_GAP` | `30` | Short loss window eligible for conservative stable-ID recovery. |
| `COLLISION_MIN_DAMAGE_OBSERVATIONS` | `2` | Damage confirmations required for a probable impact. |

**Persistence:** `docker-compose.yml` bind-mounts `./outputs` and `./uploads`
into the container, so generated result videos and uploads land on the host
and survive restarts.

**Verify the running container:**

```bash
curl http://localhost:8010/api/health          # {"status":"ok",...}
curl http://localhost:8010/api/info             # device + model-load status
```

Because the image is CPU-only, `/api/info` reports `"device":"cpu"` and video
processing is slower than a local GPU run — expected, and the tradeoff for a
portable image. To use a host GPU instead, a CUDA base image plus
`--gpus all` / the NVIDIA Container Toolkit would be required (not configured
here by design).

## Local Setup

Create and activate a virtual environment:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run Phase 1 smoke tests:

```powershell
py -3.11 tests\smoke_phase1.py
```

Run the app:

```powershell
py -3.11 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

Open:

* `http://127.0.0.1:8010/`
* `http://127.0.0.1:8010/page/damage`
* `http://127.0.0.1:8010/page/pipeline`
* `http://127.0.0.1:8010/docs`

## Runtime Model Paths

All active runtime model artifacts are loaded from `models/` and SHA-256
verified before deserialization:

| Model | Path Used by Code | Current Workspace Status |
| ----- | ----------------- | ------------------------ |
| Car detector/tracker | `models/best_f1_detect.pt` | Present and hash-pinned in `app/config.py`. |
| Damage detector | `models/best_carDD.pt` | Present and hash-pinned in `app/config.py`. |
| Team classifier | `models/f1_team_classifier.pkl` | Present and hash-pinned in `app/config.py`. |

## Required Directories

`app/config.py` creates these directories on import:

* `uploads/images/`
* `uploads/videos/`
* `outputs/videos/`
* `app/static/results/`

`app/main.py` also mounts:

* `app/static/`
* `outputs/`

Those parent directories must exist before startup.

## CORS Configuration

Default allowed origins:

* `http://localhost:8010`
* `http://127.0.0.1:8010`

Override for deployment:

```powershell
$env:ALLOWED_ORIGINS = "https://example.com,https://admin.example.com"
```

## Current Run Notes

The previous runtime path mismatches have been fixed in code:

1. Damage inference uses `models/best_carDD.pt`.
2. Tracking uses `tracker/bytetrack.yaml`.
3. `/pipeline/run` allows whitelisted videos under `tracker/` and `uploads/videos/`.
4. Pipeline result playback uses the web-safe `video_path`.
5. Pipeline outputs are generated as `.mp4` for browser playback.

Before a clean run, install dependencies and confirm the three model files in the runtime model table exist.
