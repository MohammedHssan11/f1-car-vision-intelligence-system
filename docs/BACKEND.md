# Backend Codebase Forensic Analysis

This document describes the current backend source files and the runtime behavior they implement.

## Component: Application Layer (`app/`)

### `app/main.py`

* **Purpose:** Initializes the FastAPI application, configures CORS, mounts static directories, registers routes, and exposes `GET /api`.
* **CORS:** Imports `ALLOWED_ORIGINS` from `app.config`. Default allowed origins are `http://localhost:8000` and `http://127.0.0.1:8000`; deployments can override them with the `ALLOWED_ORIGINS` environment variable.
* **Static Mounts:**
  * `/static` -> `app/static`
  * `/outputs` -> `outputs`
* **Router Registration:** Includes the router from `app.api`.
* **Startup Dependency:** `app/static` and `outputs` must exist before `StaticFiles` mounts are created. `app/config.py` creates important child directories such as `app/static/results` and `outputs/videos`.

### `app/config.py`

* **Purpose:** Centralizes paths, upload policy, CORS configuration, and model integrity verification.
* **Directory Constants:**
  * `UPLOAD_IMAGE_DIR = BASE_DIR / "uploads" / "images"`
  * `UPLOAD_VIDEO_DIR = BASE_DIR / "uploads" / "videos"`
  * `OUTPUT_VIDEO_DIR = BASE_DIR / "outputs" / "videos"`
  * `OUTPUT_IMAGE_DIR = BASE_DIR / "app" / "static" / "results"`
  * `SAMPLE_IMAGE_PATH = BASE_DIR / "splited_dataset" / "val" / "valid_images" / "ferrari_f1_car_0014.jpg"`
  * `SAMPLE_VIDEO_PATH = BASE_DIR / "tracker" / "tracker2.mp4"`
* **Upload Policy:**
  * Images: `.jpg`, `.jpeg`, `.png`, `.bmp`; max 20 MB.
  * Videos: `.mp4`, `.avi`, `.mov`, `.mkv`; max 500 MB.
* **Model Integrity:** `verify_model_integrity()` checks `models/f1_team_classifier.pkl` against a pinned SHA-256 before FastAI deserializes it.

### `app/api.py`

* **Purpose:** Defines page routes, upload endpoints, validation helpers, and status APIs.
* **Important Helpers:**
  * `_save_validated_upload()`: validates extension, streams uploads to disk in chunks, enforces configured size caps, and deletes oversized partial files.
  * `_resolve_safe_video_path()`: resolves `/pipeline/run` form input under whitelisted local roots (`tracker/` and `uploads/videos/`) and rejects absolute paths or path traversal.
* **Routes:**
  * `GET /`: renders `index.html`.
  * `GET /page/damage`: renders `damage_image.html`.
  * `GET /page/pipeline`: renders `full_pipeline.html`.
  * `GET /jobs/{job_id}`: renders `job_status.html`.
  * `GET /jobs/{job_id}/result`: renders the final debrief when a job is complete, or the job status page while pending/failed.
  * `POST /damage/image`: saves a validated image upload and calls `detect_damage_image()`.
  * `POST /demo/image`: runs the configured sample image and renders a downloadable annotated result.
  * `POST /jobs/pipeline/video`: saves and validates a video upload, queues it as a background job, and redirects to the job page.
  * `POST /jobs/pipeline/run`: validates a safe local video path, queues it as a background job, and redirects to the job page.
  * `POST /jobs/demo/video`: queues the configured sample video as a background job.
  * `POST /pipeline/video`: saves a validated video upload and calls `run_full_pipeline()`.
  * `POST /demo/video`: runs the configured sample video and renders the debrief page.
  * `POST /pipeline/run`: runs the pipeline on a safe relative path under `tracker/` or `uploads/videos/`.
  * `GET /api/jobs/{job_id}`: returns JSON status for background video jobs.
  * `GET /api/health`: returns service health JSON.
  * `GET /api/info`: returns the feature list plus runtime `device` (CPU/GPU) and per-model load status JSON.
* **Local Demo Default:** `full_pipeline.html` defaults the text input to `tracker/tracker2.mp4`, which is allowed by the backend's local-video whitelist. The same file is used by `/demo/video`.

### `app/jobs.py`

* **Purpose:** Maintains process-local background video jobs and runs the full pipeline outside the request/response cycle.
* **Worker Model:** Uses `ThreadPoolExecutor(max_workers=1)` so video jobs are serialized. This avoids concurrent mutation of the shared tracking state in `tracking_memory.tracker_core.cars`.
* **State Store:** In-memory dictionary protected by a lock. Restarting the process clears job metadata.
* **Statuses:** `queued`, `running`, `completed`, and `failed`.
* **Duplicate Handling:** If a new job uses the same source path as an active job, it is still queued with its own output file and marked with `duplicate_of`.
* **Progress:** Receives frame progress from `src.pipeline.run_full_pipeline()` through an optional callback.

## Component: Service Integration Layer (`services/`)

### `services/damage_image.py`

Adds `src/` to `sys.path`, imports `detect_damage_image` from `src/detection.py`, and re-exports it for the API layer.

### `services/full_pipeline.py`

Adds `src/` to `sys.path`, imports `run_full_pipeline` from `src/pipeline.py`, and re-exports it for the API layer.

## Component: Core Processing Layer (`src/`)

### `src/classification.py`

* **Purpose:** Trains a FastAI ResNet34 classifier for F1 team/livery recognition.
* **Dataset Path:** Hardcoded to `C:\TERM 7\computer vision\final project\Formula One Cars`.
* **Export Path:** Hardcoded to `C:\TERM 7\computer vision\final project\f1_team_classifier.pkl`.
* **Important Mismatch:** Runtime inference loads `models/f1_team_classifier.pkl`, so the training script's export location does not currently match the runtime model path.

### `src/detection.py`

* **Purpose:** Runs single-image YOLO damage detection and applies geometry-based class validation.
* **Active Damage Model Path:** `models/best_carDD.pt`, configured by `app.config.DAMAGE_MODEL_PATH`.
* **Key Thresholds:**
  * `BASE_CONF = 0.45`
  * `TIRE_MIN_Y = 0.75`
  * `GLASS_MAX_Y = 0.55`
  * `MIN_BOX_HEIGHT = 0.03`
  * `MAX_TIRE_AREA = 0.15`
  * `GLASS_MISSING_SINGLE = 0.12`
  * `GLASS_MISSING_TOTAL = 0.18`
  * `MIN_GLASS_CONF = 0.70`
* **Output:** Writes `[input_stem]_damage.jpg` under the configured output image directory.

### `src/tracking.py`

* **Purpose:** Defines `CarState`, the state object for a tracked vehicle.
* **Tracked Fields:** ID, team, damage state, speed history, acceleration history, smoothed speed, collision frames, first/last seen frame, last position, last bounding box, and path length.
* **Motion Logic:** Computes pixel distance per frame, converts it to pixels/second using FPS, stores acceleration, and applies exponential speed smoothing with `alpha = 0.3`.

### `src/pipeline.py`

* **Purpose:** Runs the full video analysis loop.
* **Active Car Model Path:** `models/best_f1_detect.pt`, verified by SHA-256 before loading.
* **Active Damage Model Path:** `models/best_carDD.pt`.
* **Active Team Model Path:** `models/f1_team_classifier.pkl`, verified by hash before `load_learner()`.
* **Core Constants (all env-overridable):**
  * `CONF_TH = 0.4` (`PIPELINE_CONF_TH`)
  * `DAMAGE_CONF_TH = 0.4` (`PIPELINE_DAMAGE_CONF_TH`)
  * `TEAM_CONF_TH = 0.6` (`PIPELINE_TEAM_CONF_TH`)
  * `DAMAGE_EVERY_N_FRAMES = 10` (`DAMAGE_EVERY_N_FRAMES`)
  * `MAX_FRAMES = 3000` (`MAX_FRAMES`)
  * temporary OpenCV codec `mp4v` (`PIPELINE_VIDEO_CODEC`), then published H.264/AAC with ffmpeg
* **Model access:** models are pulled from the lazy cache in `src/model_loader.py` on the first video request, not at import; see "Performance" below.
* **Tracker Config:** `tracker/bytetrack.yaml`, configured by `app.config.TRACKER_CONFIG_PATH`.
* **Progress Callback:** `run_full_pipeline(..., progress_callback=...)` can report processed frame count and total frame count to a caller.
* **Return Summary:** `cars_tracked`, `total_overtakes`, `total_collisions`, `damage_type_counts`, `damage_by_car`, `team_counts`, `frames_processed`, `source_video`, `output_video`, `output_video_name`, and (Phase 5) `processing_seconds`, `processing_fps`, `output_size_mb`.

## Performance (Phase 5)

* **Lazy, cached model loading (`src/model_loader.py`).** The car detector, damage detector, and FastAI team classifier are each built only on first use, behind a thread-safe cache, and reused afterward. Consequences:
  * **Startup is ~7x faster:** `import app.main` dropped from **6.7s -> 1.0s** because no model loads at import time.
  * **Image-only requests never load the video stack.** A `/damage/image` (or `/demo/image`) call loads only `damage_detector`; `car_detector` and `team_classifier` stay unloaded (verify via `/api/info`).
* **Device detection & model-load status** are exposed on `GET /api/info` (`device` + `models` blocks). CUDA is used automatically when available (verified on an RTX 4050; YOLO auto-moves to `cuda:0` on first inference).
* **Batched damage inference.** Per stride frame, all eligible car crops are run through the damage model in a single batched forward pass instead of one YOLO call per car, collapsing N GPU launches into one.
* **Timing logs.** `[MODELS]` lines report per-model load time; `[TIMING]` lines report "models ready" time, and processed frames / fps / output size / codec at the end of each run. `src/detection.py` logs per-image damage-inference latency.
* **Tuned demo defaults.** `DAMAGE_EVERY_N_FRAMES` 5 -> 10 and `MAX_FRAMES` 5000 -> 3000 trade a little damage-tracking granularity for a snappier local demo; both revert via env vars.
* **Output encoding.** OpenCV writes an intermediate `mp4v` file, then ffmpeg publishes H.264/AAC with `yuv420p` pixel format and `+faststart` metadata. This is HTML5-browser compatible, retains source audio when present, and is verified in the pipeline summary as `h264/aac`.

## Component: State & Event Layer (`tracking_memory/`)

### `tracking_memory/damage_state.py`

Tracks damage type counts and first/last seen frames. Severity is `HIGH` if seen for more than 30 frames, `MEDIUM` if seen for more than 10 frames, otherwise `LOW`.

### `tracking_memory/damage_assigner.py`

Assigns a damage detection to a car when the damage box and the car's last bounding box have IoU greater than `0.3`.

### `tracking_memory/collision_detector.py`

Flags a collision when the latest acceleration is below `-300` and a new damage type first appears on the same frame. A 15-frame cooldown prevents duplicate collision events.

### `tracking_memory/overtake_detector.py`

Ranks active cars by `path_length`. A car is marked as overtaking when its rank improves and the 40-frame cooldown has elapsed.

### `tracking_memory/tracker_core.py`

Maintains the global `cars` dictionary and creates/updates `CarState` instances for track IDs emitted by YOLO/ByteTrack.

### `tracking_memory/utils.py`

Provides the bounding-box Intersection over Union helper used by damage assignment.

## Standalone Scripts and Harnesses

| File | Purpose |
| ---- | ------- |
| `damage_video_inference.py` | Standalone accident/damage timeline script that writes `damage_timeline.csv` and `damage_timeline.png`. |
| `fix_yoloDF.py` | Deduplicates and normalizes team image datasets. |
| `splitDF_yolo.py` | Splits preprocessed image datasets into train/validation folders. |
| `tracker/tracking_test.py` | Local OpenCV visualization harness. |
| `tracker/tracking_test2.py` | Headless tracking harness that writes an AVI output. |
| `test1.py` | Basic car model smoke script. |
| `test2.py` | Basic damage model smoke script. |
