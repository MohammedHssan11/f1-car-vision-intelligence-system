# API Reference Documentation

This document reflects the current FastAPI routes in `app/main.py`, `app/api.py`, and the shared settings in `app/config.py`.

## Global API Rules

* **Authentication:** None. The current code exposes all page and API routes without login, API keys, or role checks.
* **CORS:** `app/main.py` reads `ALLOWED_ORIGINS` from `app.config`. By default only `http://localhost:8000` and `http://127.0.0.1:8000` are allowed. Deployments can override this with the `ALLOWED_ORIGINS` environment variable.
* **Allowed Methods:** `GET` and `POST`.
* **Allowed Headers:** All headers are allowed.
* **Upload Naming:** Uploaded files are renamed to UUID-based filenames before saving.
* **Upload Size Limits:** Images are capped at 20 MB; videos are capped at 500 MB.
* **Content Validation:** Uploaded images are verified with Pillow and uploaded videos are opened/read with OpenCV before inference starts.
* **Error UI:** HTML endpoints render friendly error panels instead of exposing raw tracebacks to the browser.

## Static Asset Routes

| Route | Local Directory | Purpose |
| ---- | --------------- | ------- |
| `/static` | `app/static/` | Serves static frontend assets and processed image results under `app/static/results/`. |
| `/outputs` | `outputs/` | Serves processed videos, especially files written under `outputs/videos/`. |

## Endpoint Inventory

### Root Info JSON

* **Route:** `/api`
* **Method:** `GET`
* **Response Type:** `application/json`

```json
{
  "message": "F1 Computer Vision API is running",
  "pages": {
    "home": "/",
    "image_damage_page": "/page/damage",
    "full_pipeline_page": "/page/pipeline"
  }
}
```

### UI Pages

| Route | Method | Template |
| ----- | ------ | -------- |
| `/` | `GET` | `app/templates/index.html` |
| `/page/damage` | `GET` | `app/templates/damage_image.html` |
| `/page/pipeline` | `GET` | `app/templates/full_pipeline.html` |

### Demo Actions

| Route | Method | Purpose |
| ----- | ------ | ------- |
| `/demo/image` | `POST` | Runs the configured sample image and renders `damage_image.html` with a downloadable annotated result. |
| `/demo/video` | `POST` | Runs the configured sample video and renders `pipeline_result.html` with a downloadable output video. |
| `/jobs/demo/video` | `POST` | Queues the configured sample video as a background job and redirects to the job status page. |

Configured demo paths live in `app/config.py`:

* `SAMPLE_IMAGE_PATH = splited_dataset/val/valid_images/ferrari_f1_car_0014.jpg`
* `SAMPLE_VIDEO_PATH = tracker/tracker2.mp4`

### Background Video Jobs

| Route | Method | Purpose |
| ----- | ------ | ------- |
| `/jobs/pipeline/run` | `POST` | Validates a whitelisted local video path, queues a background job, and redirects to `/jobs/{job_id}`. |
| `/jobs/pipeline/video` | `POST` | Validates an uploaded video, queues a background job, and redirects to `/jobs/{job_id}`. |
| `/jobs/demo/video` | `POST` | Queues the configured sample video and redirects to `/jobs/{job_id}`. |
| `/jobs/{job_id}` | `GET` | Renders the polling job status page. |
| `/api/jobs/{job_id}` | `GET` | Returns JSON status for queued/running/completed/failed jobs. |
| `/jobs/{job_id}/result` | `GET` | Renders `pipeline_result.html` after completion, or the status page while running/failed. |

Job status JSON fields include:

* `job_id`
* `status`: `queued`, `running`, `completed`, or `failed`
* `progress`
* `frames_processed`
* `total_frames`
* `source_label`
* `output_name`
* `output_video_url`
* `duplicate_of`
* `message`
* `error`
* `summary`

The in-memory job store is process-local. Restarting the FastAPI process clears queued/completed job metadata, but completed video files remain under `outputs/videos/` until cleanup.

### Chassis Visual Scanner

* **Route:** `/damage/image`
* **Method:** `POST`
* **Content Type:** `multipart/form-data`
* **Field:** `file`
* **Allowed Extensions:** `.jpg`, `.jpeg`, `.png`, `.bmp`
* **Maximum Size:** 20 MB

Internal flow:

1. `_save_validated_upload()` validates the file extension and streams the file to disk in chunks.
2. The input is saved under `uploads/images/[uuid][suffix]`.
3. `_validate_image_content()` verifies the image bytes, format, and dimensions.
4. `services.damage_image.detect_damage_image()` calls the current `src/detection.py` logic.
5. The annotated result is written under `app/static/results/`.
6. The same `damage_image.html` template is rendered with `result_image=/static/results/[name]_damage.jpg` and a download link.

Error behavior:

* `400 Bad Request` for unsupported file extensions.
* `400 Bad Request` for unreadable or mismatched image content.
* `413 Payload Too Large` when the upload exceeds 20 MB.
* `500 Internal Server Error` if OpenCV/model inference fails after validation.

### Process Uploaded Video Stream

* **Route:** `/pipeline/video`
* **Method:** `POST`
* **Content Type:** `multipart/form-data`
* **Field:** `file`
* **Allowed Extensions:** `.mp4`, `.avi`, `.mov`, `.mkv`
* **Maximum Size:** 500 MB

Internal flow:

1. `_save_validated_upload()` validates the file extension and streams the video to disk in chunks.
2. The input is saved under `uploads/videos/[uuid][suffix]`.
3. `_validate_video_content()` confirms OpenCV can open the video and read at least one frame.
4. The output name is built as `out_[uuid].mp4`.
5. `services.full_pipeline.run_full_pipeline()` calls `src/pipeline.py`.
6. The response renders `pipeline_result.html` with `video_path=/outputs/videos/out_[uuid].mp4`, a download link, and a summary dictionary.

The default UI now uses `/jobs/pipeline/video` for uploads so the browser receives a job page immediately. `/pipeline/video` remains available as a direct synchronous compatibility endpoint.

Error behavior:

* `400 Bad Request` for unsupported file extensions.
* `400 Bad Request` for unreadable video content.
* `413 Payload Too Large` when the upload exceeds 500 MB.
* `500 Internal Server Error` for model, video codec, tracker config, or missing-weight failures.

### Process Server-Local Video Path

* **Route:** `/pipeline/run`
* **Method:** `POST`
* **Content Type:** `application/x-www-form-urlencoded`
* **Field:** `video_path`

Current code behavior:

1. `video_path` is treated as a relative path under whitelisted local roots: `tracker/` and `uploads/videos/`.
2. Absolute paths and paths escaping those directories through `..` are rejected.
3. The resolved file is validated with OpenCV before the pipeline starts.
4. Missing files return `404 Video not found`.
5. Successful runs write `outputs/videos/result.mp4`.
6. The response renders `pipeline_result.html` with summary cards, damage type counts, and output links.

The default UI value is `tracker/tracker2.mp4`, which is allowed because `tracker/` is one of the whitelisted local video roots.

The default UI now uses `/jobs/pipeline/run` for local paths so the browser receives a job page immediately. `/pipeline/run` remains available as a direct synchronous compatibility endpoint.

### Pipeline Summary Fields

Successful video pipeline routes return an HTML report backed by a summary dictionary with these keys:

* `cars_tracked`
* `total_overtakes`
* `total_collisions`
* `damage_type_counts`
* `damage_by_car`
* `team_counts`
* `frames_processed`
* `source_video`
* `output_video`
* `output_video_name`

### Health Status

* **Route:** `/api/health`
* **Method:** `GET`
* **Response Type:** `application/json`

```json
{
  "status": "ok",
  "service": "F1 Damage & Tracking System"
}
```

### Feature Catalog + Runtime Info

* **Route:** `/api/info`
* **Method:** `GET`
* **Response Type:** `application/json`

Reports the feature catalog plus live runtime device and model-load state.
The `device` block comes from torch (CUDA is used automatically when
available); the `models` block reflects the lazy cache in
`src/model_loader.py` — each model shows `loaded: false` until the first
request that actually needs it, at which point `loaded` flips to `true` and
`load_seconds` records how long it took. Hitting `/api/info` itself does
**not** load any model.

```json
{
  "message": "F1 Computer Vision API",
  "features": [
    "Car Detection",
    "Tracking",
    "Speed Estimation",
    "Damage Detection",
    "Collision Detection",
    "Overtake Detection",
    "Background Video Jobs"
  ],
  "device": {
    "torch_version": "2.6.0+cu124",
    "cuda_available": true,
    "device": "cuda",
    "cuda_device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "cuda_device_count": 1
  },
  "models": {
    "car_detector": {"path": "...best.pt", "loaded": false, "load_seconds": null},
    "damage_detector": {"path": "...best_carDD.pt", "loaded": true, "load_seconds": 2.3},
    "team_classifier": {"path": "...f1_team_classifier.pkl", "loaded": false, "load_seconds": null}
  }
}
```

## Error Codes and Behaviors

| Status | Source | Meaning |
| ------ | ------ | ------- |
| `400` | `app/api.py` | Unsupported upload extension, absolute path, or path traversal attempt. |
| `400` | `app/api.py` | Uploaded image/video content is unreadable or does not match the expected media type. |
| `404` | `app/api.py` | Server-local `video_path` is allowed but does not resolve to an existing file. |
| `413` | `app/api.py` | Upload exceeds the configured image or video size limit. |
| `422` | FastAPI | Required form fields are missing. |
| `500` | Runtime/model layer | OpenCV, YOLO, FastAI, missing model path, missing tracker config, or video writer failures. |
