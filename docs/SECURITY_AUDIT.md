# Forensic Security Audit Report

This audit reflects the current code in `app/main.py`, `app/api.py`, `app/config.py`, `src/detection.py`, and `src/pipeline.py`.

## Current Security Summary

| Severity | Status | Issue | Evidence | Impact |
| -------- | ------ | ----- | -------- | ------ |
| **HIGH** | **Partially mitigated** | Pickle deserialization risk for FastAI team model | `src/pipeline.py` calls `verify_model_integrity(team_model_path)` before `load_learner(team_model_path)` | Tampered `models/f1_team_classifier.pkl` is rejected when the pinned hash does not match. |
| **HIGH** | **Mitigated in backend** | Path traversal via `/pipeline/run` | `app/api.py` uses `_resolve_safe_upload_path(video_path, UPLOAD_VIDEO_DIR)` | Absolute paths and `..` escapes are rejected before OpenCV receives the path. |
| **MEDIUM** | **Mitigated for local demo** | Unsafe uploads | `app/api.py` uses `_save_validated_upload()`, `_validate_image_content()`, and `_validate_video_content()` | Extension, size, and readable media content are checked before inference. |
| **MEDIUM** | **Partially mitigated** | Hardcoded absolute model/dataset paths | Runtime paths are centralized in `app/config.py`; `src/classification.py` still uses local training/export paths | Runtime portability is improved, but retraining remains tied to this workstation layout. |
| **MEDIUM** | **Open** | YOLO model integrity is not pinned | `src/pipeline.py` and `src/detection.py` load `.pt` files without checksum verification | Replaced model files may alter inference behavior; PyTorch model loading also deserves trust controls. |
| **LOW** | **Open** | Missing authentication / authorization | `app/main.py`, `app/api.py` | Anyone with network access to the service can use upload and inference endpoints. |
| **LOW** | **Improved** | Open CORS | `app/config.py` defines `ALLOWED_ORIGINS` | CORS is no longer wildcard by default, but deployment must set origins intentionally. |

## Detailed Findings

### 1. FastAI Pickle Deserialization

* **Status:** Partially mitigated.
* **Location:** `src/pipeline.py`, `app/config.py`
* **Current behavior:** `src/pipeline.py` verifies `models/f1_team_classifier.pkl` with `verify_model_integrity()` before calling FastAI `load_learner()`.
* **Verified pinned hash:** `d6594c0c1d5c7804ed65e20ce228eb07dbdf6b2829a244105d08577a53b244af`
* **Residual risk:** The application still uses a pickle-based FastAI export, so safe operation depends on keeping the pinned hash current and only updating it after manual model verification.

### 2. Server-Local Video Path Traversal

* **Status:** Mitigated in backend.
* **Location:** `app/api.py`
* **Current behavior:** `/pipeline/run` rejects empty values, absolute paths, and any relative path that resolves outside the whitelisted local roots: `tracker/` and `uploads/videos/`.
* **Current default:** `app/templates/full_pipeline.html` defaults to `tracker/tracker2.mp4`, which is now allowed by the whitelist.

### 3. Upload Handling

* **Status:** Partially mitigated.
* **Location:** `app/api.py`, `app/config.py`
* **Current behavior:** Uploads are streamed to disk in 1 MB chunks, renamed with UUIDs, checked against extension allow-lists, and rejected if they exceed configured byte limits.
* **Image policy:** `.jpg`, `.jpeg`, `.png`, `.bmp`; maximum 20 MB.
* **Video policy:** `.mp4`, `.avi`, `.mov`, `.mkv`; maximum 500 MB.
* **Remaining gap:** The app does not rely on MIME headers as an authority and does not do deep forensic file parsing; however, invalid image/video bytes are rejected before model inference.

### 4. Hardcoded Absolute Paths

* **Status:** Partially mitigated.
* **Locations:**
  * `src/classification.py`: training dataset and export paths still use absolute `C:\TERM 7\...` paths.
* **Current workspace evidence:** Runtime model, tracker, upload, output, and demo sample paths are centralized in `app/config.py`; the remaining hardcoded paths are in the training script.
* **Recommendation:** Move training dataset/export paths into configuration or CLI arguments before retraining.

### 5. Tracker Config Path Mismatch

* **Status:** Mitigated.
* **Location:** `src/pipeline.py`
* **Current behavior:** `car_model.track()` uses `app.config.TRACKER_CONFIG_PATH`, which points to `tracker/bytetrack.yaml`.
* **Current workspace evidence:** The configured tracker file exists.
* **Impact:** The previous tracker path mismatch is resolved.

### 6. Missing Authentication

* **Status:** Open.
* **Location:** `app/main.py`, `app/api.py`
* **Current behavior:** No authentication middleware, route guards, API keys, user accounts, or authorization checks are present.
* **Recommendation:** Keep the app bound to localhost for demos. Add API key or session/JWT authentication before exposing it on a shared network.

### 7. CORS Configuration

* **Status:** Improved.
* **Location:** `app/config.py`, `app/main.py`
* **Current behavior:** Default allowed origins are `http://localhost:8000` and `http://127.0.0.1:8000`; deployments can set `ALLOWED_ORIGINS`.
* **Recommendation:** In deployment, set `ALLOWED_ORIGINS` to exact frontend origins only.
