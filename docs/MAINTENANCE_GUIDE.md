# System Maintenance and Tuning Guide

This guide covers model retraining, threshold tuning, and cleanup tasks for the current codebase.

## Classification Model Retraining Pipeline

`src/classification.py` trains a FastAI ResNet34 classifier from a folder dataset.

Current code behavior:

* Pass the dataset folder with `--dataset-dir`, or set `TRAINING_DATASET_DIR`; no workstation-specific path is assumed.
* Corrupt images are reported and training stops without deleting source data.
* `ImageDataLoaders.from_folder()` uses `valid_pct=0.2`, `seed=42`, `Resize(224)`, and default augmentations.
* The model is fine-tuned for 5 epochs.
* The default export is the runtime path `models/f1_team_classifier.pkl`, written atomically.

Important maintenance note:

* Run `py -3.11 src\classification.py --dataset-dir "C:\path\to\team-images"`.
* The command prints the new SHA-256. Review the exported model and update its `TRUSTED_MODEL_HASHES` entry in `app/config.py` before starting the service; the app intentionally refuses an unpinned model.

## Dataset Preprocessing Scripts

### `fix_yoloDF.py`

Deduplicates images by MD5, normalizes filenames by team folder name, and writes consolidated images to the configured output folder.

Run:

```powershell
python fix_yoloDF.py
```

### `splitDF_yolo.py`

Splits the preprocessed dataset into train/validation subsets.

Run:

```powershell
python splitDF_yolo.py
```

## Core Pipeline Thresholds

These constants live in `src/pipeline.py` and every one can be overridden
with an environment variable of the same name (no code edit needed) — useful
for tuning a demo without touching source:

| Constant / Env var | Default | Purpose |
| -------- | ------------- | ------- |
| `PIPELINE_CONF_TH` | `0.4` | Car detection confidence threshold. |
| `PIPELINE_DAMAGE_CONF_TH` | `0.4` | Damage detection confidence threshold during video processing. |
| `PIPELINE_TEAM_CONF_TH` | `0.6` | Minimum FastAI team confidence before accepting a team label. |
| `DAMAGE_EVERY_N_FRAMES` | `10` | Stride for running (batched) damage inference on tracked car crops. Raised from 5 -> 10 for local demo speed; lower it for finer damage tracking at the cost of runtime. |
| `MAX_FRAMES` | `3000` | Safety cap for processed frames per video. Lowered from 5000 to bound worst-case runtime on long uploads. |
| `PIPELINE_VIDEO_CODEC` | `mp4v` | FourCC for the temporary OpenCV file. The published website video is always H.264/AAC via ffmpeg. |
| `PIPELINE_H264_CRF` | `23` | Published-video quality: lower is higher quality/larger; 23 is balanced. |
| `PIPELINE_SHOW_SPEED` | `false` | Enables per-car pixel-speed text. Leave off for readable, uncluttered video. |
| `TRACK_REASSOCIATION_MAX_GAP` | `30` | Largest short tracker dropout eligible for conservative stable-ID recovery. |
| `TRACK_REASSOCIATION_MAX_SCORE` | `1.4` | Maximum motion/overlap/scale match score; lower values are stricter. |
| `COLLISION_DECEL_THRESHOLD` | `-1500` | Candidate-impact threshold in px/s² after smoothing; calibrate only with labelled video. |
| `COLLISION_MIN_DAMAGE_OBSERVATIONS` | `2` | Separate damage inference observations required before an impact event. |

See `docs/BACKEND.md` -> "Performance" for the lazy model cache, batched
damage inference, and timing logs added in Phase 5.

Update these constants in `src/detection.py` for single-image damage inference:

| Constant | Current Value | Purpose |
| -------- | ------------- | ------- |
| `BASE_CONF` | `0.45` | Base YOLO confidence for image damage detection. |
| `TIRE_MIN_Y` | `0.75` | Tire detections must be low in the image. |
| `GLASS_MAX_Y` | `0.55` | Low glass detections are treated as scratches. |
| `MIN_GLASS_CONF` | `0.70` | Low-confidence glass detections become scratches. |
| `GLASS_MISSING_SINGLE` | `0.12` | Single-box glass-missing area threshold. |
| `GLASS_MISSING_TOTAL` | `0.18` | Aggregate glass-missing area threshold. |

## Model Path Maintenance

Current active paths:

* Car model: `models/best_f1_detect.pt`
* Damage model: `models/best_carDD.pt`
* Team model: `models/f1_team_classifier.pkl`

Current recommended cleanup:

* Keep runtime model paths in `app/config.py`.
* Pin a SHA-256 for every replacement runtime model before deployment; the car detector, damage detector, and team classifier are all checked before loading.
* Refresh `DAMAGE_MODEL_PATH` if a new damage detector replaces `models/best_carDD.pt`.
* Keep `TRACKER_CONFIG_PATH` aligned with `tracker/bytetrack.yaml`.

## Smoke Tests

Run the Phase 1 smoke tests with Python 3.11:

```powershell
py -3.11 tests\test_reliability.py
py -3.11 tests\smoke_phase1.py
```

The smoke test covers:

* `/api/health`
* invalid image-content rejection
* invalid video-content rejection
* image upload and annotated-result rendering
* local sample-video pipeline run

Run the Phase 2 demo-readiness smoke tests with Python 3.11:

```powershell
py -3.11 tests\smoke_phase2.py
```

The Phase 2 smoke test covers:

* home page sample-action controls
* `/demo/image` sample image processing and image download link rendering
* `/demo/video` sample video processing, debrief rendering, and output video download link rendering

Run the Phase 3 background-job smoke tests with Python 3.11:

```powershell
py -3.11 tests\smoke_phase3.py
```

The Phase 3 smoke test covers:

* queued sample-video job submission
* `/api/jobs/{job_id}` polling until completion
* `/jobs/{job_id}/result` debrief rendering after completion
* friendly bad local-path submission behavior
* failed background job status rendering

Run the Phase 4 benchmark harness with Python 3.11:

```powershell
py -3.11 benchmarks\run_benchmarks.py
py -3.11 tests\smoke_phase4.py
```

The benchmark runner writes:

* `benchmarks/benchmark_results.md`
* `benchmarks/results/benchmark_results.json`
* annotated image outputs under `benchmarks/results/images/`
* benchmark video outputs under `outputs/videos/`

The current manifest only has team labels for image samples. Car boxes, damage labels, and video event labels are not available, so those observations are intentionally not scored as accuracy.

## Directory Cleanup Routine

Uploaded and generated media persists until removed. To clear generated files while preserving directories, run these commands carefully from the project root:

```powershell
py -3.11 cleanup_outputs.py
py -3.11 cleanup_outputs.py --yes
```

The first command is a dry run. The second command deletes generated files from `uploads/images/`, `uploads/videos/`, `app/static/results/`, and `outputs/videos/` while preserving the parent directories.
