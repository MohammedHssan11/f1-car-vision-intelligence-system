# Consolidated Documentation Index

This file intentionally stays short so it does not drift from the source docs. Use the individual files in this folder as the canonical documentation.

## Documentation Files

* `README.md`
* `ARCHITECTURE.md`
* `API_REFERENCE.md`
* `BACKEND.md`
* `AI_SYSTEM.md`
* `DATABASE.md`
* `DEPLOYMENT.md`
* `FRONTEND.md`
* `MAINTENANCE_GUIDE.md`
* `SECURITY_AUDIT.md`

## Current Codebase Alignment Notes

* Model and tracker paths are centralized in `app/config.py`.
* Damage inference uses `models/best_carDD.pt`.
* Video tracking uses `tracker/bytetrack.yaml`.
* `/pipeline/run` accepts whitelisted files from `tracker/` and `uploads/videos/`.
* Pipeline outputs are generated as `.mp4` and templates use the web-safe `video_path`.
* Remaining maintenance note: `src/classification.py` exports to root `f1_team_classifier.pkl`, while runtime loads `models/f1_team_classifier.pkl`.
