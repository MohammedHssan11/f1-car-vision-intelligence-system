from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
import uuid

from services.full_pipeline import run_full_pipeline


FINAL_STATUSES = {"completed", "failed"}


class VideoJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-job")

    def submit(self, video_path: Path, source_label: str, output_stem: str) -> dict:
        job_id = uuid.uuid4().hex
        now = _utc_now()
        safe_stem = _safe_output_stem(output_stem)
        output_name = f"{safe_stem}_{job_id[:8]}.mp4"
        resolved_video_path = str(Path(video_path).resolve())

        with self._lock:
            duplicate_of = self._find_active_duplicate(resolved_video_path)
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "message": "Queued for processing",
                "progress": 0,
                "frames_processed": 0,
                "total_frames": None,
                "source_label": source_label,
                "source_video": resolved_video_path,
                "output_name": output_name,
                "output_video_url": f"/outputs/videos/{output_name}",
                "status_url": f"/api/jobs/{job_id}",
                "page_url": f"/jobs/{job_id}",
                "result_url": f"/jobs/{job_id}/result",
                "duplicate_of": duplicate_of,
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "error": None,
                "summary": None,
            }

        self._executor.submit(self._run, job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job is not None else None

    def _find_active_duplicate(self, resolved_video_path: str) -> str | None:
        for job in self._jobs.values():
            if (
                job["source_video"] == resolved_video_path
                and job["status"] not in FINAL_STATUSES
            ):
                return job["job_id"]
        return None

    def _update(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(updates)

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return

        self._update(
            job_id,
            status="running",
            message="Processing video",
            started_at=_utc_now(),
        )

        def progress_callback(frames_processed: int, total_frames: int | None) -> None:
            progress = _progress_percent(frames_processed, total_frames)
            self._update(
                job_id,
                progress=progress,
                frames_processed=frames_processed,
                total_frames=total_frames,
                message=f"Processing frame {frames_processed}",
            )

        try:
            summary = run_full_pipeline(
                video_path=Path(job["source_video"]),
                output_name=job["output_name"],
                progress_callback=progress_callback,
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                message="Video processing failed",
                error=str(exc),
                completed_at=_utc_now(),
            )
            return

        self._update(
            job_id,
            status="completed",
            message="Processing complete",
            progress=100,
            frames_processed=summary.get("frames_processed", job.get("frames_processed", 0)),
            total_frames=summary.get("frames_processed", job.get("total_frames")),
            summary=summary,
            completed_at=_utc_now(),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_output_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return stem.strip("_") or "video_job"


def _progress_percent(frames_processed: int, total_frames: int | None) -> int:
    if not total_frames:
        return 0
    return max(0, min(99, int((frames_processed / total_frames) * 100)))


video_jobs = VideoJobStore()
