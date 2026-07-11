from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    UploadFile,
    File,
    Form
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uuid
import cv2
from PIL import Image, UnidentifiedImageError

# =============================
# SERVICES
# =============================
from services.damage_image import detect_damage_image
from services.full_pipeline import run_full_pipeline
from app.jobs import video_jobs

from app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    BASE_DIR,
    DEFAULT_LOCAL_VIDEO_PATH,
    LOCAL_VIDEO_ROOTS,
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    OUTPUT_IMAGE_DIR,
    OUTPUT_VIDEO_DIR,
    SAMPLE_IMAGE_PATH,
    SAMPLE_VIDEO_PATH,
    UPLOAD_IMAGE_DIR,
    UPLOAD_VIDEO_DIR,
)

# =============================
# ROUTER
# =============================
router = APIRouter()

# =============================
# TEMPLATES
# =============================
templates = Jinja2Templates(directory="app/templates")

IMAGE_FORMATS_BY_EXTENSION = {
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".png": {"PNG"},
    ".bmp": {"BMP"},
}


# =============================
# UPLOAD / PATH SAFETY HELPERS
# =============================
async def _save_validated_upload(
    file: UploadFile,
    dest_dir: Path,
    allowed_extensions: set[str],
    max_size_bytes: int,
) -> Path:
    """
    Save an uploaded file after validating its extension and enforcing a
    size cap, streaming to disk in chunks so an oversized upload can't
    exhaust memory or disk before being rejected.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_extensions)}",
        )

    dest_path = dest_dir / f"{uuid.uuid4().hex}{ext}"
    written = 0
    chunk_size = 1024 * 1024

    with open(dest_path, "wb") as out_f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_size_bytes:
                out_f.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {max_size_bytes} bytes",
                )
            out_f.write(chunk)

    if written == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return dest_path


def _http_detail(exc: HTTPException) -> str:
    if isinstance(exc.detail, str):
        return exc.detail
    return str(exc.detail)


def _remove_file(path: Path | None) -> None:
    if path is not None:
        Path(path).unlink(missing_ok=True)


def _static_result_url(output_name: str) -> str:
    return f"/static/results/{output_name}"


def _output_video_url(output_name: str) -> str:
    return f"/outputs/videos/{output_name}"


def _relative_label(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return Path(path).name


def _render_error(
    request: Request,
    template_name: str,
    message: str,
    status_code: int,
    **context,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        template_name,
        {
            "error_message": message,
            **context,
        },
        status_code=status_code,
    )


def _render_damage_result(
    request: Request,
    output_name: str,
    source_label: str,
) -> HTMLResponse:
    result_url = _static_result_url(output_name)
    return templates.TemplateResponse(
        request,
        "damage_image.html",
        {
            "result_image": result_url,
            "download_url": result_url,
            "source_label": source_label,
        },
    )


def _render_pipeline_result(
    request: Request,
    video_path: Path,
    output_name: str,
    source_label: str,
) -> HTMLResponse:
    summary = run_full_pipeline(
        video_path=video_path,
        output_name=output_name,
    )
    video_url = _output_video_url(output_name)
    return templates.TemplateResponse(
        request,
        "pipeline_result.html",
        {
            "summary": summary,
            "video_path": video_url,
            "video_download_url": video_url,
            "output_video_name": output_name,
            "source_label": source_label,
        },
    )


def _render_job_status(request: Request, job: dict, status_code: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "job_status.html",
        {"job": job},
        status_code=status_code,
    )


def _redirect_to_job(job: dict) -> RedirectResponse:
    return RedirectResponse(url=job["page_url"], status_code=303)


def _queue_video_job(video_path: Path, source_label: str, output_stem: str) -> dict:
    return video_jobs.submit(
        video_path=video_path,
        source_label=source_label,
        output_stem=output_stem,
    )


def _validate_image_content(path: Path) -> None:
    try:
        with Image.open(path) as img:
            image_format = img.format
            width, height = img.size
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image") from exc

    expected_formats = IMAGE_FORMATS_BY_EXTENSION.get(path.suffix.lower())
    if expected_formats and image_format not in expected_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Image content is {image_format or 'unknown'}, not {sorted(expected_formats)}",
        )

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Uploaded image has invalid dimensions")


def _validate_video_content(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Video file is not readable")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()

        if width <= 0 or height <= 0 or not ok:
            raise HTTPException(status_code=400, detail="Video file has no readable frames")
    finally:
        cap.release()


def _is_under_root(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _resolve_safe_video_path(user_supplied: str) -> Path:
    """
    Resolve a relative video path against whitelisted local roots only.
    Bundled demo clips under tracker/ and uploads under uploads/videos/ are
    allowed; arbitrary filesystem paths are rejected.
    """
    requested = Path(user_supplied or "")
    if not user_supplied or requested.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid video path")

    allowed_roots = [Path(root).resolve() for root in LOCAL_VIDEO_ROOTS]
    candidates = [
        (BASE_DIR / requested).resolve(),
        (UPLOAD_VIDEO_DIR / requested).resolve(),
    ]

    path_was_allowed = False
    for candidate in candidates:
        if not any(_is_under_root(candidate, root) for root in allowed_roots):
            continue

        path_was_allowed = True

        if candidate.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported video type")

        if candidate.is_file():
            return candidate

    if path_was_allowed:
        raise HTTPException(status_code=404, detail="Video not found")

    raise HTTPException(status_code=400, detail="Invalid video path")

# =============================
# PAGES (HTML)
# =============================

@router.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
    )


@router.get("/page/damage", response_class=HTMLResponse)
def damage_page(request: Request):
    return templates.TemplateResponse(
        request,
        "damage_image.html",
    )


@router.get("/page/pipeline", response_class=HTMLResponse)
def pipeline_page(request: Request):
    return templates.TemplateResponse(
        request,
        "full_pipeline.html",
        {
            "default_video_path": DEFAULT_LOCAL_VIDEO_PATH,
            "sample_video_path": _relative_label(SAMPLE_VIDEO_PATH),
        }
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_status_page(request: Request, job_id: str):
    job = video_jobs.get(job_id)
    if job is None:
        return _render_error(
            request,
            "full_pipeline.html",
            "Job not found",
            404,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )
    status_code = 500 if job["status"] == "failed" else 200
    return _render_job_status(request, job, status_code=status_code)


@router.get("/jobs/{job_id}/result", response_class=HTMLResponse)
def job_result_page(request: Request, job_id: str):
    job = video_jobs.get(job_id)
    if job is None:
        return _render_error(
            request,
            "full_pipeline.html",
            "Job not found",
            404,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )

    if job["status"] != "completed":
        status_code = 500 if job["status"] == "failed" else 202
        return _render_job_status(request, job, status_code=status_code)

    return templates.TemplateResponse(
        request,
        "pipeline_result.html",
        {
            "summary": job["summary"],
            "video_path": job["output_video_url"],
            "video_download_url": job["output_video_url"],
            "output_video_name": job["output_name"],
            "source_label": job["source_label"],
        },
    )

# =============================
# IMAGE DAMAGE DETECTION
# =============================

@router.post("/damage/image", response_class=HTMLResponse)
async def damage_image(
    request: Request,
    file: UploadFile = File(...)
):
    input_path = None
    try:
        input_path = await _save_validated_upload(
            file, UPLOAD_IMAGE_DIR, ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_SIZE_BYTES
        )
        _validate_image_content(input_path)

        output_name = detect_damage_image(input_path, OUTPUT_IMAGE_DIR)

        return _render_damage_result(request, output_name, file.filename or input_path.name)
    except HTTPException as exc:
        _remove_file(input_path)
        return _render_error(request, "damage_image.html", _http_detail(exc), exc.status_code)
    except Exception as exc:
        return _render_error(
            request,
            "damage_image.html",
            f"Image processing failed: {exc}",
            500,
        )

# =============================
# DEMO SAMPLE IMAGE
# =============================

@router.post("/demo/image", response_class=HTMLResponse)
def demo_image(request: Request):
    try:
        if not SAMPLE_IMAGE_PATH.is_file():
            raise HTTPException(status_code=404, detail="Sample image not found")

        _validate_image_content(SAMPLE_IMAGE_PATH)
        output_name = detect_damage_image(SAMPLE_IMAGE_PATH, OUTPUT_IMAGE_DIR)

        return _render_damage_result(
            request,
            output_name,
            _relative_label(SAMPLE_IMAGE_PATH),
        )
    except HTTPException as exc:
        return _render_error(request, "damage_image.html", _http_detail(exc), exc.status_code)
    except Exception as exc:
        return _render_error(
            request,
            "damage_image.html",
            f"Sample image processing failed: {exc}",
            500,
        )

# =============================
# BACKGROUND VIDEO JOBS
# =============================

@router.post("/jobs/pipeline/video", response_class=HTMLResponse)
async def queue_pipeline_video(
    request: Request,
    file: UploadFile = File(...)
):
    input_path = None
    try:
        input_path = await _save_validated_upload(
            file, UPLOAD_VIDEO_DIR, ALLOWED_VIDEO_EXTENSIONS, MAX_VIDEO_SIZE_BYTES
        )
        _validate_video_content(input_path)
        job = _queue_video_job(
            video_path=input_path,
            source_label=file.filename or input_path.name,
            output_stem=f"job_upload_{input_path.stem}",
        )
        return _redirect_to_job(job)
    except HTTPException as exc:
        _remove_file(input_path)
        return _render_error(
            request,
            "full_pipeline.html",
            _http_detail(exc),
            exc.status_code,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )
    except Exception as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            f"Video job submission failed: {exc}",
            500,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )


@router.post("/jobs/pipeline/run", response_class=HTMLResponse)
def queue_pipeline_path(
    request: Request,
    video_path: str = Form(...)
):
    try:
        safe_path = _resolve_safe_video_path(video_path)
        _validate_video_content(safe_path)
        job = _queue_video_job(
            video_path=safe_path,
            source_label=_relative_label(safe_path),
            output_stem="job_result",
        )
        return _redirect_to_job(job)
    except HTTPException as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            _http_detail(exc),
            exc.status_code,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )
    except Exception as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            f"Video job submission failed: {exc}",
            500,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )


@router.post("/jobs/demo/video", response_class=HTMLResponse)
def queue_demo_video(request: Request):
    try:
        if not SAMPLE_VIDEO_PATH.is_file():
            raise HTTPException(status_code=404, detail="Sample video not found")

        _validate_video_content(SAMPLE_VIDEO_PATH)
        job = _queue_video_job(
            video_path=SAMPLE_VIDEO_PATH,
            source_label=_relative_label(SAMPLE_VIDEO_PATH),
            output_stem="job_demo_tracker2",
        )
        return _redirect_to_job(job)
    except HTTPException as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            _http_detail(exc),
            exc.status_code,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )
    except Exception as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            f"Sample video job submission failed: {exc}",
            500,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )

# =============================
# FULL PIPELINE (UPLOAD VIDEO)
# =============================

@router.post("/pipeline/video", response_class=HTMLResponse)
async def pipeline_video(
    request: Request,
    file: UploadFile = File(...)
):
    input_path = None
    try:
        input_path = await _save_validated_upload(
            file, UPLOAD_VIDEO_DIR, ALLOWED_VIDEO_EXTENSIONS, MAX_VIDEO_SIZE_BYTES
        )
        _validate_video_content(input_path)
        output_name = f"out_{input_path.stem}.mp4"

        # =============================
        # RUN PIPELINE
        # =============================
        return _render_pipeline_result(
            request,
            input_path,
            output_name,
            file.filename or input_path.name,
        )
    except HTTPException as exc:
        _remove_file(input_path)
        return _render_error(
            request,
            "full_pipeline.html",
            _http_detail(exc),
            exc.status_code,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )
    except Exception as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            f"Video processing failed: {exc}",
            500,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )

# =============================
# FULL PIPELINE (DEMO SAMPLE VIDEO)
# =============================

@router.post("/demo/video", response_class=HTMLResponse)
def demo_video(request: Request):
    try:
        if not SAMPLE_VIDEO_PATH.is_file():
            raise HTTPException(status_code=404, detail="Sample video not found")

        _validate_video_content(SAMPLE_VIDEO_PATH)
        return _render_pipeline_result(
            request,
            SAMPLE_VIDEO_PATH,
            "demo_tracker2.mp4",
            _relative_label(SAMPLE_VIDEO_PATH),
        )
    except HTTPException as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            _http_detail(exc),
            exc.status_code,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )
    except Exception as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            f"Sample video processing failed: {exc}",
            500,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )

# =============================
# FULL PIPELINE (VIDEO PATH FORM)
# =============================

@router.post("/pipeline/run", response_class=HTMLResponse)
def run_pipeline_page(
    request: Request,
    video_path: str = Form(...)
):
    """
    Run pipeline using a video already present under a whitelisted local root.

    `video_path` is resolved against tracker/ and uploads/videos/ and
    validated so it cannot escape those directories (no absolute paths,
    no '..' traversal) — see _resolve_safe_upload_path.
    """

    try:
        safe_path = _resolve_safe_video_path(video_path)
        _validate_video_content(safe_path)
        output_name = "result.mp4"

        return _render_pipeline_result(
            request,
            safe_path,
            output_name,
            _relative_label(safe_path),
        )
    except HTTPException as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            _http_detail(exc),
            exc.status_code,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )
    except Exception as exc:
        return _render_error(
            request,
            "full_pipeline.html",
            f"Video processing failed: {exc}",
            500,
            default_video_path=DEFAULT_LOCAL_VIDEO_PATH,
            sample_video_path=_relative_label(SAMPLE_VIDEO_PATH),
        )

# =============================
# API STATUS / HEALTH
# =============================

@router.get("/api/jobs/{job_id}", response_class=JSONResponse)
def api_job_status(job_id: str):
    job = video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/api/health", response_class=JSONResponse)
def health_check():
    return {
        "status": "ok",
        "service": "F1 Damage & Tracking System"
    }


@router.get("/api/info", response_class=JSONResponse)
def api_info():
    # Imported lazily so `import app.api` (and the module-import cost of the
    # whole app) never triggers loading torch/ultralytics unless /api/info is
    # actually hit.
    from src.model_loader import get_device_info, get_model_status

    return {
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
        "device": get_device_info(),
        "models": get_model_status(),
    }
