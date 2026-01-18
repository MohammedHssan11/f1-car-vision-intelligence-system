from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    File,
    Form
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import uuid

# =============================
# SERVICES
# =============================
from services.damage_image import detect_damage_image
from services.full_pipeline import run_full_pipeline

# =============================
# ROUTER
# =============================
router = APIRouter()

# =============================
# TEMPLATES
# =============================
templates = Jinja2Templates(directory="app/templates")

# =============================
# DIRECTORIES
# =============================
UPLOAD_IMAGE_DIR = Path("uploads/images")
UPLOAD_VIDEO_DIR = Path("uploads/videos")

OUTPUT_VIDEO_DIR = Path("outputs/videos")      # ✅ مهم
OUTPUT_IMAGE_DIR = Path("app/static/results")

UPLOAD_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# PAGES (HTML)
# =============================

@router.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@router.get("/page/damage", response_class=HTMLResponse)
def damage_page(request: Request):
    return templates.TemplateResponse(
        "damage_image.html",
        {"request": request}
    )


@router.get("/page/pipeline", response_class=HTMLResponse)
def pipeline_page(request: Request):
    return templates.TemplateResponse(
        "full_pipeline.html",
        {"request": request}
    )

# =============================
# IMAGE DAMAGE DETECTION
# =============================

@router.post("/damage/image", response_class=HTMLResponse)
async def damage_image(
    request: Request,
    file: UploadFile = File(...)
):
    ext = Path(file.filename).suffix
    filename = f"{uuid.uuid4().hex}{ext}"
    input_path = UPLOAD_IMAGE_DIR / filename

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_name = detect_damage_image(input_path, OUTPUT_IMAGE_DIR)

    return templates.TemplateResponse(
        "damage_image.html",
        {
            "request": request,
            "result_image": f"/static/results/{output_name}"
        }
    )

# =============================
# FULL PIPELINE (UPLOAD VIDEO)
# =============================

@router.post("/pipeline/video", response_class=HTMLResponse)
async def pipeline_video(
    request: Request,
    file: UploadFile = File(...)
):
    ext = Path(file.filename).suffix
    input_name = f"{uuid.uuid4().hex}{ext}"
    output_name = f"out_{input_name.replace(ext, '.avi')}"

    input_path = UPLOAD_VIDEO_DIR / input_name

    # save uploaded video
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # =============================
    # RUN PIPELINE
    # =============================
    summary = run_full_pipeline(
        video_path=input_path,
        output_name=output_name
    )

    return templates.TemplateResponse(
        "full_pipeline.html",
        {
            "request": request,
            "video_path": f"/outputs/videos/{output_name}",
            "summary": summary
        }
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
    Run pipeline using existing video path
    """

    output_name = "result.avi"

    summary = run_full_pipeline(
        video_path=video_path,
        output_name=output_name
    )

    return templates.TemplateResponse(
        "pipeline_result.html",
        {
            "request": request,
            "summary": summary,
            "video_path": f"/outputs/videos/{output_name}"
        }
    )

# =============================
# API STATUS / HEALTH
# =============================

@router.get("/api/health", response_class=JSONResponse)
def health_check():
    return {
        "status": "ok",
        "service": "F1 Damage & Tracking System"
    }


@router.get("/api/info", response_class=JSONResponse)
def api_info():
    return {
        "message": "F1 Computer Vision API",
        "features": [
            "Car Detection",
            "Tracking",
            "Speed Estimation",
            "Damage Detection",
            "Collision Detection",
            "Overtake Detection"
        ]
    }
