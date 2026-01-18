from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router

# =============================
# APP INIT
# =============================
app = FastAPI(
    title="F1 Damage & Tracking System",
    description="Image Damage Detection + Full Race Pipeline",
    version="1.0"
)

# =============================
# CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# STATIC FILES
# =============================

# Serve static assets (CSS, images, result videos)
# app/static/  -> /static
app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)

# Serve pipeline output videos directly
# outputs/ -> /outputs
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)

# =============================
# ROUTES
# =============================
app.include_router(router)

# =============================
# ROOT API (JSON)
# =============================
@app.get("/api")
def api_root():
    return {
        "message": "F1 Computer Vision API is running 🚀",
        "pages": {
            "home": "/",
            "image_damage_page": "/page/damage",
            "full_pipeline_page": "/page/pipeline"
        }
    }
