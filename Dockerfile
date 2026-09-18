# =========================================================
# F1 Damage & Tracking System — CPU-only runtime image.
#
# Python 3.11 is required: the bundled FastAI team-classifier
# pickle will not deserialize on 3.10 or older.
# =========================================================
FROM python:3.11-slim AS runtime

# Fail fast, unbuffered logs, no pip cache bloat in the layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV needs libGL + glib even in headless use. ffmpeg publishes the
# annotated intermediate as an H.264/AAC MP4 that browsers can play.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependencies (own layer, cached across code changes) ---
# Install the CPU-only torch/torchvision from PyTorch's CPU index
# FIRST. On Linux the default PyPI torch wheel bundles CUDA (~2GB+);
# pinning the +cpu build keeps the image small and portable. These
# versions satisfy the torch/torchvision pins in requirements.txt,
# so the next step does not re-download a CUDA build.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# --- Application code + model weights + demo assets ---
# .dockerignore trims the training dataset and generated media so
# only the servable app and the pinned models land in the image.
COPY . .

# Ensure runtime output dirs exist even if trimmed from context.
RUN mkdir -p outputs/videos uploads/images uploads/videos app/static/results

EXPOSE 8000

# Container-friendly probe for orchestrators / compose healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

# Bind to 0.0.0.0 so the port is reachable from the host.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
