# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for GDAL / rasterio / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY backend/requirements.txt .

# Install CPU-only PyTorch (smaller image for free-tier deployment)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps (skip GDAL binary, use system GDAL)
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.3.0 \
    rasterio==1.3.10 \
    requests==2.31.0 \
    python-dotenv==1.0.1 \
    aiofiles==23.2.1 \
    httpx==0.27.0 \
    tqdm==4.66.2

# Copy application
COPY backend/  /app/backend/
COPY frontend/ /app/frontend/
COPY data/     /app/data/
COPY models/   /app/models/
COPY scripts/  /app/scripts/

# Create necessary directories
RUN mkdir -p /app/data/demo /app/data/alerts /app/models

# Pre-generate the synthetic demo scene
RUN python -c "
import sys; sys.path.insert(0,'backend')
from modules.data_acquisition import _generate_demo_scene
_generate_demo_scene((83.2,26.5,83.6,26.9),'2024-01-01')
print('Demo scene ready.')
"

EXPOSE 8000

# Use exec form so SIGTERM is received by uvicorn directly
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
