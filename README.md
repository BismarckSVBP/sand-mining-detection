# 🛰️ Illegal Sand Mining Detection from Space

**Satellite-based real-time detection of illegal sand mining using image processing and deep learning.**

> Group G5 · B.Tech CSE · MMMUT Gorakhpur  
> Abhishek Singh · Abhay Kumar · Parv Agarwal · Sonu Kumar  
> Supervisor: Dr. Ninni Singh

---

## 📋 Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [Quick Demo (no install)](#3-quick-demo-no-install)
4. [Local Setup — Python (recommended for dev)](#4-local-setup--python)
5. [Local Setup — Docker](#5-local-setup--docker)
6. [Training the Model](#6-training-the-model)
7. [Using Real Satellite Data (free, no account)](#7-using-real-satellite-data-free-no-account)
8. [Using Google Earth Engine (optional)](#8-using-google-earth-engine-optional)
9. [Production Deployment (free tier)](#9-production-deployment-free-tier)
10. [API Reference](#10-api-reference)
11. [How to Check the Demo](#11-how-to-check-the-demo)
12. [Open-Source Datasets for Training](#12-open-source-datasets-for-training)
13. [Architecture Deep-dive](#13-architecture-deep-dive)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. System Overview

```
Satellite (Sentinel-2 / Landsat-8)
         │
         ▼
┌─────────────────────┐
│  Data Acquisition   │  GEE / AWS Open Data / synthetic demo
└────────┬────────────┘
         │  (6-band GeoTIFF)
         ▼
┌─────────────────────┐
│  Preprocessing      │  NDWI · NDVI · Sand Index · river corridor · tiling
└────────┬────────────┘
         │  (7-ch 256×256 patches)
         ▼
┌─────────────────────┐
│  U-Net (ResNet-50)  │  semantic segmentation → probability map
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Change Detection   │  MCI · site tracking · severity scoring
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Alert Generation   │  GeoJSON · REST API · Leaflet map
└─────────────────────┘
```

**Expected performance** (after training on real data):

| Metric | Target |
|---|---|
| Classification accuracy | ≥ 90% |
| Precision (mining class) | ≥ 85% |
| Recall (mining class) | ≥ 88% |
| Mean IoU | ≥ 0.75 |
| Alert latency | < 12 hours |
| False positive rate | < 10% |

---

## 2. Project Structure

```
sand-mining-detection/
├── backend/
│   ├── main.py                  FastAPI application
│   ├── requirements.txt
│   └── modules/
│       ├── model.py             U-Net + ResNet-50 + DiceBCE loss
│       ├── preprocessing.py     Spectral indices, river masking, tiling
│       ├── data_acquisition.py  GEE / AWS STAC / synthetic data
│       ├── inference.py         Full detection pipeline
│       ├── change_detection.py  Temporal tracking, MCI, severity
│       └── alert_generator.py   GeoJSON reports, email summaries
├── frontend/
│   └── index.html               Single-page Leaflet.js dashboard
├── scripts/
│   ├── train.py                 Training loop (synthetic + real data)
│   └── download_demo_data.py    Download open-source Sentinel-2 tiles
├── tests/
│   └── test_pipeline.py         Pytest test suite
├── models/                      Saved model weights (.pth)
├── data/
│   ├── demo/                    Cached synthetic scenes
│   ├── raw/                     Downloaded satellite tiles
│   ├── processed/               Training patches (images/ + masks/)
│   └── alerts/                  Saved alert reports (JSON)
├── Dockerfile
├── docker-compose.yml
├── render.yaml                  One-click Render.com deploy
└── README.md
```

---

## 3. Quick Demo (no install)

If you just want to see the system working **without installing anything**:

```bash
# Option A — curl the hosted demo (after deploying to Render)
curl https://YOUR-APP.onrender.com/api/demo | python -m json.tool

# Option B — open the API docs in a browser
# After any local or cloud deployment:
# http://localhost:8000/docs  → Swagger UI with Try it out buttons
```

---

## 4. Local Setup — Python

### Prerequisites
- Python 3.10 or 3.11 (3.12 works but GDAL may need manual install)
- pip ≥ 23
- Git

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/sand-mining-detection.git
cd sand-mining-detection

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install core dependencies (lightweight, no GDAL)
pip install fastapi uvicorn pydantic numpy scipy \
            requests python-dotenv aiofiles

# 4. Install PyTorch (CPU version — smaller, works everywhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Optional: full geospatial stack (for real satellite imagery)
#    On Ubuntu/Debian:
#    sudo apt-get install libgdal-dev
#    pip install rasterio GDAL geopandas shapely
#
#    On Windows: use conda for GDAL
#    conda install -c conda-forge gdal rasterio geopandas

# 6. Generate the synthetic demo scene
python scripts/download_demo_data.py

# 7. Start the server
cd backend   # or stay in root and use:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# OR from inside backend/:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 8. Open the dashboard
#    http://localhost:8000/
#    http://localhost:8000/docs   ← Swagger UI
```

### Run the tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 5. Local Setup — Docker

### Prerequisites
- Docker Desktop ≥ 24 (or Docker Engine + Compose plugin on Linux)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sand-mining-detection.git
cd sand-mining-detection

# 2. Build and start (first build takes ~3–5 min)
docker compose up --build

# 3. Open
#    http://localhost:8000/        ← Dashboard
#    http://localhost:8000/docs    ← API docs

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build
```

---

## 6. Training the Model

### Using synthetic data (no downloads, ~2 min)

```bash
python scripts/train.py --demo --epochs 10 --batch-size 4

# Faster smoke test
python scripts/train.py --demo --epochs 2 --n-samples 100
```

The trained weights are saved to `models/unet_weights.pth`.  
Restart the server — it will automatically load them.

### Using real labeled data

Prepare your data in this layout:

```
data/processed/
  images/
    patch_00001.npy   # shape (7, 256, 256) float32, values normalized
    patch_00002.npy
    ...
  masks/
    patch_00001.npy   # shape (1, 256, 256) float32, values 0.0 or 1.0
    patch_00002.npy
    ...
```

Then:

```bash
python scripts/train.py \
    --data-dir data/processed \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.0001
```

For GPU training:

```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Training automatically uses GPU when available
python scripts/train.py --data-dir data/processed --epochs 100
```

---

## 7. Using Real Satellite Data (free, no account)

### Option A — AWS Open Data (Sentinel-2, no auth required)

```bash
# Download real Sentinel-2 tiles for the three target rivers
python scripts/download_demo_data.py --river all --year 2024 --dry-season

# Individual rivers
python scripts/download_demo_data.py --river ghaghra
python scripts/download_demo_data.py --river rapti
python scripts/download_demo_data.py --river gandak
```

This uses the [AWS Element84 STAC API](https://earth-search.aws.element84.com/v1)  
which provides free access to Sentinel-2 L2A data with no account or API key.

**Requires:** `rasterio` (`pip install rasterio`)

### Option B — Copernicus Open Access Hub (ESA, free account)

```bash
pip install sentinelsat

# Create a free account at https://scihub.copernicus.eu/dhus/
# Then set credentials:
export SENTINEL_USER=your_username
export SENTINEL_PASS=your_password

# The data_acquisition module automatically uses these if set
```

### Option C — USGS Earth Explorer (Landsat, free account)

1. Register at https://earthexplorer.usgs.gov
2. Search for Landsat-8/9 Collection 2 Level-2 tiles
3. Download bands: SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7
4. Place in `data/raw/RIVER_NAME/`
5. Use the preprocessing module to convert to the 6-band stack

---

## 8. Using Google Earth Engine (optional)

GEE provides the best data quality but requires account setup.

```bash
# 1. Create a free account at https://earthengine.google.com/
# 2. Install the Python API
pip install earthengine-api

# 3. Authenticate (opens browser)
earthengine authenticate

# 4. Test access
python -c "import ee; ee.Initialize(); print('GEE ready')"

# 5. Use in the API — set source="gee" in requests
```

In the dashboard dropdown, select **"Google Earth Engine"** as the data source.

---

## 9. Production Deployment (free tier)

### Option A — Render.com (recommended, easiest)

Render offers a **free tier** with 512 MB RAM, sufficient for demo.

1. Push this repo to a **public GitHub repository**
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**
3. Connect your GitHub repo
4. Render detects `render.yaml` automatically → click **Apply**
5. Wait ~5 minutes for the first build
6. Your app is live at `https://sand-mining-api.onrender.com`

> ⚠️ Free tier spins down after 15 min of inactivity (cold start ~30 s).

### Option B — Railway.app (free $5/month credit)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Option C — Fly.io (free allowance)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch --dockerfile Dockerfile
fly deploy
```

### Option D — Hugging Face Spaces (free, Docker)

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** SDK
3. Push this repo — the `Dockerfile` is auto-detected

### Option E — Google Cloud Run (free 2M requests/month)

```bash
gcloud run deploy sand-mining \
    --source . \
    --platform managed \
    --region asia-south1 \
    --allow-unauthenticated \
    --memory 512Mi
```

---

## 10. API Reference

Base URL: `http://localhost:8000` (or your deployed URL)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard (Leaflet map UI) |
| GET | `/health` | Health check |
| POST | `/api/detect` | Run detection pipeline |
| GET | `/api/demo` | Quick demo with synthetic data |
| GET | `/api/alerts` | List all saved reports |
| GET | `/api/alerts/{id}` | Get specific report |
| GET | `/api/alerts/geojson` | Latest detections as GeoJSON |
| GET | `/api/sites` | All sites from DB |
| GET | `/api/sites/geojson` | Sites as GeoJSON |
| GET | `/docs` | Swagger UI |

### Example: POST /api/detect

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [83.2, 26.5, 83.6, 26.9],
    "date_start": "2024-01-01",
    "date_end": "2024-01-31",
    "source": "demo"
  }'
```

Response structure:

```json
{
  "report_id": "RPT_20240115_103045",
  "generated_at": "2024-01-15T10:30:45Z",
  "summary": {
    "total_sites_detected": 3,
    "by_severity": {"LOW": 1, "MEDIUM": 1, "HIGH": 1, "CRITICAL": 0},
    "total_area_ha": 12.5
  },
  "geojson": { "type": "FeatureCollection", "features": [...] },
  "sites": [...]
}
```

---

## 11. How to Check the Demo

### Step 1 — Start the server (choose one method above)

```bash
uvicorn backend.main:app --port 8000 --reload
```

### Step 2 — Open the dashboard

Navigate to **http://localhost:8000/**

### Step 3 — Run a demo detection

Click the **⚡ Quick Demo** button — this:
1. Generates a synthetic Sentinel-2 scene of the Ghaghra river
2. Runs the preprocessing pipeline (NDWI, river corridor, tiling)
3. Passes patches through the U-Net (random weights unless trained)
4. Applies change detection and severity scoring
5. Displays results on the Leaflet map with colour-coded polygons

### Step 4 — Explore results

- **Click any red/purple polygon** on the map for site details
- **Results tab** → list of all saved reports
- **Sites tab** → database of all detected locations
- **Satellite layer** → switch to Esri imagery using the layer control (top-right)

### Step 5 — Download GeoJSON for QGIS

```bash
curl http://localhost:8000/api/alerts/geojson > detections.geojson
```

Open in QGIS: Layer → Add Layer → Add Vector Layer → select `detections.geojson`

### Step 6 — Try the Swagger UI

Open **http://localhost:8000/docs** → Try out the `/api/detect` endpoint with:
- `source: "demo"` for synthetic data
- `source: "aws"` for real Sentinel-2 (requires `rasterio`)
- `source: "gee"` for GEE (requires auth)

---

## 12. Open-Source Datasets for Training

| Dataset | Description | URL | License |
|---|---|---|---|
| **Global Surface Mining** (Maus et al. 2020) | 21,000+ labeled mining polygons worldwide | [doi:10.1038/s41597-020-00624-w](https://doi.org/10.1038/s41597-020-00624-w) | CC BY 4.0 |
| **EuroSAT** | Sentinel-2 land cover, 27k labeled patches | [GitHub](https://github.com/phelber/EuroSAT) | MIT |
| **LandCover.ai** | 0.25 m aerial imagery with masks | [landcover.ai](https://landcover.ai/) | CC BY 4.0 |
| **Sen12MS** | Sentinel-1/2 + land cover labels | [GitHub](https://github.com/schmitt-muc/SEN12MS) | CC BY 4.0 |
| **BigEarthNet** | 590k Sentinel-2 patches, multi-label | [bigearthnet.eu](https://bigearthnet.eu) | Custom open |
| **DOTA** | Remote sensing object detection | [captain-whu.github.io](https://captain-whu.github.io/DOTA/) | Research |
| **AWS Sentinel-2** | Complete archive, free via STAC | [earth-search.aws.element84.com](https://earth-search.aws.element84.com/v1) | Open |
| **USGS Landsat** | Archive since 1972, free | [earthexplorer.usgs.gov](https://earthexplorer.usgs.gov) | Open |

### How to prepare your own training labels

1. Download Sentinel-2 tiles for river areas using `download_demo_data.py`
2. Open in **QGIS** → Digitize mining polygons as a vector layer
3. Export labels as binary raster (1=mining, 0=background)
4. Run preprocessing to convert to 256×256 patches
5. Train with `python scripts/train.py --data-dir data/processed`

---

## 13. Architecture Deep-dive

### Model: U-Net + ResNet-50

```
Input (7, 256, 256)   ← RGB + NIR + SWIR1 + SWIR2 + Sand Index
     │
     ├─ ResNet-50 Encoder ─────────────────────────────────────┐
     │   conv1 (64ch, /2)                                       │
     │   layer1 (256ch, /4)                                     │
     │   layer2 (512ch, /8)                                     │
     │   layer3 (1024ch, /16)                                   │
     │   layer4 (2048ch, /32)                                   │
     │                                                          │
     └─ U-Net Decoder (bilinear upsampling + skip connections) ─┘
          dec4: 2048→512 + skip(1024)
          dec3:  512→256 + skip(512)
          dec2:  256→128 + skip(256)
          dec1:  128→64  + skip(64)
          dec0:   64→32  (no skip)
          head:   32→1   sigmoid

Output (1, 256, 256)  ← probability map [0,1]
```

### Spectral indices

| Index | Formula | Purpose |
|---|---|---|
| NDWI | (Green − NIR) / (Green + NIR) | Water body detection |
| MNDWI | (Green − SWIR1) / (Green + SWIR1) | Urban/turbid water |
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation density |
| Sand Index | SWIR1 / Green | Dry sand detection |
| BSI | ((SWIR1+Red) − (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue)) | Bare soil |

### Change detection — Morphological Complexity Index (MCI)

```
MCI = perimeter / (2 × √(π × area))

MCI = 1.0   → perfect circle (natural sandbar)
MCI > 1.3   → irregular shape (likely anthropogenic)
MCI > 1.5   → highly complex (almost certainly mining)
```

---

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'rasterio'`

The geospatial stack is optional for the demo. Install it:

```bash
# Linux
sudo apt-get install libgdal-dev
pip install rasterio GDAL

# macOS
brew install gdal
pip install rasterio

# Windows (recommended via conda)
conda install -c conda-forge gdal rasterio geopandas
```

### `RuntimeError: GEE initialisation failed`

Run `earthengine authenticate` and follow the browser prompts.  
Or use `source: "demo"` or `source: "aws"` instead.

### `torch not found` or model warnings

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### No mining sites detected

Expected with untrained (random) weights. Either:
1. Train the model: `python scripts/train.py --demo --epochs 20`
2. The demo scene has a built-in synthetic mining anomaly — random weights may or may not detect it

### Docker build fails on GDAL

GDAL is only needed for reading real satellite tiles. The demo runs without it.  
Remove the GDAL apt-get line from the Dockerfile if building fails.

### Port 8000 already in use

```bash
uvicorn backend.main:app --port 8080 --reload
# or kill the process:
lsof -ti:8000 | xargs kill -9
```

---
