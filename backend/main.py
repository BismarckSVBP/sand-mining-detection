"""
FastAPI REST backend for the Illegal Sand Mining Detection system.

Endpoints:
  GET  /                       → Serve frontend HTML
  POST /api/detect             → Run detection on a scene
  GET  /api/alerts             → List all alert reports
  GET  /api/alerts/{report_id} → Get a specific report
  GET  /api/alerts/geojson     → Latest GeoJSON (for QGIS / Leaflet)
  GET  /api/sites              → All detected sites (DB)
  GET  /api/demo               → Run demo with synthetic data
  GET  /health                 → Health check
"""

from __future__ import annotations
import os
import json
import logging
import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Adjust sys.path so relative imports work regardless of invocation method
import sys
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_acquisition import acquire_scene
from modules.inference import run_inference, load_model
from modules.change_detection import process_detections, get_all_sites, get_active_sites
from modules.alert_generator import generate_alert_report, sites_to_geojson, format_email_body

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Illegal Sand Mining Detection API",
    description=(
        "Satellite-based detection of illegal sand mining using "
        "image processing and deep learning. "
        "Group G5 · MMMUT Gorakhpur."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
ALERT_DIR    = Path(__file__).parent.parent / "data" / "alerts"

if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

# Pre-warm model on startup
_model = None


@app.on_event("startup")
async def startup():
    global _model
    try:
        _model = load_model()
        logger.info(f"Model ready on {_model.head[0].weight.device}")
    except Exception as e:
        logger.warning(f"Model warm-up skipped: {e}")


# ─── Request / Response models ────────────────────────────────────────────────

class DetectRequest(BaseModel):
    bbox: list[float] = Field(
        default=[83.2, 26.5, 83.6, 26.9],
        description="[lon_min, lat_min, lon_max, lat_max] WGS-84",
        min_length=4, max_length=4,
    )
    date_start: str = Field(default="2024-01-01", description="YYYY-MM-DD")
    date_end: str   = Field(default="2024-01-31", description="YYYY-MM-DD")
    source: str     = Field(
        default="demo",
        description="Data source: 'demo' | 'aws' | 'gee'",
    )
    cloud_pct: float = Field(default=20.0, ge=0, le=100)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h2>Sand Mining Detection API is running. Visit <a href='/docs'>/docs</a></h2>")


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}


@app.post("/api/detect", summary="Run detection on a satellite scene")
async def detect(req: DetectRequest):
    """
    Download or synthesise a satellite scene and run the full
    detection + change-detection pipeline.
    Returns a structured alert report.
    """
    try:
        # 1. Acquire scene
        logger.info(f"Acquiring scene: bbox={req.bbox} source={req.source}")
        scene = acquire_scene(
            bbox=tuple(req.bbox),
            date_range=(req.date_start, req.date_end),
            source=req.source,
            cloud_pct=req.cloud_pct,
        )

        # 2. Inference
        result = run_inference(scene, model=_model)

        # 3. Change detection
        sites = process_detections(
            result["detections"], scene_date=req.date_start
        )

        # 4. Alert report
        report = generate_alert_report(sites, scene["meta"])

        return report

    except Exception as e:
        logger.exception("Detection pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/demo", summary="Run demo detection with synthetic data")
async def demo():
    """Quick demo run using a synthetic Ghaghra river scene."""
    req = DetectRequest(
        bbox=[83.2, 26.5, 83.6, 26.9],
        date_start="2024-03-15",
        date_end="2024-03-30",
        source="demo",
    )
    return await detect(req)


@app.get("/api/alerts", summary="List all saved alert reports")
async def list_alerts():
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(ALERT_DIR.glob("*.json"), reverse=True)[:20]:
        try:
            data = json.loads(path.read_text())
            reports.append({
                "report_id": data.get("report_id"),
                "generated_at": data.get("generated_at"),
                "total_sites": data.get("summary", {}).get("total_sites_detected", 0),
                "total_area_ha": data.get("summary", {}).get("total_area_ha", 0),
            })
        except Exception:
            pass
    return {"count": len(reports), "reports": reports}


@app.get("/api/alerts/geojson", summary="Latest detections as GeoJSON")
async def latest_geojson():
    """Returns GeoJSON suitable for loading in QGIS, ArcGIS, or Leaflet."""
    sites = get_all_sites()
    return sites_to_geojson(sites)


@app.get("/api/alerts/{report_id}", summary="Get a specific alert report")
async def get_report(report_id: str):
    path = ALERT_DIR / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return json.loads(path.read_text())


@app.get("/api/sites", summary="All detected sites from DB")
async def all_sites(
    min_severity: str = Query(default="LOW",
                              description="Minimum severity: LOW | MEDIUM | HIGH | CRITICAL"),
):
    return {"sites": get_active_sites(min_severity)}


@app.get("/api/sites/geojson", summary="All sites as GeoJSON")
async def sites_geojson(min_severity: str = Query(default="LOW")):
    sites = get_active_sites(min_severity)
    return sites_to_geojson(sites)
