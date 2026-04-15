"""
Satellite image acquisition module.

Primary source  : Google Earth Engine (Sentinel-2 L2A)
Fallback source : AWS Open Data Registry (Sentinel-2 COGs, no auth required)
Demo mode       : Synthetic/cached tiles included in the repo

Usage
-----
from modules.data_acquisition import acquire_scene

scene = acquire_scene(
    bbox=(lon_min, lat_min, lon_max, lat_max),
    date_range=("2024-01-01", "2024-01-31"),
    source="auto",          # "gee" | "aws" | "demo"
)
# scene["bands"]     → (6, H, W) float32 [0–1]
# scene["meta"]      → dict with date, sensor, cloud_pct, crs, transform
"""

from __future__ import annotations
import os
import json
import logging
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parents[2] / "data"
DEMO_DIR = DATA_DIR / "demo"

# Sentinel-2 band names (L2A surface reflectance)
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
SCALE_FACTOR = 10_000  # DN → reflectance


# ─── Google Earth Engine acquisition ──────────────────────────────────────────

def _acquire_gee(bbox: Tuple[float, float, float, float],
                 date_start: str,
                 date_end: str,
                 cloud_pct: float = 20.0) -> Dict:
    """Download median composite via GEE Python API."""
    try:
        import ee
        import requests
        ee.Initialize()
    except Exception as e:
        raise RuntimeError(
            f"GEE initialisation failed: {e}. "
            "Run `earthengine authenticate` and ensure the ee package is installed."
        )

    region = ee.Geometry.BBox(*bbox)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .select(S2_BANDS)
    )

    count = collection.size().getInfo()
    if count == 0:
        # Fallback: widen search window
        logger.warning("No cloud-free Sentinel-2 scenes; trying Landsat-8")
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
            .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        )

    composite = collection.median().divide(SCALE_FACTOR).toFloat()

    url = composite.getDownloadURL({
        "bands": S2_BANDS,
        "region": region,
        "scale": 10,
        "format": "NPY",
    })

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    arr = np.load(
        __import__("io").BytesIO(resp.content), allow_pickle=True
    )  # structured array from GEE

    # Convert structured array to (6, H, W) float32
    bands = np.stack([arr[b].astype(np.float32) for b in S2_BANDS])
    bands = np.clip(bands, 0.0, 1.0)

    return {
        "bands": bands,
        "meta": {
            "source": "GEE/Sentinel-2",
            "date_start": date_start,
            "date_end": date_end,
            "bbox": bbox,
            "cloud_pct": cloud_pct,
            "shape": bands.shape,
        },
    }


# ─── AWS Open Data (no auth) ──────────────────────────────────────────────────

def _acquire_aws_demo(bbox: Tuple[float, float, float, float],
                      date_start: str,
                      date_end: str) -> Dict:
    """
    Use sentinelsat / STAC API to search Sentinel-2 tiles on AWS.
    This is a lightweight STAC search — no authentication needed.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests library not available")

    lon_min, lat_min, lon_max, lat_max = bbox

    stac_url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 20}},
        "limit": 5,
    }

    resp = requests.post(stac_url, json=payload, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    if not features:
        logger.warning("No STAC results for query; switching to demo mode")
        return _generate_demo_scene(bbox, date_start)

    # Pick the least-cloudy scene
    item = min(features, key=lambda f: f["properties"].get("eo:cloud_cover", 999))
    logger.info(f"Selected STAC scene: {item['id']} cloud={item['properties'].get('eo:cloud_cover')}%")

    assets = item["assets"]
    band_keys = {
        "B02": 0, "blue": 0,
        "B03": 1, "green": 1,
        "B04": 2, "red": 2,
        "B08": 3, "nir": 3,
        "B11": 4, "swir16": 4,
        "B12": 5, "swir22": 5,
    }

    arrays = {}
    for key, idx in band_keys.items():
        if key in assets and idx not in arrays:
            href = assets[key]["href"]
            try:
                arr = _read_cog_band(href, bbox)
                if arr is not None:
                    arrays[idx] = arr
            except Exception as e:
                logger.debug(f"Could not fetch band {key}: {e}")

    if len(arrays) < 6:
        logger.warning(f"Only {len(arrays)}/6 bands fetched; using demo scene")
        return _generate_demo_scene(bbox, date_start)

    H, W = arrays[0].shape
    bands = np.stack([arrays[i] for i in range(6)]).astype(np.float32)
    bands = np.clip(bands / SCALE_FACTOR, 0.0, 1.0)

    return {
        "bands": bands,
        "meta": {
            "source": "AWS/Sentinel-2",
            "scene_id": item["id"],
            "date": item["properties"]["datetime"][:10],
            "cloud_pct": item["properties"].get("eo:cloud_cover"),
            "bbox": bbox,
            "shape": bands.shape,
        },
    }


def _read_cog_band(href: str,
                   bbox: Tuple[float, float, float, float],
                   out_size: int = 512) -> Optional[np.ndarray]:
    """Read a Cloud-Optimised GeoTIFF band from a URL (windowed read)."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
    except ImportError:
        return None

    with rasterio.open(href) as src:
        dst_crs = CRS.from_epsg(4326)
        bounds = transform_bounds(dst_crs, src.crs, *bbox)
        window = from_bounds(*bounds, transform=src.transform)
        data = src.read(
            1,
            window=window,
            out_shape=(out_size, out_size),
            resampling=Resampling.bilinear,
        )
    return data.astype(np.float32)


# ─── Demo / synthetic scene ───────────────────────────────────────────────────

def _generate_demo_scene(bbox: Tuple[float, float, float, float],
                         date: str,
                         size: int = 512) -> Dict:
    """
    Generate a realistic-looking synthetic Sentinel-2 scene for demo purposes.
    Includes a simulated river channel with sand bars and a mining anomaly.
    """
    rng = np.random.default_rng(seed=42)
    H = W = size

    # Base reflectances (approximate river-corridor scene)
    bands = np.zeros((6, H, W), dtype=np.float32)

    # Background: mixed vegetation / bare soil
    bands[0] = rng.uniform(0.04, 0.08, (H, W))   # Blue
    bands[1] = rng.uniform(0.07, 0.12, (H, W))   # Green
    bands[2] = rng.uniform(0.06, 0.11, (H, W))   # Red
    bands[3] = rng.uniform(0.22, 0.38, (H, W))   # NIR (high → veg)
    bands[4] = rng.uniform(0.10, 0.20, (H, W))   # SWIR1
    bands[5] = rng.uniform(0.05, 0.12, (H, W))   # SWIR2

    # River channel (diagonal band)
    river_mask = np.zeros((H, W), bool)
    for i in range(H):
        cx = int(W * 0.4 + (W * 0.2) * np.sin(2 * np.pi * i / H))
        river_mask[i, max(0, cx-25):min(W, cx+25)] = True

    bands[:, river_mask] = [0.03, 0.05, 0.04, 0.03, 0.02, 0.01]  # water dark

    # Natural sandbars
    sb_mask = np.zeros((H, W), bool)
    for i in range(H):
        cx = int(W * 0.4 + (W * 0.2) * np.sin(2 * np.pi * i / H))
        sb_mask[i, max(0, cx+20):min(W, cx+55)] = True
    bands[:, sb_mask] = [0.22, 0.26, 0.24, 0.18, 0.32, 0.20]   # bright sand

    # Simulated mining anomaly (anomalously bright, irregular patch)
    r0, c0 = 180, 225
    mine_mask = np.zeros((H, W), bool)
    mine_mask[r0:r0+40, c0:c0+60] = True
    bands[:, mine_mask] = [0.28, 0.32, 0.30, 0.15, 0.42, 0.28]  # excavated sand

    # Add mild noise
    bands += rng.normal(0, 0.005, bands.shape).astype(np.float32)
    bands = np.clip(bands, 0.0, 1.0)

    # Save for reuse
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    cache = DEMO_DIR / "demo_scene.npy"
    np.save(cache, bands)

    return {
        "bands": bands,
        "meta": {
            "source": "synthetic_demo",
            "date": date,
            "bbox": bbox,
            "shape": bands.shape,
            "note": "Synthetic scene — replace with real satellite data for production",
        },
    }


def _load_cached_demo(bbox, date_start):
    cache = DEMO_DIR / "demo_scene.npy"
    if cache.exists():
        bands = np.load(cache)
        return {"bands": bands, "meta": {"source": "cached_demo", "bbox": bbox, "date": date_start, "shape": bands.shape}}
    return _generate_demo_scene(bbox, date_start)


# ─── Public API ───────────────────────────────────────────────────────────────

def acquire_scene(
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    source: str = "auto",
    cloud_pct: float = 20.0,
) -> Dict:
    """
    Acquire a satellite scene for the given bounding box and date range.

    Parameters
    ----------
    bbox        : (lon_min, lat_min, lon_max, lat_max) in WGS-84
    date_range  : ("YYYY-MM-DD", "YYYY-MM-DD")
    source      : "auto" | "gee" | "aws" | "demo"
    cloud_pct   : maximum cloud cover threshold (%)

    Returns
    -------
    dict with keys "bands" (6, H, W) float32 and "meta" dict
    """
    date_start, date_end = date_range

    if source == "gee":
        return _acquire_gee(bbox, date_start, date_end, cloud_pct)
    elif source == "aws":
        return _acquire_aws_demo(bbox, date_start, date_end)
    elif source == "demo":
        return _load_cached_demo(bbox, date_start)
    else:  # auto
        try:
            return _acquire_gee(bbox, date_start, date_end, cloud_pct)
        except Exception as e:
            logger.warning(f"GEE failed ({e}); trying AWS STAC …")
            try:
                return _acquire_aws_demo(bbox, date_start, date_end)
            except Exception as e2:
                logger.warning(f"AWS STAC failed ({e2}); using demo scene")
                return _load_cached_demo(bbox, date_start)
