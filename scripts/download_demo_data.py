"""
Download open-source Sentinel-2 imagery for the Ghaghra / Rapti / Gandak rivers.

No API key required — uses AWS Open Data Registry (public, no auth).

Usage
-----
python scripts/download_demo_data.py
python scripts/download_demo_data.py --river rapti --year 2023
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("download")

# ─── River bounding boxes (WGS-84) ────────────────────────────────────────────
RIVERS = {
    "ghaghra": {
        "bbox": [82.9, 26.4, 83.5, 27.0],
        "description": "Ghaghra (Ghaghara) river, Uttar Pradesh",
    },
    "rapti": {
        "bbox": [83.0, 27.0, 83.8, 27.6],
        "description": "Rapti river, Uttar Pradesh / Bihar border",
    },
    "gandak": {
        "bbox": [84.0, 25.8, 84.8, 26.5],
        "description": "Gandak river, Bihar",
    },
}

DATA_DIR = Path(__file__).parent.parent / "data"


def _try_aws_stac(bbox, date_start, date_end, out_path):
    """Download a real Sentinel-2 scene from AWS via STAC."""
    try:
        import requests
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
    except ImportError as e:
        logger.warning(f"Missing library ({e}); falling back to synthetic data")
        return False

    lon_min, lat_min, lon_max, lat_max = bbox
    stac_url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 20}},
        "limit": 5,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }

    try:
        resp = requests.post(stac_url, json=payload, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        logger.warning(f"STAC search failed: {e}")
        return False

    if not features:
        logger.warning("No cloud-free scenes found in date range")
        return False

    item = features[0]
    logger.info(f"Found scene: {item['id']}")
    logger.info(f"  Cloud cover: {item['properties'].get('eo:cloud_cover')}%")
    logger.info(f"  Date: {item['properties']['datetime'][:10]}")

    band_map = {
        "B02": 0, "B03": 1, "B04": 2,
        "B08": 3, "B11": 4, "B12": 5,
    }
    # AWS asset keys
    aws_keys = ["blue", "green", "red", "nir", "swir16", "swir22"]
    stac_keys = ["B02", "B03", "B04", "B08", "B11", "B12"]

    arrays = {}
    assets = item["assets"]

    for s_key, aws_key, idx in zip(stac_keys, aws_keys, range(6)):
        href = None
        for k in [s_key, aws_key, s_key.lower()]:
            if k in assets:
                href = assets[k]["href"]
                break
        if href is None:
            logger.warning(f"Could not find asset for band {s_key}")
            continue

        try:
            logger.info(f"  Downloading band {s_key} …")
            with rasterio.open(href) as src:
                dst_crs = CRS.from_epsg(4326)
                bounds = transform_bounds(dst_crs, src.crs, *bbox)
                window = from_bounds(*bounds, transform=src.transform)
                data = src.read(
                    1,
                    window=window,
                    out_shape=(512, 512),
                    resampling=Resampling.bilinear,
                )
            arrays[idx] = data.astype(np.float32) / 10_000.0
        except Exception as e:
            logger.warning(f"  Failed to read {s_key}: {e}")

    if len(arrays) < 6:
        logger.warning(f"Only {len(arrays)}/6 bands downloaded; skipping real scene")
        return False

    bands = np.stack([arrays[i] for i in range(6)])
    bands = np.clip(bands, 0.0, 1.0)

    meta = {
        "source": "AWS/Sentinel-2",
        "scene_id": item["id"],
        "date": item["properties"]["datetime"][:10],
        "cloud_pct": item["properties"].get("eo:cloud_cover"),
        "bbox": bbox,
        "shape": list(bands.shape),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "bands.npy", bands)
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved → {out_path}")
    return True


def _generate_synthetic(bbox, river_name, out_path):
    """Generate and save a realistic synthetic scene."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from modules.data_acquisition import _generate_demo_scene

    scene = _generate_demo_scene(bbox, "2024-01-01", size=512)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "bands.npy", scene["bands"])
    (out_path / "meta.json").write_text(json.dumps(scene["meta"], indent=2))
    logger.info(f"Synthetic scene saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-2 demo data")
    parser.add_argument("--river", choices=list(RIVERS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--dry-season", action="store_true",
                        help="Use dry season (Nov–Mar) — less cloud cover")
    args = parser.parse_args()

    rivers = RIVERS if args.river == "all" else {args.river: RIVERS[args.river]}

    for name, info in rivers.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {info['description']}")

        if args.dry_season:
            date_start = f"{args.year - 1}-11-01"
            date_end   = f"{args.year}-02-28"
        else:
            date_start = f"{args.year}-01-01"
            date_end   = f"{args.year}-03-31"

        out_path = DATA_DIR / "raw" / name

        success = _try_aws_stac(info["bbox"], date_start, date_end, out_path)
        if not success:
            logger.info("Generating synthetic scene instead …")
            _generate_synthetic(info["bbox"], name, out_path)

    logger.info("\nDone! Data saved in data/raw/")
    logger.info("Run the API and visit http://localhost:8000/ to view detections.")


if __name__ == "__main__":
    main()
