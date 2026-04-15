"""
Multi-temporal change detection module.

Compares current classification results against historical records to:
  - Identify newly-detected mining sites
  - Flag expanding operations (>20 % area growth)
  - Escalate sites active for ≥3 consecutive periods to CRITICAL
  - Compute Morphological Complexity Index (MCI) to distinguish
    anthropogenic vs natural sandbar changes
"""

from __future__ import annotations
import json
import math
import datetime
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parents[2] / "data" / "site_history.json"


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DetectedSite:
    site_id: str
    lat: float
    lon: float
    area_m2: float
    confidence: float
    bbox: List[float]          # [lon_min, lat_min, lon_max, lat_max]
    date_detected: str
    severity: str = "LOW"
    status: str = "NEW"        # NEW | EXPANDING | STABLE | CLOSED
    mci: float = 0.0           # Morphological Complexity Index
    consecutive_periods: int = 1
    history: List[Dict] = field(default_factory=list)


SEVERITY_THRESHOLDS = {
    "LOW":      (0,      5_000),
    "MEDIUM":   (5_000,  25_000),
    "HIGH":     (25_000, 100_000),
    "CRITICAL": (100_000, float("inf")),
}


# ─── Persistence (lightweight JSON store; swap for PostGIS in prod) ───────────

def _load_db() -> Dict:
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt site DB; starting fresh")
    return {}


def _save_db(db: Dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db, indent=2, default=str))


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _bbox_overlap(a: List[float], b: List[float]) -> bool:
    """Check if two [lon_min, lat_min, lon_max, lat_max] boxes overlap."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _iou_bboxes(a: List[float], b: List[float]) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-12)


def compute_mci(mask: np.ndarray) -> float:
    """
    Morphological Complexity Index — approximation of fractal dimension.
    High MCI (>1.3) → irregular anthropogenic boundary.
    Low MCI (<1.1)  → smooth natural sandbar.
    """
    if mask.sum() < 5:
        return 0.0
    perimeter = _approx_perimeter(mask)
    area = mask.sum()
    if area == 0:
        return 0.0
    # Normalised: circle has MCI=1, more complex shapes > 1
    return perimeter / (2 * math.sqrt(math.pi * area) + 1e-8)


def _approx_perimeter(mask: np.ndarray) -> float:
    """Count boundary pixels (4-connectivity)."""
    padded = np.pad(mask, 1, constant_values=0)
    boundary = (
        (padded[1:-1, 1:-1] != padded[:-2, 1:-1]) |
        (padded[1:-1, 1:-1] != padded[2:,  1:-1]) |
        (padded[1:-1, 1:-1] != padded[1:-1, :-2]) |
        (padded[1:-1, 1:-1] != padded[1:-1, 2:])
    )
    return float(boundary.sum())


# ─── Severity rating ──────────────────────────────────────────────────────────

def _rate_severity(area_m2: float,
                   consecutive: int,
                   status: str,
                   mci: float) -> str:
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= area_m2 < hi:
            sev = label
            break
    else:
        sev = "CRITICAL"

    # Escalate for long-running / expanding / morphologically complex sites
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    idx = levels.index(sev)
    if consecutive >= 3 or status == "EXPANDING" or mci > 1.4:
        idx = min(idx + 1, len(levels) - 1)
    return levels[idx]


# ─── Unique site ID ───────────────────────────────────────────────────────────

def _make_id(lat: float, lon: float, date: str) -> str:
    grid_lat = round(lat, 3)
    grid_lon = round(lon, 3)
    return f"SITE_{grid_lat}_{grid_lon}_{date.replace('-', '')}"


# ─── Core change detection ────────────────────────────────────────────────────

def process_detections(
    raw_detections: List[Dict],
    scene_date: str,
    mask_by_site: Optional[Dict[str, np.ndarray]] = None,
) -> List[DetectedSite]:
    """
    Compare new raw detections against historical DB.

    Parameters
    ----------
    raw_detections : list of dicts from inference module, each with:
                     lat, lon, area_m2, confidence, bbox
    scene_date     : "YYYY-MM-DD"
    mask_by_site   : optional {site_id: binary_mask} for MCI computation

    Returns
    -------
    list of DetectedSite with status / severity filled in
    """
    db = _load_db()
    today = scene_date
    results: List[DetectedSite] = []

    for det in raw_detections:
        lat, lon = det["lat"], det["lon"]
        area = det["area_m2"]
        conf = det["confidence"]
        bbox = det.get("bbox", [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001])

        site_id = _make_id(lat, lon, today)

        # Check if overlaps with existing site
        matched_key = None
        for existing_key, rec in db.items():
            if _iou_bboxes(bbox, rec["bbox"]) > 0.20:
                matched_key = existing_key
                break

        mci = 0.0
        if mask_by_site and site_id in mask_by_site:
            mci = compute_mci(mask_by_site[site_id])

        if matched_key:
            rec = db[matched_key]
            prev_area = rec["area_m2"]
            consecutive = rec.get("consecutive_periods", 1) + 1

            if area > prev_area * 1.20:
                status = "EXPANDING"
            elif area < prev_area * 0.80:
                status = "STABLE"  # shrinking
            else:
                status = "STABLE"

            severity = _rate_severity(area, consecutive, status, mci)

            history_entry = {
                "date": today, "area_m2": area, "confidence": conf, "mci": mci
            }
            site = DetectedSite(
                site_id=matched_key,
                lat=lat, lon=lon,
                area_m2=area,
                confidence=conf,
                bbox=bbox,
                date_detected=rec["date_detected"],
                severity=severity,
                status=status,
                mci=mci,
                consecutive_periods=consecutive,
                history=rec.get("history", []) + [history_entry],
            )
            db[matched_key] = asdict(site)

        else:
            severity = _rate_severity(area, 1, "NEW", mci)
            site = DetectedSite(
                site_id=site_id,
                lat=lat, lon=lon,
                area_m2=area,
                confidence=conf,
                bbox=bbox,
                date_detected=today,
                severity=severity,
                status="NEW",
                mci=mci,
                consecutive_periods=1,
                history=[{"date": today, "area_m2": area, "confidence": conf, "mci": mci}],
            )
            db[site_id] = asdict(site)

        results.append(site)
        logger.info(
            f"[{site.status}] {site.site_id} | "
            f"area={area:.0f} m² | conf={conf:.2f} | sev={severity}"
        )

    _save_db(db)
    return results


def get_all_sites() -> List[Dict]:
    """Return all historical sites from the DB."""
    return list(_load_db().values())


def get_active_sites(min_severity: str = "LOW") -> List[Dict]:
    """Return sites filtered by minimum severity."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    min_idx = levels.index(min_severity)
    return [
        s for s in _load_db().values()
        if levels.index(s.get("severity", "LOW")) >= min_idx
    ]
