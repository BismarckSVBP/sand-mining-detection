"""
Alert generation and formatting.

Produces structured alert reports in JSON and GeoJSON formats
suitable for consumption by GIS portals, REST clients, and email.
"""

from __future__ import annotations
import json
import datetime
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Optional

from .change_detection import DetectedSite

logger = logging.getLogger(__name__)

ALERT_DIR = Path(__file__).parents[2] / "data" / "alerts"

SEVERITY_COLOURS = {
    "LOW":      "#2ecc71",   # green
    "MEDIUM":   "#f39c12",   # orange
    "HIGH":     "#e74c3c",   # red
    "CRITICAL": "#8e44ad",   # purple
}


# ─── GeoJSON export ───────────────────────────────────────────────────────────

def sites_to_geojson(sites: List[DetectedSite]) -> Dict:
    features = []
    for site in sites:
        d = asdict(site) if hasattr(site, "__dataclass_fields__") else site
        bbox = d.get("bbox", [d["lon"] - 0.001, d["lat"] - 0.001,
                               d["lon"] + 0.001, d["lat"] + 0.001])
        lon_min, lat_min, lon_max, lat_max = bbox
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [lon_min, lat_min], [lon_max, lat_min],
                [lon_max, lat_max], [lon_min, lat_max],
                [lon_min, lat_min],
            ]],
        }
        sev = d.get("severity", "LOW")
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "site_id":     d["site_id"],
                "severity":    sev,
                "status":      d.get("status", "NEW"),
                "area_m2":     d["area_m2"],
                "area_ha":     round(d["area_m2"] / 10_000, 3),
                "confidence":  d["confidence"],
                "mci":         round(d.get("mci", 0.0), 3),
                "date_detected": d["date_detected"],
                "consecutive_periods": d.get("consecutive_periods", 1),
                "color":       SEVERITY_COLOURS.get(sev, "#95a5a6"),
                "lat":         d["lat"],
                "lon":         d["lon"],
            },
        })

    return {
        "type": "FeatureCollection",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total_sites": len(features),
        "features": features,
    }


# ─── Full alert report ────────────────────────────────────────────────────────

def generate_alert_report(
    sites: List[DetectedSite],
    scene_meta: Dict,
    save: bool = True,
) -> Dict:
    """
    Generate a structured alert report.

    Returns a dict with summary stats + full GeoJSON + per-site details.
    """
    total   = len(sites)
    by_sev  = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for s in sites:
        d = asdict(s) if hasattr(s, "__dataclass_fields__") else s
        by_sev[d.get("severity", "LOW")] += 1

    geojson = sites_to_geojson(sites)

    report = {
        "report_id": f"RPT_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "scene_meta": scene_meta,
        "summary": {
            "total_sites_detected": total,
            "by_severity": by_sev,
            "total_area_ha": round(
                sum((asdict(s) if hasattr(s, "__dataclass_fields__") else s)["area_m2"]
                    for s in sites) / 10_000, 2
            ),
        },
        "geojson": geojson,
        "sites": [
            asdict(s) if hasattr(s, "__dataclass_fields__") else s
            for s in sites
        ],
        "instructions": {
            "view_map": "Open the web interface and click 'View Map' to see detections on Leaflet",
            "download_geojson": "GET /api/alerts/geojson to download GeoJSON for QGIS/ArcGIS",
            "api_docs": "Visit /docs for full Swagger API documentation",
        },
    }

    if save:
        ALERT_DIR.mkdir(parents=True, exist_ok=True)
        path = ALERT_DIR / f"{report['report_id']}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"Alert report saved → {path}")

    return report


# ─── Email-style text summary ─────────────────────────────────────────────────

def format_email_body(report: Dict) -> str:
    s = report["summary"]
    lines = [
        f"ILLEGAL SAND MINING DETECTION ALERT",
        f"Report ID : {report['report_id']}",
        f"Generated : {report['generated_at']}",
        f"",
        f"SUMMARY",
        f"  Total sites detected : {s['total_sites_detected']}",
        f"  Total affected area  : {s['total_area_ha']} ha",
        f"",
        f"BY SEVERITY",
        f"  🟣 CRITICAL : {s['by_severity']['CRITICAL']}",
        f"  🔴 HIGH     : {s['by_severity']['HIGH']}",
        f"  🟠 MEDIUM   : {s['by_severity']['MEDIUM']}",
        f"  🟢 LOW      : {s['by_severity']['LOW']}",
        f"",
        f"TOP SITES",
    ]
    sites = sorted(
        report["sites"],
        key=lambda x: x.get("area_m2", 0),
        reverse=True,
    )[:5]
    for i, site in enumerate(sites, 1):
        lines.append(
            f"  {i}. [{site.get('severity','?')}] {site['site_id']} "
            f"| {site['area_m2']:.0f} m² | "
            f"Lat {site['lat']:.4f}, Lon {site['lon']:.4f}"
        )

    lines += [
        f"",
        f"Download full GeoJSON for GIS import:",
        f"  GET /api/alerts/geojson",
        f"",
        f"View on interactive map:",
        f"  http://localhost:8000/",
    ]
    return "\n".join(lines)
