"""
Inference engine — ties together preprocessing → model → post-processing.

Produces a list of detected mining polygons with GPS coordinates,
area estimates, and confidence scores.
"""

from __future__ import annotations
import logging
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .preprocessing import Preprocessor, reconstruct_from_patches
from .model import SandMiningUNet

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parents[2] / "models" / "unet_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.50          # binary classification cutoff
MIN_AREA_PX = 50          # minimum connected component size in pixels (≈5 000 m²)


# ─── Pixel → geographic coordinate helpers ────────────────────────────────────

def _pixel_to_latlon(row: int, col: int,
                     bbox: Tuple[float, float, float, float],
                     H: int, W: int) -> Tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = bbox
    lat = lat_max - (row / H) * (lat_max - lat_min)
    lon = lon_min + (col / W) * (lon_max - lon_min)
    return lat, lon


def _pixel_area_m2(bbox: Tuple[float, float, float, float],
                   H: int, W: int) -> float:
    """Area of a single pixel in m² (rough equatorial approximation)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_c = (lat_min + lat_max) / 2
    deg_per_m_lat = 1 / 111_320
    deg_per_m_lon = 1 / (111_320 * math.cos(math.radians(lat_c)))
    px_h = (lat_max - lat_min) / H / deg_per_m_lat
    px_w = (lon_max - lon_min) / W / deg_per_m_lon
    return px_h * px_w


# ─── Connected component labelling (pure NumPy) ───────────────────────────────

def _label_components(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """Simple flood-fill labelling — avoids requiring scipy at runtime."""
    try:
        from scipy.ndimage import label as scipy_label
        return scipy_label(binary)
    except ImportError:
        pass

    H, W = binary.shape
    labels = np.zeros_like(binary, dtype=np.int32)
    current = 0
    for i in range(H):
        for j in range(W):
            if binary[i, j] and labels[i, j] == 0:
                current += 1
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= H or c < 0 or c >= W:
                        continue
                    if not binary[r, c] or labels[r, c]:
                        continue
                    labels[r, c] = current
                    stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
    return labels, current


# ─── Model loading ────────────────────────────────────────────────────────────

_model_cache: Optional[SandMiningUNet] = None


def load_model(weights_path: Optional[Path] = None) -> SandMiningUNet:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model = SandMiningUNet(in_channels=7, pretrained=False)

    if weights_path is None:
        weights_path = MODEL_PATH

    if weights_path.exists():
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning(
            "No trained weights found — using random initialisation.\n"
            "Run scripts/train.py or download pre-trained weights.\n"
            "Detection results will be meaningless until weights are trained."
        )

    model.to(DEVICE).eval()
    _model_cache = model
    return model


# ─── Single-patch inference ───────────────────────────────────────────────────

@torch.no_grad()
def infer_patches(patches: List[np.ndarray],
                  model: SandMiningUNet,
                  batch_size: int = 8) -> List[np.ndarray]:
    """Run model on a list of (7,256,256) patches, return probability maps."""
    results = []
    for i in range(0, len(patches), batch_size):
        batch = patches[i : i + batch_size]
        tensor = torch.tensor(np.stack(batch), dtype=torch.float32).to(DEVICE)
        probs = model(tensor).squeeze(1).cpu().numpy()   # (B, 256, 256)
        results.extend([probs[j] for j in range(len(batch))])
    return results


# ─── Post-processing ──────────────────────────────────────────────────────────

def extract_detections(prob_map: np.ndarray,
                       bbox: Tuple[float, float, float, float],
                       threshold: float = THRESHOLD,
                       min_area_px: int = MIN_AREA_PX) -> List[Dict]:
    """
    Convert probability map to list of detection dicts.
    Each dict: {lat, lon, area_m2, confidence, bbox_geo}
    """
    H, W = prob_map.shape
    binary = (prob_map > threshold).astype(np.uint8)
    labels, n_components = _label_components(binary)
    px_area = _pixel_area_m2(bbox, H, W)
    detections = []

    for comp_id in range(1, n_components + 1):
        comp_mask = labels == comp_id
        px_count = comp_mask.sum()
        if px_count < min_area_px:
            continue

        # Centre of mass
        rows, cols = np.where(comp_mask)
        cr, cc = rows.mean(), cols.mean()
        lat, lon = _pixel_to_latlon(int(cr), int(cc), bbox, H, W)

        # Bounding box in geo-coords
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        lat_max, lon_min = _pixel_to_latlon(r0, c0, bbox, H, W)
        lat_min, lon_max = _pixel_to_latlon(r1, c1, bbox, H, W)

        conf = float(prob_map[comp_mask].mean())
        area = float(px_count * px_area)

        detections.append({
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "area_m2": round(area, 1),
            "confidence": round(conf, 4),
            "bbox": [
                round(lon_min, 6), round(lat_min, 6),
                round(lon_max, 6), round(lat_max, 6),
            ],
        })

    return detections


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_inference(scene: Dict,
                  model: Optional[SandMiningUNet] = None,
                  patch_size: int = 256,
                  overlap: int = 64,
                  threshold: float = THRESHOLD) -> Dict:
    """
    Run the full inference pipeline on a scene dict.

    Parameters
    ----------
    scene : output of data_acquisition.acquire_scene()
    model : loaded SandMiningUNet (loaded automatically if None)

    Returns
    -------
    dict with keys:
      "detections"   : list of detection dicts
      "prob_map"     : (H, W) float32 probability map
      "binary_map"   : (H, W) uint8 binary map
      "meta"         : scene metadata
    """
    if model is None:
        model = load_model()

    bands = scene["bands"]   # (6, H, W)
    bbox  = scene["meta"]["bbox"]

    # Preprocessing
    patches, meta_list, corridor = Preprocessor.process(
        bands, patch_size=patch_size, overlap=overlap
    )

    _, H, W = bands.shape

    if not patches:
        logger.warning("No river-corridor patches found in this scene.")
        return {
            "detections": [],
            "prob_map": np.zeros((H, W), np.float32),
            "binary_map": np.zeros((H, W), np.uint8),
            "meta": scene["meta"],
        }

    # Inference
    prob_patches = infer_patches(patches, model)

    # Reconstruction
    prob_map = reconstruct_from_patches(prob_patches, meta_list, H, W)
    binary_map = (prob_map > threshold).astype(np.uint8)

    # Extract detections
    detections = extract_detections(prob_map, bbox, threshold)

    logger.info(
        f"Inference complete: {len(detections)} mining site(s) detected "
        f"in {len(patches)} patches"
    )

    return {
        "detections": detections,
        "prob_map": prob_map,
        "binary_map": binary_map,
        "meta": scene["meta"],
    }
