"""
Image preprocessing pipeline:
  - Radiometric normalisation (histogram matching)
  - Spectral index computation: NDWI, NDVI, Sand Index, MNDWI, BSI
  - River corridor masking (500 m buffer around water)
  - Patch tiling (256×256, 64-pixel overlap)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

# ─── Band layout (Sentinel-2 L2A order used throughout the project) ───────────
# Index:  0=B2(Blue)  1=B3(Green)  2=B4(Red)  3=B8(NIR)  4=B11(SWIR1)  5=B12(SWIR2)
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]


# ─── Spectral indices ─────────────────────────────────────────────────────────

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) > 1e-10, a / b, 0.0)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters 1996)."""
    return _safe_div(green - nir, green + nir)


def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified NDWI — better for urban/sediment-laden water."""
    return _safe_div(green - swir1, green + swir1)


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    return _safe_div(nir - red, nir + red)


def compute_sand_index(swir1: np.ndarray, green: np.ndarray) -> np.ndarray:
    """
    Custom Sand Index: high positive values = dry exposed sand;
    negative values = water / dense vegetation.
    """
    return _safe_div(swir1, green + 1e-6)


def compute_bsi(blue: np.ndarray, red: np.ndarray,
                nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Bare Soil Index."""
    num = (swir1 + red) - (nir + blue)
    den = (swir1 + red) + (nir + blue)
    return _safe_div(num, den)


# ─── Water mask & river corridor ──────────────────────────────────────────────

def build_water_mask(ndwi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return (ndwi > threshold).astype(np.uint8)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Binary dilation using a square structuring element (pure NumPy)."""
    from scipy.ndimage import binary_dilation
    struct = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=bool)
    return binary_dilation(mask.astype(bool), structure=struct).astype(np.uint8)


def build_river_corridor(ndwi: np.ndarray,
                         buffer_pixels: int = 50) -> np.ndarray:
    """
    Return binary mask covering active water + 500 m buffer.
    At Sentinel-2 10 m resolution, 500 m ≈ 50 pixels.
    """
    water = build_water_mask(ndwi)
    return dilate_mask(water, buffer_pixels)


# ─── Radiometric normalisation ────────────────────────────────────────────────

def histogram_match_band(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Histogram matching of a single band (src → ref distribution).
    Works on float arrays in [0, 1].
    """
    src_flat = src.ravel()
    ref_flat = ref.ravel()

    # CDFs
    src_vals, src_counts = np.unique(src_flat, return_counts=True)
    ref_vals, ref_counts = np.unique(ref_flat, return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(float)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_counts).astype(float)
    ref_cdf /= ref_cdf[-1]

    interp = np.interp(src_cdf, ref_cdf, ref_vals)
    lookup = np.interp(src_flat, src_vals, interp)
    return lookup.reshape(src.shape).astype(np.float32)


def normalise_patch(patch: np.ndarray,
                    mean: Optional[np.ndarray] = None,
                    std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Z-score normalisation per channel.
    If mean/std not provided, compute from the patch itself.
    """
    if mean is None:
        mean = patch.mean(axis=(1, 2), keepdims=True)
    if std is None:
        std = patch.std(axis=(1, 2), keepdims=True) + 1e-8
    return (patch - mean) / std


# ─── Full 7-channel stack builder ─────────────────────────────────────────────

def build_7ch_stack(bands: np.ndarray) -> np.ndarray:
    """
    bands: (6, H, W) float32 in [0, 1] — Blue, Green, Red, NIR, SWIR1, SWIR2
    Returns (7, H, W) — above 6 bands + Sand Index as channel 6.
    """
    blue, green, red, nir, swir1, swir2 = [bands[i] for i in range(6)]
    si = compute_sand_index(swir1, green)
    si_norm = np.clip((si - si.min()) / (si.ptp() + 1e-8), 0, 1).astype(np.float32)
    return np.concatenate([bands, si_norm[np.newaxis]], axis=0)   # (7, H, W)


# ─── Patch tiling ─────────────────────────────────────────────────────────────

def tile_image(image: np.ndarray,
               patch_size: int = 256,
               overlap: int = 64,
               corridor_mask: Optional[np.ndarray] = None
               ) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Tile (C, H, W) image into overlapping patches.
    Returns patches and their metadata (row_start, col_start, row_end, col_end).
    Skips patches with <10% river corridor coverage when mask provided.
    """
    _, H, W = image.shape
    stride = patch_size - overlap
    patches, meta = [], []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            r2, c2 = r + patch_size, c + patch_size
            if corridor_mask is not None:
                coverage = corridor_mask[r:r2, c:c2].mean()
                if coverage < 0.10:
                    continue
            patches.append(image[:, r:r2, c:c2])
            meta.append({"r0": r, "c0": c, "r1": r2, "c1": c2})

    return patches, meta


def reconstruct_from_patches(patches: List[np.ndarray],
                              meta: List[Dict],
                              H: int, W: int) -> np.ndarray:
    """Reconstruct probability map by averaging overlapping patches."""
    accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for patch, m in zip(patches, meta):
        p = patch.squeeze()
        accum[m["r0"]:m["r1"], m["c0"]:m["c1"]] += p
        count[m["r0"]:m["r1"], m["c0"]:m["c1"]] += 1
    return np.where(count > 0, accum / count, 0.0)


# ─── Main preprocessing entry point ───────────────────────────────────────────

class Preprocessor:
    """Stateless preprocessing helper used by both training and inference."""

    # Dataset-level statistics (will be overridden when computed from real data)
    MEAN = np.array([0.0785, 0.0989, 0.0924, 0.2615, 0.1813, 0.1182, 1.8342],
                    dtype=np.float32)[:, None, None]
    STD  = np.array([0.0412, 0.0517, 0.0631, 0.0812, 0.0745, 0.0621, 0.9451],
                    dtype=np.float32)[:, None, None]

    @classmethod
    def process(cls,
                bands: np.ndarray,
                patch_size: int = 256,
                overlap: int = 64,
                buffer_pixels: int = 50
                ) -> Tuple[List[np.ndarray], List[Dict], np.ndarray]:
        """
        Full preprocessing pipeline.
        bands: (6, H, W) float32 [0–1]
        Returns (normalised_patches, meta, corridor_mask)
        """
        _, green, _, nir, swir1, _ = [bands[i] for i in range(6)]
        ndwi = compute_ndwi(green, nir)
        corridor = build_river_corridor(ndwi, buffer_pixels)

        stack = build_7ch_stack(bands)                # (7, H, W)
        stack_norm = (stack - cls.MEAN) / cls.STD     # z-score

        patches, meta = tile_image(stack_norm, patch_size, overlap, corridor)
        return patches, meta, corridor
