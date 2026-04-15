"""
Basic tests for Sand Mining Detection pipeline.
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import numpy as np
import torch
import pytest


# ─── Preprocessing tests ──────────────────────────────────────────────────────

def test_spectral_indices():
    from modules.preprocessing import compute_ndwi, compute_ndvi, compute_sand_index

    green = np.array([[0.1, 0.2]], dtype=np.float32)
    nir   = np.array([[0.3, 0.05]], dtype=np.float32)
    ndwi  = compute_ndwi(green, nir)
    # NDWI = (green-nir)/(green+nir)
    expected0 = (0.1 - 0.3) / (0.1 + 0.3)
    assert abs(ndwi[0, 0] - expected0) < 1e-5

    red  = np.array([[0.08]], dtype=np.float32)
    ndvi = compute_ndvi(nir[:, :1], red)
    expected_ndvi = (0.3 - 0.08) / (0.3 + 0.08)
    assert abs(ndvi[0, 0] - expected_ndvi) < 1e-5


def test_build_7ch_stack():
    from modules.preprocessing import build_7ch_stack
    bands = np.random.rand(6, 64, 64).astype(np.float32)
    stack = build_7ch_stack(bands)
    assert stack.shape == (7, 64, 64)
    assert stack.dtype == np.float32


def test_river_corridor():
    from modules.preprocessing import build_water_mask, build_river_corridor
    green = np.zeros((100, 100), np.float32)
    nir   = np.ones( (100, 100), np.float32)
    # NDWI will be negative → no water mask
    from modules.preprocessing import compute_ndwi
    ndwi = compute_ndwi(green, nir)
    water = build_water_mask(ndwi)
    assert water.sum() == 0


def test_patch_tiling():
    from modules.preprocessing import tile_image
    img = np.random.rand(7, 512, 512).astype(np.float32)
    patches, meta = tile_image(img, patch_size=256, overlap=64)
    assert len(patches) > 0
    assert patches[0].shape == (7, 256, 256)


def test_reconstruct():
    from modules.preprocessing import tile_image, reconstruct_from_patches
    img = np.ones((7, 256, 256), np.float32)
    patches, meta = tile_image(img, patch_size=256, overlap=0)
    prob_patches = [np.ones((256, 256), np.float32) * 0.8 for _ in patches]
    recon = reconstruct_from_patches(prob_patches, meta, 256, 256)
    assert abs(recon.mean() - 0.8) < 0.01


# ─── Model tests ──────────────────────────────────────────────────────────────

def test_model_forward():
    from modules.model import SandMiningUNet
    model = SandMiningUNet(in_channels=7, pretrained=False)
    x = torch.zeros(2, 7, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)
    assert (out >= 0).all() and (out <= 1).all()


def test_loss():
    from modules.model import DiceBCELoss, SandMiningUNet
    model = SandMiningUNet(in_channels=7, pretrained=False)
    criterion = DiceBCELoss()
    x = torch.zeros(2, 7, 256, 256)
    y = torch.zeros(2, 1, 256, 256)
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, y)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_metrics():
    from modules.model import compute_metrics
    pred = torch.tensor([[[0.9, 0.1, 0.8, 0.2]]])
    tgt  = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    m = compute_metrics(pred, tgt)
    assert m["precision"] == pytest.approx(1.0, abs=0.01)
    assert m["recall"]    == pytest.approx(1.0, abs=0.01)


# ─── Data acquisition tests ───────────────────────────────────────────────────

def test_demo_scene():
    from modules.data_acquisition import acquire_scene
    scene = acquire_scene(
        bbox=(83.2, 26.5, 83.6, 26.9),
        date_range=("2024-01-01", "2024-01-31"),
        source="demo",
    )
    assert "bands" in scene
    assert scene["bands"].shape[0] == 6
    assert scene["bands"].dtype == np.float32
    assert scene["bands"].min() >= 0.0
    assert scene["bands"].max() <= 1.0


# ─── Change detection tests ───────────────────────────────────────────────────

def test_mci():
    from modules.change_detection import compute_mci
    # Empty mask
    assert compute_mci(np.zeros((50, 50), np.uint8)) == 0.0
    # Square → MCI ~1.13 (close to 1 = simple shape)
    sq = np.zeros((50, 50), np.uint8)
    sq[10:30, 10:30] = 1
    mci = compute_mci(sq)
    assert 1.0 < mci < 1.5


def test_process_detections():
    from modules.change_detection import process_detections
    detections = [
        {"lat": 26.7, "lon": 83.3, "area_m2": 50000, "confidence": 0.85,
         "bbox": [83.29, 26.69, 83.31, 26.71]},
    ]
    sites = process_detections(detections, "2024-01-15")
    assert len(sites) == 1
    assert sites[0].severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# ─── Inference tests ──────────────────────────────────────────────────────────

def test_full_inference_pipeline():
    from modules.data_acquisition import acquire_scene
    from modules.model import SandMiningUNet
    from modules.inference import run_inference

    scene = acquire_scene(
        bbox=(83.2, 26.5, 83.6, 26.9),
        date_range=("2024-01-01", "2024-01-31"),
        source="demo",
    )
    model = SandMiningUNet(in_channels=7, pretrained=False)
    result = run_inference(scene, model=model, patch_size=256, overlap=64)

    assert "detections"  in result
    assert "prob_map"    in result
    assert "binary_map"  in result
    assert isinstance(result["detections"], list)
    assert result["prob_map"].ndim == 2
