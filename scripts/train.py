"""
Training script for the Sand Mining U-Net.

Usage
-----
# Using demo/synthetic data (no downloads required):
python scripts/train.py --epochs 5 --demo

# Using real labeled data:
python scripts/train.py --data-dir data/processed --epochs 50

Open-source datasets that can be used as real training data:
  1. Global Surface Mining (Maus et al. 2020)
     https://doi.org/10.1038/s41597-020-00624-w
  2. MineSat Dataset (Raza et al. 2022)
     https://github.com/MarcCoru/meteor (contact authors)
  3. EuroSAT (land cover, for pre-training)
     https://github.com/phelber/EuroSAT
  4. LandCover.ai (high-res labeled imagery)
     https://landcover.ai/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Ensure parent package importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from modules.model import SandMiningUNet, DiceBCELoss, compute_metrics
from modules.preprocessing import Preprocessor, build_7ch_stack

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("train")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = Path(__file__).parent.parent / "models" / "unet_weights.pth"


# ─── Synthetic dataset (for demo / CI testing) ────────────────────────────────

class SyntheticSandDataset(Dataset):
    """
    Generates random synthetic 7-channel patches with simulated mining masks.
    Adequate for smoke-testing the training loop; not for real deployment.
    """

    def __init__(self, n_samples: int = 500, patch_size: int = 256):
        self.n = n_samples
        self.ps = patch_size
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        ps = self.ps

        # 6-band background (vegetation / bare soil)
        bands = np.zeros((6, ps, ps), dtype=np.float32)
        bands[0] = rng.uniform(0.04, 0.08, (ps, ps))
        bands[1] = rng.uniform(0.07, 0.12, (ps, ps))
        bands[2] = rng.uniform(0.06, 0.11, (ps, ps))
        bands[3] = rng.uniform(0.22, 0.38, (ps, ps))
        bands[4] = rng.uniform(0.10, 0.20, (ps, ps))
        bands[5] = rng.uniform(0.05, 0.12, (ps, ps))

        mask = np.zeros((ps, ps), dtype=np.float32)

        # Randomly add 0–3 mining patches
        n_mines = rng.integers(0, 4)
        for _ in range(n_mines):
            r = rng.integers(20, ps - 60)
            c = rng.integers(20, ps - 60)
            h = rng.integers(20, 60)
            w = rng.integers(20, 80)
            bands[:, r:r+h, c:c+w] = [0.28, 0.32, 0.30, 0.15, 0.42, 0.28]
            mask[r:r+h, c:c+w] = 1.0

        # Add noise
        bands += rng.normal(0, 0.005, bands.shape).astype(np.float32)
        bands = np.clip(bands, 0.0, 1.0)

        # Build 7-channel stack and normalise
        stack = build_7ch_stack(bands)
        stack = (stack - Preprocessor.MEAN) / Preprocessor.STD

        return (
            torch.tensor(stack, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        )


# ─── Real disk dataset ────────────────────────────────────────────────────────

class DiskPatchDataset(Dataset):
    """
    Load pre-saved .npy patch pairs from data/processed/.
    Expected layout:
      data/processed/
        images/   *.npy  shape (7, 256, 256) float32
        masks/    *.npy  shape (1, 256, 256) float32  values {0, 1}
    """

    def __init__(self, data_dir: Path):
        self.img_paths = sorted((data_dir / "images").glob("*.npy"))
        self.msk_paths = sorted((data_dir / "masks").glob("*.npy"))
        assert len(self.img_paths) == len(self.msk_paths), \
            "Mismatch between images and masks"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        msk = np.load(self.msk_paths[idx]).astype(np.float32)
        return torch.tensor(img), torch.tensor(msk)


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {DEVICE}")

    # Dataset
    if args.demo or not Path(args.data_dir).exists():
        logger.info("Using synthetic dataset for training")
        dataset = SyntheticSandDataset(n_samples=args.n_samples)
    else:
        logger.info(f"Loading real dataset from {args.data_dir}")
        dataset = DiskPatchDataset(Path(args.data_dir))

    val_size  = max(1, int(0.15 * len(dataset)))
    trn_size  = len(dataset) - val_size
    trn_ds, val_ds = random_split(dataset, [trn_size, val_size])

    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=0, pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    model = SandMiningUNet(in_channels=7, pretrained=not args.no_pretrain)
    model.to(DEVICE)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # — Train —
        model.train()
        trn_loss = 0.0
        for imgs, masks in trn_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trn_loss += loss.item() * imgs.size(0)
        trn_loss /= trn_size

        # — Validate —
        model.eval()
        val_loss = 0.0
        val_metrics = {"precision": 0, "recall": 0, "f1": 0, "iou": 0}
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item() * imgs.size(0)
                m = compute_metrics(preds.cpu(), masks.cpu())
                for k in val_metrics:
                    val_metrics[k] += m[k]
        val_loss /= val_size
        n_batches = len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= n_batches

        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"trn={trn_loss:.4f} | val={val_loss:.4f} | "
            f"F1={val_metrics['f1']:.3f} | IoU={val_metrics['iou']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {MODEL_SAVE_PATH}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sand Mining U-Net")
    parser.add_argument("--data-dir",    default="data/processed",
                        help="Root dir of processed patches (images/ + masks/)")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--n-samples",   type=int, default=500,
                        help="Number of synthetic samples (--demo mode only)")
    parser.add_argument("--demo",        action="store_true",
                        help="Use synthetic data (no real images needed)")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Skip ImageNet pretrained weights")
    args = parser.parse_args()
    train(args)
