"""
U-Net with ResNet-50 encoder for sand mining semantic segmentation.
Accepts 7-channel input: RGB, NIR, SWIR1, SWIR2, Sand Index.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ─── Decoder block ────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── U-Net ────────────────────────────────────────────────────────────────────

class SandMiningUNet(nn.Module):
    """
    U-Net with ResNet-50 encoder for 7-channel multispectral input.
    Output: single-channel sigmoid probability map (1 = mining activity).
    """

    def __init__(self, in_channels: int = 7, pretrained: bool = True):
        super().__init__()

        # Load backbone
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        # Adapt first conv to in_channels (replicate RGB weights, scale by 3/in_channels)
        orig_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            # tile RGB weights across extra channels, then rescale
            rep = (in_channels // 3) + 1
            tiled = orig_conv.weight.repeat(1, rep, 1, 1)[:, :in_channels, :, :]
            new_conv.weight.copy_(tiled * (3.0 / in_channels))
        backbone.conv1 = new_conv

        # Encoder stages
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2  64ch
        self.pool  = backbone.maxpool                                             # /4
        self.enc1  = backbone.layer1   # /4   256ch
        self.enc2  = backbone.layer2   # /8   512ch
        self.enc3  = backbone.layer3   # /16  1024ch
        self.enc4  = backbone.layer4   # /32  2048ch

        # Decoder
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512,  512,  256)
        self.dec2 = DecoderBlock(256,  256,  128)
        self.dec1 = DecoderBlock(128,   64,   64)
        self.dec0 = DecoderBlock(64,     0,   32)   # no skip (back to full res)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)           # H/2
        p  = self.pool(e0)          # H/4
        e1 = self.enc1(p)           # H/4
        e2 = self.enc2(e1)          # H/8
        e3 = self.enc3(e2)          # H/16
        e4 = self.enc4(e3)          # H/32

        # Decoder with skip connections
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1)          # no skip

        return self.head(d0)        # (B, 1, H, W)  values in [0,1]


# ─── Dice + BCE loss ──────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_val = self.bce(pred, target)
        pred_f  = pred.view(-1)
        tgt_f   = target.view(-1)
        inter   = (pred_f * tgt_f).sum()
        dice    = 1 - (2.0 * inter + self.smooth) / (
            pred_f.sum() + tgt_f.sum() + self.smooth
        )
        return bce_val + dice


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_iou(pred_bin: torch.Tensor, target: torch.Tensor) -> float:
    inter = (pred_bin & target.bool()).float().sum()
    union = (pred_bin | target.bool()).float().sum()
    return (inter / (union + 1e-8)).item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    pred_bin = (pred > threshold).bool()
    tgt_bin  = target.bool()
    tp = (pred_bin & tgt_bin).float().sum()
    fp = (pred_bin & ~tgt_bin).float().sum()
    fn = (~pred_bin & tgt_bin).float().sum()
    precision = (tp / (tp + fp + 1e-8)).item()
    recall    = (tp / (tp + fn + 1e-8)).item()
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = compute_iou(pred_bin, target)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
