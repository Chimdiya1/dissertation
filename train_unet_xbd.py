import os
import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Paths (EDIT)
# ----------------------------
TRAIN_CSV = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/tiles_256/train_tiles_balanced.csv"
VAL_CSV   = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/tiles_256/val_tiles.csv"

OUT_DIR = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/checkpoints_unet"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Training config
# ----------------------------
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
NUM_WORKERS = 2
SEED = 42

# If True, compute class weights from training masks (slower first run, better weights)
COMPUTE_CLASS_WEIGHTS = True

# Loss blend
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
DICE_EXCLUDE_BG = True  # compute Dice on classes 1..4 only

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Step 9 augmentation (reuse)
# ----------------------------
def augment_tile(x6: np.ndarray, mask: np.ndarray, rng: random.Random):
    # spatial
    if rng.random() < 0.5:
        x6 = np.flip(x6, axis=1); mask = np.flip(mask, axis=1)
    if rng.random() < 0.5:
        x6 = np.flip(x6, axis=0); mask = np.flip(mask, axis=0)
    k = rng.randint(0, 3)
    if k:
        x6 = np.rot90(x6, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

    # photometric (input only)
    if x6.dtype != np.float32 and x6.dtype != np.float64:
        x6f = x6.astype(np.float32) / 255.0
    else:
        x6f = x6.astype(np.float32)

    if rng.random() < 0.8:
        x6f = x6f + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.8:
        c = rng.uniform(0.9, 1.1)
        mean = x6f.mean(axis=(0, 1), keepdims=True)
        x6f = (x6f - mean) * c + mean

    x6f = np.clip(x6f, 0.0, 1.0).astype(np.float32)
    return x6f, mask.astype(np.uint8)


class XBDTilesDataset(Dataset):
    def __init__(self, csv_path: str, augment: bool = False, seed: int = 42):
        self.augment = augment
        self.seed = seed

        self.rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        if not self.rows:
            raise ValueError(f"No rows found in {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        x6 = np.load(r["sixch_tile_path"])  # (256,256,6)
        mask = np.array(Image.open(r["mask_tile_path"]), dtype=np.uint8)  # (256,256)

        if self.augment:
            rng = random.Random(self.seed * 10_000 + idx)
            x6, mask = augment_tile(x6, mask, rng)
        else:
            if x6.dtype != np.float32 and x6.dtype != np.float64:
                x6 = x6.astype(np.float32) / 255.0
            else:
                x6 = x6.astype(np.float32)

        x_t = torch.from_numpy(x6).permute(2, 0, 1)  # (6,H,W)
        y_t = torch.from_numpy(mask.astype(np.int64)) # (H,W)
        return x_t, y_t


# ----------------------------
# Simple U-Net (pure PyTorch)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=5, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)  # logits (N,C,H,W)


# ----------------------------
# Loss: Weighted CE + Soft Dice
# ----------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int, exclude_bg: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.exclude_bg = exclude_bg
        self.eps = eps

    def forward(self, logits, target):
        # logits: (N,C,H,W), target: (N,H,W)
        probs = F.softmax(logits, dim=1)  # (N,C,H,W)
        target_1h = F.one_hot(target, num_classes=self.num_classes).permute(0,3,1,2).float()

        if self.exclude_bg:
            probs = probs[:, 1:, :, :]
            target_1h = target_1h[:, 1:, :, :]

        dims = (0, 2, 3)  # sum over N,H,W
        intersection = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


def compute_class_weights_from_train(train_csv: str, num_classes: int = 5):
    counts = np.zeros(num_classes, dtype=np.int64)
    with open(train_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = np.array(Image.open(r["mask_tile_path"]), dtype=np.uint8)
            for c in range(num_classes):
                counts[c] += int((m == c).sum())

    # Inverse frequency with smoothing; keep background smaller
    freqs = counts / max(counts.sum(), 1)
    weights = 1.0 / (np.log(1.02 + freqs))  # stable
    weights = weights / weights.mean()
    # Optional: reduce background influence
    weights[0] *= 0.3
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------
# Metrics: IoU (classes 1..4)
# ----------------------------
@torch.no_grad()
def mean_iou(pred, target, num_classes=5, exclude_bg=True):
    # pred/target: (N,H,W) int64
    cls_range = range(1, num_classes) if exclude_bg else range(num_classes)
    ious = []
    for c in cls_range:
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = CE_WEIGHT * ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, ce_loss, dice_loss):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = CE_WEIGHT * ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)
        total_loss += loss.item()

        pred = torch.argmax(logits, dim=1)  # (N,H,W)
        total_iou += mean_iou(pred, y, num_classes=NUM_CLASSES, exclude_bg=True)

    return total_loss / max(len(loader), 1), total_iou / max(len(loader), 1)


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    train_ds = XBDTilesDataset(TRAIN_CSV, augment=True, seed=SEED)
    val_ds = XBDTilesDataset(VAL_CSV, augment=False, seed=SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    if COMPUTE_CLASS_WEIGHTS:
        print("Computing class weights from training masks...")
        class_w = compute_class_weights_from_train(TRAIN_CSV, NUM_CLASSES).to(DEVICE)
    else:
        # reasonable defaults if you don’t want to scan masks
        class_w = torch.tensor([0.2, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=DEVICE)

    print("Class weights:", class_w.detach().cpu().numpy())

    model = UNet(in_channels=6, num_classes=NUM_CLASSES, base=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    ce_loss = nn.CrossEntropyLoss(weight=class_w)
    dice_loss = SoftDiceLoss(num_classes=NUM_CLASSES, exclude_bg=DICE_EXCLUDE_BG)

    best_val_iou = -1.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss)
        va_loss, va_iou = validate(model, val_loader, ce_loss, dice_loss)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_mIoU(1-4)={va_iou:.4f}")

        # Save best
        if va_iou > best_val_iou:
            best_val_iou = va_iou
            ckpt_path = os.path.join(OUT_DIR, "best_unet.pth")
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_iou": best_val_iou,
                "class_weights": class_w.detach().cpu().numpy(),
            }, ckpt_path)
            print("  Saved best ->", ckpt_path)

    # Save last
    last_path = os.path.join(OUT_DIR, "last_unet.pth")
    torch.save({"model_state": model.state_dict()}, last_path)
    print("Saved last ->", last_path)


if __name__ == "__main__":
    main()
