import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet_model import UNet
from train_data import XBDTilesDataset
from train_losses import SoftDiceLoss
from train_metrics import mean_iou

TRAIN_CSV = "/content/train/tiles_256/train_tiles_balanced.csv"
VAL_CSV   = "/content/train/tiles_256/val_tiles.csv"

OUT_DIR = "/content/train/checkpoints_unet"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
NUM_WORKERS = 2

CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
DICE_EXCLUDE_BG = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
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
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = CE_WEIGHT * ce_loss(logits, y) + DICE_WEIGHT * dice_loss(logits, y)
        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        total_iou += mean_iou(pred, y, num_classes=NUM_CLASSES, exclude_bg=True)
    return total_loss / max(len(loader), 1), total_iou / max(len(loader), 1)

def main():
    class_w = torch.tensor([0.2, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=DEVICE)

    train_ds = XBDTilesDataset(TRAIN_CSV, augment=True, seed=42)
    val_ds   = XBDTilesDataset(VAL_CSV, augment=False, seed=42)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(in_channels=6, num_classes=NUM_CLASSES, base=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    ce_loss = nn.CrossEntropyLoss(weight=class_w)
    dice_loss = SoftDiceLoss(num_classes=NUM_CLASSES, exclude_bg=DICE_EXCLUDE_BG)

    best_val_iou = -1.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss)
        va_loss, va_iou = validate(model, val_loader, ce_loss, dice_loss)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_mIoU(1-4)={va_iou:.4f}")

        if va_iou > best_val_iou:
            best_val_iou = va_iou
            ckpt_path = os.path.join(OUT_DIR, "best_unet.pth")
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_val_iou": best_val_iou}, ckpt_path)
            print("  Saved best ->", ckpt_path)

    last_path = os.path.join(OUT_DIR, "last_unet.pth")
    torch.save({"model_state": model.state_dict()}, last_path)
    print("Saved last ->", last_path)

if __name__ == "__main__":
    main()
