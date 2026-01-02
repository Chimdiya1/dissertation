import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet_old import UNet
from dataloader import PrePostDataset  # update to include masks
import os
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Dataset
# -------------------------
train_dataset = PrePostDataset(
    pre_dir="/content/data/train/pre",
    post_dir="/content/data/train/post",
    mask_dir="/content/data/train/masks",
    augment=True
)

val_dataset = PrePostDataset(
    pre_dir="/content/data/val/pre",
    post_dir="/content/data/val/post",
    mask_dir="/content/data/val/masks",
    augment=False  # no augmentation on validation!
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=0)

# -------------------------
# Model, Loss, Optimizer
# -------------------------
model = UNet(in_channels=6, out_channels=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

results = {"epochs": [], "iou": [], "dice": []}
metrics_path = "/content/checkpoints/metrics.json"

# -------------------------
# Training Loop (5 epochs)
# -------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        mask = batch["mask"].to(device)

        inp = torch.cat([pre, post], dim=1)  # 6 channels

        pred = model(inp)
        loss = loss_fn(pred, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            mask = batch["mask"].to(device)

            inp = torch.cat([pre, post], dim=1)
            pred = model(inp)
            loss = loss_fn(pred, mask)
            total_val_loss += loss.item()


    # Compute metrics (example: using prediction threshold)
    with torch.no_grad():
        pred_bin = (pred > 0.5).float()

        intersection = (pred_bin * mask).sum(dim=(1,2,3))
        union = pred_bin.sum(dim=(1,2,3)) + mask.sum(dim=(1,2,3)) - intersection
        dice_score = (2 * intersection) / (pred_bin.sum(dim=(1,2,3)) + mask.sum(dim=(1,2,3)) + 1e-6)
        iou_score = intersection / (union + 1e-6)

    mean_iou = iou_score.mean().item()
    mean_dice = dice_score.mean().item()

    results["epochs"].append(epoch+1)
    results["iou"].append(mean_iou)
    results["dice"].append(mean_dice)

    # Save metrics so plot script can read them
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss:.4f}")


checkpoint_dir = "/content/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save(
        model.state_dict(),
        f"{checkpoint_dir}/unet_epoch_{epoch+1}.pth"
    )
print(f"Saved checkpoint: unet_epoch_{epoch+1}.pth")
