import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from unet import UNet
from dataloader import PrePostDataset
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# Metrics: IoU + Dice
# -------------------------------------------------------
def compute_iou(pred, mask, eps=1e-6):
    pred = pred > 0.5
    mask = mask > 0.5

    intersection = (pred & mask).sum().item()
    union = (pred | mask).sum().item()

    return intersection / (union + eps)


def compute_dice(pred, mask, eps=1e-6):
    pred = pred > 0.5
    mask = mask > 0.5

    intersection = (pred & mask).sum().item()
    dice = (2 * intersection) / (pred.sum().item() + mask.sum().item() + eps)

    return dice


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
def evaluate(model, loader, save_preds=False, pred_dir="predictions"):
    model.eval()
    os.makedirs(pred_dir, exist_ok=True)

    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            mask = batch["mask"].to(device)

            inp = torch.cat([pre, post], dim=1)

            pred = model(inp)
            pred = torch.sigmoid(pred)

            # Compute metrics
            for b in range(pred.shape[0]):
                pred_b = pred[b, 0].cpu().numpy()
                mask_b = mask[b, 0].cpu().numpy()

                iou_scores.append(compute_iou(pred_b, mask_b))
                dice_scores.append(compute_dice(pred_b, mask_b))

                # Save predictions
                if save_preds:
                    pred_img = (pred_b * 255).astype(np.uint8)
                    cv2.imwrite(f"{pred_dir}/pred_{i}_{b}.png", pred_img)

    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)

    print("--------------- Evaluation Results ---------------")
    print(f"Mean IoU:  {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print("--------------------------------------------------")

    return mean_iou, mean_dice


# -------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------
def main():

    # -------------------------
    # Validation Dataset
    # -------------------------
    val_dataset = PrePostDataset(
        pre_dir="/content/data/val/pre",
        post_dir="/content/data/val/post",
        mask_dir="/content/data/val/masks",
        augment=False   # No augmentation during evaluation
    )

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # -------------------------
    # Load Model
    # -------------------------
    model = UNet(in_channels=6, out_channels=1).to(device)

    checkpoint_path = "/content/checkpoints/unet_epoch_5.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print(f"Loaded model from: {checkpoint_path}")

    # -------------------------
    # Run evaluation
    # -------------------------
    evaluate(
        model,
        val_loader,
        save_preds=True,           # Set to False if you don't need PNG outputs
        pred_dir="/content/predictions"
    )


if __name__ == "__main__":
    main()
