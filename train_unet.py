import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet
from dataloader import PrePostDataset  # update to include masks
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Dataset
# -------------------------
dataset = PrePostDataset(
    pre_dir="/content/data/pre",
    post_dir="/content/data/post",
    mask_dir="/content/data/masks",
    augment=True
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# -------------------------
# Model, Loss, Optimizer
# -------------------------
model = UNet(in_channels=6, out_channels=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# Training Loop (5 epochs)
# -------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in loader:
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

    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss:.4f}")


checkpoint_dir = "/content/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save(
        model.state_dict(),
        f"{checkpoint_dir}/unet_epoch_{epoch+1}.pth"
    )
print(f"Saved checkpoint: unet_epoch_{epoch+1}.pth")
