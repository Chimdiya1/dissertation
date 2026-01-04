import numpy as np
from pathlib import Path
from PIL import Image
import torch

from unet_model import UNet
from stitch_utils import predict_full_scene_from_6ch

BEST_CKPT = "/content/train/checkpoints_unet/best_unet.pth"
INPUT_6CH_FOLDER = Path("/content/train/inputs_6ch")
OUT_PRED_FOLDER = Path("/content/train/preds_fullscene")
OUT_PRED_FOLDER.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 256
STRIDE = 128
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(ckpt_path):
    model = UNet(in_channels=6, num_classes=5, base=32)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return model

def main():
    model = load_model(BEST_CKPT)

    npy_files = sorted(INPUT_6CH_FOLDER.glob("*_6ch.npy"))
    if not npy_files:
        print("No *_6ch.npy found in", INPUT_6CH_FOLDER)
        return

    for p in npy_files:  # run all
        x6 = np.load(p)
        pred = predict_full_scene_from_6ch(
            model, x6,
            tile_size=TILE_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            num_classes=5
        )
        out_name = p.name.replace("_6ch.npy", "_pred_mask.png")
        out_path = OUT_PRED_FOLDER / out_name
        Image.fromarray(pred).save(out_path)

        print("[OK]", p.name, "->", out_path, "| unique:", np.unique(pred))

if __name__ == "__main__":
    main()
