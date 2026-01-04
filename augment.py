import csv
from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


import random
import numpy as np

def augment_tile(x6: np.ndarray, mask: np.ndarray, rng: random.Random):
    """
    x6: (H,W,6) float32 in [0,1] or uint8 0..255
    mask: (H,W) uint8 with values 0..4

    Returns augmented (x6, mask) with same shapes.
    """
    assert x6.ndim == 3 and x6.shape[2] == 6
    assert mask.ndim == 2
    assert x6.shape[0] == mask.shape[0] and x6.shape[1] == mask.shape[1]

    # ----- Spatial transforms (apply to both) -----
    # Horizontal flip
    if rng.random() < 0.5:
        x6 = np.flip(x6, axis=1)
        mask = np.flip(mask, axis=1)

    # Vertical flip
    if rng.random() < 0.5:
        x6 = np.flip(x6, axis=0)
        mask = np.flip(mask, axis=0)

    # Random rotation by 0/90/180/270 degrees
    k = rng.randint(0, 3)
    if k:
        x6 = np.rot90(x6, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

    # ----- Photometric transforms (input only) -----
    # Convert to float for jitter math
    if x6.dtype != np.float32 and x6.dtype != np.float64:
        x6f = x6.astype(np.float32) / 255.0
    else:
        x6f = x6.astype(np.float32)

    # Brightness jitter (small)
    if rng.random() < 0.8:
        b = rng.uniform(-0.08, 0.08)
        x6f = x6f + b

    # Contrast jitter (small)
    if rng.random() < 0.8:
        c = rng.uniform(0.9, 1.1)
        mean = x6f.mean(axis=(0, 1), keepdims=True)
        x6f = (x6f - mean) * c + mean

    # Small Gaussian noise
    if rng.random() < 0.3:
        noise = rng.normalvariate(0.0, 0.02)
        x6f = x6f + noise

    # Clamp to [0,1]
    x6f = np.clip(x6f, 0.0, 1.0).astype(np.float32)

    return x6f, mask.astype(np.uint8)


class XBDTilesDataset(Dataset):
    def __init__(self, csv_path: str, augment: bool = False, seed: int = 42):
        self.csv_path = Path(csv_path)
        self.augment = augment
        self.rng = random.Random(seed)

        self.rows = []
        with open(self.csv_path, "r", newline="") as f:
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
            # Important: use deterministic-per-sample randomness to avoid weird correlations
            # Create a local RNG based on (global_seed + idx)
            local_rng = random.Random(self.rng.randint(0, 10**9) + idx)
            x6, mask = augment_tile(x6, mask, local_rng)
        else:
            # Ensure float32 input in [0,1]
            if x6.dtype != np.float32 and x6.dtype != np.float64:
                x6 = (x6.astype(np.float32) / 255.0)
            else:
                x6 = x6.astype(np.float32)

        # Convert to torch tensors
        # PyTorch expects (C,H,W)
        x6_t = torch.from_numpy(x6).permute(2, 0, 1)  # (6,256,256)
        y_t = torch.from_numpy(mask.astype(np.int64))  # (256,256) for CrossEntropyLoss

        return x6_t, y_t
