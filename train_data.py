import csv
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

def augment_tile(x6: np.ndarray, mask: np.ndarray, rng: random.Random):
    # spatial (same for x and mask)
    if rng.random() < 0.5:
        x6 = np.flip(x6, axis=1); mask = np.flip(mask, axis=1)
    if rng.random() < 0.5:
        x6 = np.flip(x6, axis=0); mask = np.flip(mask, axis=0)

    k = rng.randint(0, 3)
    if k:
        x6 = np.rot90(x6, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

    # photometric (x only)
    if x6.dtype != np.float32 and x6.dtype != np.float64:
        x6 = x6.astype(np.float32) / 255.0
    else:
        x6 = x6.astype(np.float32)

    if rng.random() < 0.8:
        x6 = x6 + rng.uniform(-0.08, 0.08)

    if rng.random() < 0.8:
        c = rng.uniform(0.9, 1.1)
        mean = x6.mean(axis=(0, 1), keepdims=True)
        x6 = (x6 - mean) * c + mean

    x6 = np.clip(x6, 0.0, 1.0).astype(np.float32)
    return x6, mask.astype(np.uint8)


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
        mask = np.array(Image.open(r["mask_tile_path"]), dtype=np.uint8)

        if self.augment:
            # vary randomness per sample call
            rng = random.Random(self.seed + random.randint(0, 10**9))
            x6, mask = augment_tile(x6, mask, rng)
        else:
            if x6.dtype != np.float32 and x6.dtype != np.float64:
                x6 = x6.astype(np.float32) / 255.0
            else:
                x6 = x6.astype(np.float32)

        x_t = torch.from_numpy(x6).permute(2, 0, 1)  # (6,256,256)
        y_t = torch.from_numpy(mask.astype(np.int64)) # (256,256)
        return x_t, y_t
