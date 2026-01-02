import csv
from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

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
