import numpy as np
import torch
import torch.nn.functional as F

def grid_positions(L, tile, stride):
    if L <= tile:
        return [0]
    pos = list(range(0, L - tile + 1, stride))
    if pos[-1] != L - tile:
        pos.append(L - tile)
    return pos


@torch.no_grad()
def predict_full_scene_from_6ch(
    model,
    x6: np.ndarray,
    tile_size=256,
    stride=128,
    batch_size=8,
    device=None,
    num_classes=5,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ensure float32 in [0,1]
    if x6.dtype != np.float32 and x6.dtype != np.float64:
        x6 = x6.astype(np.float32) / 255.0
    else:
        x6 = x6.astype(np.float32)

    H, W, C = x6.shape
    assert C == 6, f"Expected 6 channels, got {C}"

    ys = grid_positions(H, tile_size, stride)
    xs = grid_positions(W, tile_size, stride)

    prob_acc = np.zeros((num_classes, H, W), dtype=np.float32)
    count_acc = np.zeros((H, W), dtype=np.float32)

    model.eval()
    model.to(device)

    tiles, coords = [], []

    def flush():
        nonlocal tiles, coords
        if not tiles:
            return

        batch = np.stack(tiles, axis=0)  # (N,tile,tile,6)
        batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)  # (N,6,tile,tile)

        logits = model(batch_t)                 # (N,5,tile,tile)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        for i, (x0, y0) in enumerate(coords):
            prob_acc[:, y0:y0+tile_size, x0:x0+tile_size] += probs[i]
            count_acc[y0:y0+tile_size, x0:x0+tile_size] += 1.0

        tiles, coords = [], []

    for y0 in ys:
        for x0 in xs:
            tiles.append(x6[y0:y0+tile_size, x0:x0+tile_size, :])
            coords.append((x0, y0))
            if len(tiles) >= batch_size:
                flush()
    flush()

    prob_acc /= np.clip(count_acc[None, :, :], 1e-6, None)
    pred = np.argmax(prob_acc, axis=0).astype(np.uint8)
    return pred
