import numpy as np
import torch

@torch.no_grad()
def mean_iou(pred, target, num_classes=5, exclude_bg=True):
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
