import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int, exclude_bg: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.exclude_bg = exclude_bg
        self.eps = eps

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.exclude_bg:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]

        dims = (0, 2, 3)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()
