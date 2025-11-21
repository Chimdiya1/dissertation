import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()

        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.u3 = DoubleConv(256 + 128, 128)
        self.u2 = DoubleConv(128 + 64, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        c1 = self.d1(x)
        p1 = self.pool(c1)

        c2 = self.d2(p1)
        p2 = self.pool(c2)

        c3 = self.d3(p2)

        up3 = nn.functional.interpolate(c3, scale_factor=2, mode="bilinear")
        cat3 = torch.cat([up3, c2], dim=1)
        c4 = self.u3(cat3)

        up2 = nn.functional.interpolate(c4, scale_factor=2, mode="bilinear")
        cat2 = torch.cat([up2, c1], dim=1)
        c5 = self.u2(cat2)

        return self.out(c5)
