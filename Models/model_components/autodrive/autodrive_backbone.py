import torch
import torch.nn as nn
from Models.model_components.common_layers import (
    Conv, SPPF, C2PSA, C3K2
)


class AutoDriveBackbone(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()

        # p1/2  —  1024×512 → 512×256
        self.p1 = Conv(width[0], width[1], activation=nn.SiLU(), k=3, s=2, p=1)

        # p2/4  —  512×256 → 256×128
        self.p2 = nn.Sequential(
            Conv(width[1], width[2], activation=nn.SiLU(), k=3, s=2, p=1),
            C3K2(width[2], width[3], depth[0], csp[0], r=4)
        )

        # p3/8  —  256×128 → 128×64
        self.p3 = nn.Sequential(
            Conv(width[3], width[3], activation=nn.SiLU(), k=3, s=2, p=1),
            C3K2(width[3], width[4], depth[1], csp[0], r=4)
        )

        # p4/16  —  128×64 → 64×32
        self.p4 = nn.Sequential(
            Conv(width[4], width[4], activation=nn.SiLU(), k=3, s=2, p=1),
            C3K2(width[4], width[4], depth[1], csp[1], r=2)
        )

        # p5/32 — same layout as AutoSpeedBackbone p5: stem + block + SPPF + C2PSA
        self.p5 = nn.Sequential(
            Conv(width[4], width[5], activation=nn.SiLU(), k=3, s=2, p=1),
            C3K2(width[5], width[5], depth[1], csp[1], r=2),
            SPPF(width[5], width[5]),
            C2PSA(width[5], width[5])
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        # [B, width[5], H/32, W/32]  e.g. 512×1024 → 16×32
        return p5
