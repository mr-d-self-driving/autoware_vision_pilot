import torch
import torch.nn as nn


class AutoDriveHead(nn.Module):
    """
    Detect head for AutoDrive — outputs:
        - Curvature       : float  (1/m)
        - Distance (CIPO) : float  (m)
        - Priority        : binary (0/1)

    TODO: Architecture to be designed.
    Input is p5 from AutoDriveBackbone (Conv + C3K2 + SPPF + C2PSA),
    shape [B, width[5], H/32, W/32].
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError("AutoDriveHead is not yet implemented.")

    def forward(self, p5: torch.Tensor):
        raise NotImplementedError("AutoDriveHead is not yet implemented.")
