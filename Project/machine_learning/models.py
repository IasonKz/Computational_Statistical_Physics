"""
PyTorch models for the Heisenberg+DMI supervised ML pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RawSpinMLP(nn.Module):
    """Fully connected classifier on flattened raw spin configurations."""

    def __init__(self, input_dim: int, hidden1: int = 256,
                 hidden2: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SpinCNN3D(nn.Module):
    """Small 3D CNN for spin configurations with channels Sx,Sy,Sz."""

    def __init__(self, channels: int = 3, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)
