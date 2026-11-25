"""CBAM attention module (Woo et al., ECCV 2018).

Used as a drop-in enhancement for YOLOv8 C2f blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention with shared MLP on avg/max pooled features."""

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        max_ = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + max_)


class SpatialAttention(nn.Module):
    """Spatial attention from channel-wise avg/max maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat((avg_map, max_map), dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        return x * self.spatial_att(x)
