"""
model.py

Recommended U-Net variant for DICOM + PNG diaphragm segmentation.

Model type:
    2D Residual Attention U-Net with GroupNorm

Designed for your current main.py:
    from model import UNet
    model = UNet(n_channels=1, n_classes=1)

Why this model is suitable for your dataset:
    - DICOM CT slices are grayscale, so n_channels=1.
    - PNG masks are binary segmentation labels, so n_classes=1.
    - Residual blocks help gradients flow during training.
    - Attention gates help the decoder focus on the target diaphragm/segmentation region.
    - GroupNorm is more stable than BatchNorm when batch size is small.
    - Dropout in deeper layers helps reduce overfitting.
    - Output is raw logits, so keep using BCEWithLogitsLoss or Dice+BCE loss.

Important:
    Do NOT apply sigmoid inside this model.
    Your training/validation code should apply sigmoid only when converting logits to probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def choose_group_count(channels: int, preferred: int = 8) -> int:
    """
    GroupNorm requires channels % groups == 0.
    This helper chooses a valid group count automatically.
    """
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    """Conv2d + GroupNorm + SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(choose_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual convolution block.

    Structure:
        ConvNormAct
        Dropout2d optional
        Conv + GroupNorm
        Add shortcut
        SiLU
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(choose_group_count(out_channels), out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + identity
        out = self.activation(out)
        return out


class DownBlock(nn.Module):
    """Downsample with MaxPool, then process with a residual block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ResidualBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class AttentionGate(nn.Module):
    """
    Attention gate from Attention U-Net.

    It receives:
        g: decoder gating feature
        x: encoder skip feature

    It returns:
        x multiplied by learned spatial attention coefficients.
    """

    def __init__(self, gate_channels: int, skip_channels: int, intermediate_channels: int):
        super().__init__()

        self.gate_transform = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=True),
            nn.GroupNorm(choose_group_count(intermediate_channels), intermediate_channels),
        )

        self.skip_transform = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, bias=True),
            nn.GroupNorm(choose_group_count(intermediate_channels), intermediate_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Match spatial sizes if needed.
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        attention = self.activation(self.gate_transform(g) + self.skip_transform(x))
        attention = self.psi(attention)
        return x * attention


class UpBlock(nn.Module):
    """
    Upsample decoder feature, attention-filter the skip feature, concatenate, then refine.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_transpose: bool = True,
    ):
        super().__init__()

        if use_transpose:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )

        self.attention = AttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            intermediate_channels=max(out_channels // 2, 1),
        )

        self.conv = ResidualBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    @staticmethod
    def pad_or_crop_to_match(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Make x spatially match target. This protects the model from odd image sizes.
        """
        target_h, target_w = target.shape[2], target.shape[3]
        h, w = x.shape[2], x.shape[3]

        # Crop if x is too large.
        if h > target_h:
            start = (h - target_h) // 2
            x = x[:, :, start:start + target_h, :]
        if w > target_w:
            start = (w - target_w) // 2
            x = x[:, :, :, start:start + target_w]

        # Pad if x is too small.
        diff_h = target_h - x.shape[2]
        diff_w = target_w - x.shape[3]
        if diff_h > 0 or diff_w > 0:
            x = F.pad(
                x,
                [
                    diff_w // 2,
                    diff_w - diff_w // 2,
                    diff_h // 2,
                    diff_h - diff_h // 2,
                ],
            )
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.pad_or_crop_to_match(x, skip)

        skip = self.attention(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    2D Residual Attention U-Net.

    Recommended settings for your current script:
        UNet(n_channels=1, n_classes=1)

    Input:
        Tensor shape [B, 1, H, W]

    Output:
        Raw logits shape [B, 1, H, W]

    Channel layout when base_channels=32:
        Encoder:    32 -> 64 -> 128 -> 256 -> 512
        Decoder:    256 -> 128 -> 64 -> 32
        Output:     1 binary mask logit channel
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        base_channels: int = 32,
        dropout: float = 0.10,
        use_transpose: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.enc1 = ResidualBlock(n_channels, c1, dropout=0.0)
        self.enc2 = DownBlock(c1, c2, dropout=0.0)
        self.enc3 = DownBlock(c2, c3, dropout=dropout * 0.5)
        self.enc4 = DownBlock(c3, c4, dropout=dropout)

        # Bottleneck
        self.bottleneck = DownBlock(c4, c5, dropout=dropout)

        # Decoder with attention-gated skip connections
        self.dec4 = UpBlock(c5, c4, c4, dropout=dropout, use_transpose=use_transpose)
        self.dec3 = UpBlock(c4, c3, c3, dropout=dropout * 0.5, use_transpose=use_transpose)
        self.dec2 = UpBlock(c3, c2, c2, dropout=0.0, use_transpose=use_transpose)
        self.dec1 = UpBlock(c2, c1, c1, dropout=0.0, use_transpose=use_transpose)

        self.out = nn.Conv2d(c1, n_classes, kernel_size=1)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize convolution and normalization layers for stable training."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)          # [B, c1, H, W]
        x2 = self.enc2(x1)         # [B, c2, H/2, W/2]
        x3 = self.enc3(x2)         # [B, c3, H/4, W/4]
        x4 = self.enc4(x3)         # [B, c4, H/8, W/8]

        # Bottleneck
        x5 = self.bottleneck(x4)   # [B, c5, H/16, W/16]

        # Decoder
        x = self.dec4(x5, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        return self.out(x)
