"""neural network architectures for ptychographic reconstruction."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConvBlock", "PtychoNN", "FCUNet", "count_parameters"]


class ConvBlock(nn.Module):
    """conv -> batchnorm -> relu -> conv -> batchnorm -> relu."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """u-net encoder (downsampling path)."""

    def __init__(self, channels: List[int]):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips


class Decoder(nn.Module):
    """u-net decoder (upsampling path)."""

    def __init__(self, skip_channels: List[int], bottleneck_channels: int):
        super().__init__()
        blocks = []
        in_ch = bottleneck_channels
        for ch in reversed(skip_channels):
            blocks.append(ConvBlock(in_ch + ch, ch))
            in_ch = ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for block, skip in zip(self.blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class PtychoNN(nn.Module):
    """two-headed u-net for amplitude and phase prediction."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, depth: int = 3):
        super().__init__()

        # channel progression: [1, 16, 32, 64] for depth=3
        enc_channels = [in_channels] + [base_channels * 2**i for i in range(depth)]

        self.encoder = Encoder(enc_channels)
        self.bottleneck = ConvBlock(enc_channels[-1], enc_channels[-1])

        # separate decoders for amplitude and phase
        self.decoder_amp = Decoder(enc_channels[1:], enc_channels[-1])
        self.decoder_phase = Decoder(enc_channels[1:], enc_channels[-1])

        # output layers
        self.out_amp = nn.Conv2d(enc_channels[1], 1, 1)
        self.out_phase = nn.Conv2d(enc_channels[1], 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, skips = self.encoder(x)
        x = self.bottleneck(x)

        # amplitude: sigmoid -> [0, 1]
        amp = torch.sigmoid(self.out_amp(self.decoder_amp(x, skips)))

        # phase: tanh * pi -> [-pi, pi]
        phase = torch.tanh(self.out_phase(self.decoder_phase(x, skips))) * torch.pi

        return amp, phase


class FCUNet(nn.Module):
    """u-net for multi-slice prediction, outputs multiple slices for thick samples."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        depth: int = 3,
        num_slices: int = 1,
    ):
        super().__init__()

        enc_channels = [in_channels] + [base_channels * 2**i for i in range(depth)]

        self.encoder = Encoder(enc_channels)
        self.bottleneck = ConvBlock(enc_channels[-1], enc_channels[-1])
        self.decoder = Decoder(enc_channels[1:], enc_channels[-1])

        # multi-slice output
        self.out_amp = nn.Conv2d(enc_channels[1], num_slices, 1)
        self.out_phase = nn.Conv2d(enc_channels[1], num_slices, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)

        amp = torch.sigmoid(self.out_amp(x))
        phase = torch.tanh(self.out_phase(x)) * torch.pi

        return amp, phase


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
