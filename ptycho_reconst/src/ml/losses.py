"""Losses for amplitude/phase regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AmplitudePhaseLoss", "GradientLoss"]


class AmplitudePhaseLoss(nn.Module):
    def __init__(self, amp_weight: float = 1.0, phase_weight: float = 1.0):
        super().__init__()
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight

    def forward(self, amp_pred: torch.Tensor, phase_pred: torch.Tensor, amp_true: torch.Tensor, phase_true: torch.Tensor) -> torch.Tensor:
        amp_loss = F.l1_loss(amp_pred, amp_true)
        wrapped = torch.atan2(torch.sin(phase_pred - phase_true), torch.cos(phase_pred - phase_true))
        phase_loss = torch.mean(torch.abs(wrapped))
        return self.amp_weight * amp_loss + self.phase_weight * phase_loss


class GradientLoss(nn.Module):
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b * c, 1, h, w)
        gx = F.conv2d(x_flat, self.kx, padding=1)
        gy = F.conv2d(x_flat, self.ky, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return self.weight * torch.mean(grad_mag)
