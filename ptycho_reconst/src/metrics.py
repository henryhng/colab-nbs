"""Lightweight metrics for reconstructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["ReconstructionMetrics", "compute_amplitude_rmse", "compute_phase_mae"]


@dataclass
class ReconstructionMetrics:
    amplitude_rmse: float
    phase_mae: float


def _wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phi), torch.cos(phi))


def compute_amplitude_rmse(recon: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> float:
    amp_r = torch.abs(recon)
    amp_t = torch.abs(target)
    mse = torch.mean((amp_r - amp_t) ** 2)
    rmse = torch.sqrt(mse)
    if normalize:
        scale = amp_t.max() - amp_t.min()
        if scale > 1e-12:
            rmse = rmse / scale
    return rmse.item()


def compute_phase_mae(recon: torch.Tensor, target: torch.Tensor) -> float:
    phi_r = _wrap_phase(torch.angle(recon))
    phi_t = _wrap_phase(torch.angle(target))
    diff = _wrap_phase(phi_r - phi_t)
    return torch.mean(torch.abs(diff)).item()
