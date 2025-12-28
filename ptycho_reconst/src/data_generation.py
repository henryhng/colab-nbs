"""Synthetic data helpers for quick smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from .config import ExperimentConfig
from .forward_models import create_multislice_model, create_single_slice_model
from .data_loader import ExperimentalDataset

__all__ = ["SyntheticDataset", "generate_random_object", "generate_synthetic_dataset"]


@dataclass
class SyntheticDataset:
    diffraction_patterns: torch.Tensor
    scan_positions: torch.Tensor
    ground_truth: torch.Tensor
    metadata: Dict


def generate_random_object(size: int, device: torch.device) -> torch.Tensor:
    kx = torch.fft.fftfreq(size, device=device)
    ky = torch.fft.fftfreq(size, device=device)
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    k = torch.sqrt(kx**2 + ky**2)
    filt = torch.exp(-0.5 * (k / 0.2) ** 2)
    amp = torch.real(torch.fft.ifft2(torch.fft.fft2(torch.randn(size, size, device=device)) * filt))
    phase = torch.real(torch.fft.ifft2(torch.fft.fft2(torch.randn(size, size, device=device)) * filt))
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
    phase = 0.5 * phase / (torch.abs(phase).max() + 1e-8)
    return (0.5 + 0.5 * amp) * torch.exp(1j * phase)


def generate_synthetic_dataset(config: ExperimentConfig, num_slices: int = 1) -> SyntheticDataset:
    pos = config.get_scan_positions().to(config.device)
    obj = generate_random_object(config.object_size, config.device)
    if num_slices == 1:
        model = create_single_slice_model(config, pos)
        with torch.no_grad():
            model.object_amp.copy_(torch.abs(obj))
            model.object_phase.copy_(torch.angle(obj))
    else:
        model = create_multislice_model(config, num_slices, pos)
        with torch.no_grad():
            for i in range(num_slices):
                model.slice_amp[i].copy_(torch.abs(obj))
                model.slice_phase[i].copy_(torch.angle(obj) / num_slices)

    patterns = []
    with torch.no_grad():
        for idx in torch.arange(pos.shape[0], device=config.device):
            psi = model(idx.unsqueeze(0))
            patterns.append(torch.abs(psi) ** 2)
    patterns_t = torch.cat(patterns, dim=0).cpu()
    return SyntheticDataset(diffraction_patterns=patterns_t, scan_positions=pos.cpu(), ground_truth=obj.cpu(), metadata={"num_slices": num_slices})


def generate_ml_pairs(n_samples: int, size: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create simple (diffraction, amplitude, phase) triplets for ML training.

    Uses band-limited random amplitude/phase to form a complex object, then
    computes normalized diffraction intensities |FFT|^2.
    """
    torch.manual_seed(seed)
    amps = []
    phases = []
    diffs = []
    for _ in range(n_samples):
        obj = generate_random_object(size, device="cpu")
        amps.append(torch.abs(obj))
        phases.append(torch.angle(obj))
        diff = torch.abs(torch.fft.fftshift(torch.fft.fft2(obj))) ** 2
        diff = diff / (diff.sum() + 1e-8)
        diffs.append(diff)
    return torch.stack(diffs), torch.stack(amps), torch.stack(phases)
