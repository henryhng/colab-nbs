"""forward models for ptychographic simulation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig

__all__ = [
    "Probe",
    "SingleSlicePtychography",
    "MultislicePtychography",
    "create_single_slice_model",
    "create_multislice_model",
]


class Probe(nn.Module):
    """
    electron probe with aberration parameters.

    computes probe in Fourier space using chi(k) then transforms to real space.
    aberrations: defocus, spherical aberration (Cs), astigmatism.
    """

    DEFAULT_DEFOCUS = 50.0  # nm
    DEFAULT_CS = 0.5  # mm
    DEFAULT_ASTIG = 10.0  # nm

    def __init__(
        self,
        size: int,
        pixel_size: float,
        wavelength: float,
        convergence_angle: float,
        device,
    ):
        super().__init__()
        self.size = size
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.convergence_angle = convergence_angle
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        # frequency grids
        kx = torch.fft.fftfreq(size, d=pixel_size, device=self._device)
        ky = torch.fft.fftfreq(size, d=pixel_size, device=self._device)
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        self.register_buffer("k", torch.sqrt(kx**2 + ky**2))
        self.register_buffer("k_angle", torch.atan2(ky, kx))

        # learnable aberration parameters
        self.defocus = nn.Parameter(torch.tensor(50.0, device=self._device))
        self.Cs = nn.Parameter(torch.tensor(0.5, device=self._device))  # mm
        self.astig_mag = nn.Parameter(torch.tensor(10.0, device=self._device))
        self.astig_angle = nn.Parameter(torch.tensor(0.0, device=self._device))
        self.aperture_smooth = nn.Parameter(torch.tensor(0.1, device=self._device))

    def forward(self) -> torch.Tensor:
        """compute probe wave function, returns complex probe normalized to unit intensity."""
        # aberration function chi(k)
        lam = self.wavelength
        chi = (
            np.pi * lam * self.defocus * self.k**2
            + 0.5 * np.pi * lam**3 * (self.Cs * 1e6) * self.k**4
            + np.pi * lam * self.astig_mag * self.k**2
            * torch.cos(2 * (self.k_angle - self.astig_angle))
        )

        # soft aperture
        k_max = self.convergence_angle / lam
        smoothness = torch.abs(self.aperture_smooth) + 0.01
        aperture = torch.sigmoid((k_max - self.k) / (k_max * smoothness))

        # probe in Fourier space
        probe_k = aperture * torch.exp(1j * chi)

        # transform to real space and normalize
        probe_real = torch.fft.ifftshift(torch.fft.ifft2(probe_k))
        probe_real = probe_real / torch.sqrt(torch.sum(torch.abs(probe_real) ** 2))

        return probe_real

    def crop(self, crop_size: int) -> torch.Tensor:
        """get center-cropped probe."""
        p = self.forward()
        c = self.size // 2
        h = crop_size // 2
        return p[c - h : c + h, c - h : c + h]


class SingleSlicePtychography(nn.Module):
    """
    single-slice (projection approximation) ptychography model.
    for thin samples, exit wave = probe * object.
    """

    def __init__(self, config: ExperimentConfig, scan_positions: torch.Tensor):
        super().__init__()
        self.config = config
        self.scan_positions = scan_positions.to(config.device)

        # learnable object (amplitude and phase)
        self.object_amp = nn.Parameter(
            torch.ones(config.object_size, config.object_size, device=config.device)
        )
        self.object_phase = nn.Parameter(
            torch.zeros(config.object_size, config.object_size, device=config.device)
        )

        # probe model
        self.probe = Probe(
            size=config.diffraction_size,
            pixel_size=config.pixel_size,
            wavelength=config.wavelength,
            convergence_angle=config.convergence_angle,
            device=config.device,
        )

    @property
    def obj(self) -> torch.Tensor:
        return self.object_amp * torch.exp(1j * self.object_phase)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """simulate diffraction patterns for given scan positions."""
        obj = self.obj
        probe = self.probe.crop(self.config.probe_size)
        patches = []

        for idx in indices:
            y, x = self.scan_positions[idx]
            # extract object patch at scan position
            patch = obj[y : y + self.config.probe_size, x : x + self.config.probe_size]
            # exit wave = object * probe
            exit_wave = patch * probe
            # pad to diffraction size
            pad = (self.config.diffraction_size - self.config.probe_size) // 2
            exit_wave = F.pad(exit_wave, (pad, pad, pad, pad))
            # propagate to far field
            psi = torch.fft.fft2(exit_wave)
            patches.append(psi)

        return torch.stack(patches, dim=0)


class MultislicePtychography(nn.Module):
    """
    multislice ptychography model for thick samples.
    divides object into slices and propagates wave through each using Fresnel propagation.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        scan_positions: torch.Tensor,
        num_slices: int,
    ):
        super().__init__()
        self.config = config
        self.scan_positions = scan_positions.to(config.device)
        self.num_slices = num_slices

        # learnable slice transmission functions
        self.slice_amp = nn.ParameterList([
            nn.Parameter(torch.ones(config.object_size, config.object_size, device=config.device))
            for _ in range(num_slices)
        ])
        self.slice_phase = nn.ParameterList([
            nn.Parameter(torch.zeros(config.object_size, config.object_size, device=config.device))
            for _ in range(num_slices)
        ])

        # probe model
        self.probe = Probe(
            size=config.diffraction_size,
            pixel_size=config.pixel_size,
            wavelength=config.wavelength,
            convergence_angle=config.convergence_angle,
            device=config.device,
        )

        # precompute fresnel propagator
        self._prop_kernel = self._make_propagator(
            config.diffraction_size,
            config.pixel_size,
            config.wavelength,
            config.slice_thickness,
        )

    def _make_propagator(
        self, size: int, pixel_size: float, wavelength: float, dz: float
    ) -> torch.Tensor:
        """create fresnel propagator kernel H(k) = exp(i * pi * lambda * dz * k^2)."""
        kx = torch.fft.fftfreq(size, d=pixel_size, device=self.config.device)
        ky = torch.fft.fftfreq(size, d=pixel_size, device=self.config.device)
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        k2 = kx**2 + ky**2
        return torch.exp(1j * np.pi * wavelength * dz * k2)

    def _transmission(self, idx: int) -> torch.Tensor:
        return self.slice_amp[idx] * torch.exp(1j * self.slice_phase[idx])

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """simulate diffraction patterns using multislice propagation."""
        probe = self.probe.crop(self.config.probe_size)
        pad = (self.config.diffraction_size - self.config.probe_size) // 2
        patches = []

        for idx in indices:
            y, x = self.scan_positions[idx]
            wave = probe

            # propagate through each slice
            for s in range(self.num_slices):
                t = self._transmission(s)
                patch = t[y : y + self.config.probe_size, x : x + self.config.probe_size]
                exit_wave = wave * patch
                exit_wave = F.pad(exit_wave, (pad, pad, pad, pad))
                # fresnel propagation to next slice
                wave = torch.fft.ifft2(torch.fft.fft2(exit_wave) * self._prop_kernel)

            # final diffraction pattern
            psi = torch.fft.fft2(wave)
            patches.append(psi)

        return torch.stack(patches, dim=0)


def create_single_slice_model(
    config: ExperimentConfig, scan_positions: torch.Tensor
) -> SingleSlicePtychography:
    return SingleSlicePtychography(config, scan_positions)


def create_multislice_model(
    config: ExperimentConfig, num_slices: int, scan_positions: torch.Tensor
) -> MultislicePtychography:
    return MultislicePtychography(config, scan_positions, num_slices=num_slices)
