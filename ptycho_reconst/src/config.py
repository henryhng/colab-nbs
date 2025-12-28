"""configuration and physical constants for ptychographic reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

__all__ = [
    "ExperimentConfig",
    "CONFIG_QUICK",
    "CONFIG_MEDIUM",
    "compute_wavelength",
]

# physical constants (SI units)
ELECTRON_MASS = 9.109e-31  # kg
ELECTRON_CHARGE = 1.602e-19  # C
PLANCK_CONSTANT = 6.626e-34  # J·s
ELECTRON_REST_ENERGY = 511.0  # keV


def compute_wavelength(voltage_kev: float) -> float:
    """compute relativistic electron wavelength in nm."""
    V = voltage_kev * 1e3  # eV
    E0 = ELECTRON_REST_ENERGY * 1e3  # eV
    factor = 1 + V / (2 * E0)
    wavelength_m = PLANCK_CONSTANT / np.sqrt(
        2 * ELECTRON_MASS * ELECTRON_CHARGE * V * factor
    )
    return wavelength_m * 1e9


def _select_device(device):
    """select computation device (cuda > mps > cpu)."""
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ExperimentConfig:
    """configuration for ptychography experiments, holds all the physics parameters."""

    # default values
    DEFAULT_VOLTAGE = 200.0
    DEFAULT_PIXEL_SIZE = 0.05
    DEFAULT_CONVERGENCE = 20e-3

    # grid sizes
    object_size: int = 128
    probe_size: int = 48
    diffraction_size: int = 128
    scan_grid: int = 32

    # physics
    pixel_size: float = DEFAULT_PIXEL_SIZE  # nm/pixel
    voltage_kev: float = DEFAULT_VOLTAGE
    convergence_angle: float = DEFAULT_CONVERGENCE  # rad
    slice_thickness: float = 0.5  # nm

    # optimization defaults
    num_iterations: int = 300
    learning_rate: float = 0.5
    batch_size: int = 64

    # internal
    _device: torch.device = field(default=None, repr=False)
    _dtype_complex: torch.dtype = field(default=torch.complex64, repr=False)
    _dtype_real: torch.dtype = field(default=torch.float32, repr=False)

    def __post_init__(self):
        self._device = _select_device(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, val) -> None:
        self._device = _select_device(val)

    @property
    def dtype_complex(self) -> torch.dtype:
        return self._dtype_complex

    @property
    def dtype_real(self) -> torch.dtype:
        return self._dtype_real

    @property
    def num_positions(self) -> int:
        return self.scan_grid * self.scan_grid

    @property
    def wavelength(self) -> float:
        return compute_wavelength(self.voltage_kev)

    def get_scan_positions(self) -> torch.Tensor:
        """generate scan positions on a regular grid, returns (N, 2) array of (y, x) coords."""
        margin = self.probe_size // 2
        max_pos = self.object_size - self.probe_size - margin
        ys = torch.linspace(margin, max_pos, self.scan_grid, dtype=torch.int32)
        xs = torch.linspace(margin, max_pos, self.scan_grid, dtype=torch.int32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy.flatten(), xx.flatten()], dim=1)

    def clone(self, **overrides) -> "ExperimentConfig":
        """create a copy with optional overrides."""
        data = {
            "object_size": self.object_size,
            "probe_size": self.probe_size,
            "diffraction_size": self.diffraction_size,
            "scan_grid": self.scan_grid,
            "pixel_size": self.pixel_size,
            "voltage_kev": self.voltage_kev,
            "convergence_angle": self.convergence_angle,
            "slice_thickness": self.slice_thickness,
            "num_iterations": self.num_iterations,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
        data.update(overrides)
        return ExperimentConfig(**data)


# presets
CONFIG_QUICK = ExperimentConfig(
    object_size=96,
    probe_size=32,
    diffraction_size=96,
    scan_grid=24,
    num_iterations=120,
    batch_size=32,
    learning_rate=0.2,
)

CONFIG_MEDIUM = ExperimentConfig()
