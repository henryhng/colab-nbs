"""Data loaders and preprocessing for ptychography experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

DEFAULT_FLOAT_DTYPE = torch.float32

try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

try:
    import py4DSTEM
    HAS_PY4DSTEM = True
except Exception:
    HAS_PY4DSTEM = False


@dataclass
class ExperimentalDataset:
    diffraction_patterns: torch.Tensor  # [N, H, W]
    scan_positions: np.ndarray          # [N, 2] (y, x)
    pixel_size: Optional[float] = None
    scan_step: Optional[float] = None
    voltage_kev: Optional[float] = None
    metadata: Optional[Dict] = None


def _to_tensor(arr: np.ndarray, device: Union[str, torch.device]) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=DEFAULT_FLOAT_DTYPE, device=device)


def load_numpy(path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    return _to_tensor(np.load(path), device)


def load_hdf5(path: Union[str, Path], dataset_key: str = "data", device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    if not HAS_H5PY:
        raise ImportError("h5py is required to load HDF5 files.")
    with h5py.File(path, "r") as f:
        data = f[dataset_key][:]
    return _to_tensor(data, device)


def load_4dstem(path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> ExperimentalDataset:
    if not HAS_PY4DSTEM:
        raise ImportError("py4DSTEM is required to load .emd datacubes.")
    cube = py4DSTEM.read(path)
    Rx, Ry, Qx, Qy = cube.data.shape
    patterns = cube.data.reshape(Rx * Ry, Qx, Qy)
    positions = np.stack(np.meshgrid(np.arange(Ry), np.arange(Rx), indexing="ij"), axis=-1).reshape(-1, 2)
    metadata = {}
    if hasattr(cube, "calibration"):
        cal = cube.calibration
        if hasattr(cal, "get_Q_pixel_size"):
            metadata["q_pixel_size"] = cal.get_Q_pixel_size()
        if hasattr(cal, "get_R_pixel_size"):
            metadata["r_pixel_size"] = cal.get_R_pixel_size()
    return ExperimentalDataset(_to_tensor(patterns, device), positions, metadata=metadata)


def load_from_arrays(patterns: np.ndarray, scan_positions: np.ndarray, pixel_size: Optional[float] = None,
                     device: Union[str, torch.device] = "cpu") -> ExperimentalDataset:
    if patterns.ndim == 4:
        Ry, Rx, Qy, Qx = patterns.shape
        patterns = patterns.reshape(Ry * Rx, Qy, Qx)
    return ExperimentalDataset(_to_tensor(patterns, device), scan_positions, pixel_size=pixel_size)


def load_dataset(path: Union[str, Path], format: str = "auto", device: Union[str, torch.device] = "cpu",
                 **kwargs) -> ExperimentalDataset:
    path = Path(path)
    if format == "auto":
        ext = path.suffix.lower()
        if ext == ".npy":
            format = "numpy"
        elif ext in {".h5", ".hdf5"}:
            format = "hdf5"
        elif ext == ".emd":
            format = "py4dstem"
        else:
            raise ValueError(f"Unknown data format for {path}")

    if format == "numpy":
        patterns = load_numpy(path, device)
    elif format == "hdf5":
        patterns = load_hdf5(path, kwargs.get("dataset_key", "data"), device)
    elif format == "py4dstem":
        return load_4dstem(path, device)
    else:
        raise ValueError(f"Unsupported format: {format}")

    n = int(np.sqrt(len(patterns)))
    positions = np.array([[i, j] for i in range(n) for j in range(n)], dtype=np.int32)
    return ExperimentalDataset(patterns, positions)


def preprocess_patterns(patterns: torch.Tensor, normalize: bool = True, clamp_std: float = 10.0) -> torch.Tensor:
    """Clamp outliers and optionally normalize per-pattern."""
    x = patterns.clone()
    if clamp_std is not None:
        mean, std = x.mean(), x.std()
        x = torch.clamp(x, max=mean + clamp_std * std)
    if normalize:
        sums = x.sum(dim=(1, 2), keepdim=True)
        x = x / (sums + 1e-10)
    return x


__all__ = [
    "ExperimentalDataset",
    "load_numpy",
    "load_hdf5",
    "load_4dstem",
    "load_from_arrays",
    "load_dataset",
    "preprocess_patterns",
]
