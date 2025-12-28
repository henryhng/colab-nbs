"""Ptychography multiscattering demo (v2) with quantem-style APIs."""

from .config import ExperimentConfig, CONFIG_QUICK, CONFIG_MEDIUM
from .data_loader import ExperimentalDataset, load_dataset, load_from_arrays, preprocess_patterns
from .forward_models import Probe, SingleSlicePtychography, MultislicePtychography, create_single_slice_model, create_multislice_model
from .reconstruction import ReconstructionEngine
from .metrics import ReconstructionMetrics, compute_amplitude_rmse, compute_phase_mae

# Optional utilities (require matplotlib/scipy)
try:  # pragma: no cover
    from .visualization import plot_complex_object
except Exception:  # pragma: no cover
    plot_complex_object = None

__all__ = [
    "ExperimentConfig",
    "CONFIG_QUICK",
    "CONFIG_MEDIUM",
    "ExperimentalDataset",
    "load_dataset",
    "load_from_arrays",
    "preprocess_patterns",
    "Probe",
    "SingleSlicePtychography",
    "MultislicePtychography",
    "create_single_slice_model",
    "create_multislice_model",
    "ReconstructionEngine",
    "ReconstructionMetrics",
    "compute_amplitude_rmse",
    "compute_phase_mae",
    "plot_complex_object",
]
