"""Tiny thickness benchmark (Jetson-friendly).

Runs single-slice vs multislice recon on small synthetic datasets and reports
amplitude RMSE and runtime. Keeps sizes modest so it can run on an 8 GB GPU or CPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from src.config import CONFIG_QUICK
from src.data_generation import generate_synthetic_dataset
from src.metrics import compute_amplitude_rmse
from src.reconstruction import ReconstructionEngine
from src.forward_models import create_single_slice_model, create_multislice_model


def run_once(num_slices: int) -> dict:
    cfg = CONFIG_QUICK.clone(num_iterations=80, batch_size=32, learning_rate=0.2)
    data = generate_synthetic_dataset(cfg, num_slices=num_slices)
    patterns = data.diffraction_patterns.to(cfg.device)
    positions = data.scan_positions.to(cfg.device)

    # Single-slice
    model_s = create_single_slice_model(cfg, positions)
    eng_s = ReconstructionEngine(model_s, lr_object=cfg.learning_rate, lr_probe=cfg.learning_rate * 0.2)
    t0 = time.time()
    res_s = eng_s.reconstruct(patterns, num_iterations=cfg.num_iterations, batch_size=cfg.batch_size, verbose=False)
    t_s = time.time() - t0

    # Multislice
    model_m = create_multislice_model(cfg, num_slices, positions)
    eng_m = ReconstructionEngine(model_m, lr_object=cfg.learning_rate, lr_probe=cfg.learning_rate * 0.2)
    t0 = time.time()
    res_m = eng_m.reconstruct(patterns, num_iterations=cfg.num_iterations, batch_size=cfg.batch_size, verbose=False)
    t_m = time.time() - t0

    # Metrics vs ground truth
    gt = data.ground_truth
    # For multislice, multiply slices to compare to projected ground truth
    proj_m = torch.ones_like(gt)
    for sl in res_m.reconstruction:
        proj_m = proj_m * sl

    return {
        "slices": num_slices,
        "single_loss": res_s.final_loss,
        "multi_loss": res_m.final_loss,
        "single_rmse": compute_amplitude_rmse(res_s.reconstruction, gt),
        "multi_rmse": compute_amplitude_rmse(proj_m, gt),
        "time_single_s": t_s,
        "time_multi_s": t_m,
    }


def main():
    torch.manual_seed(0)
    results = []
    for slices in [1, 2, 4]:
        out = run_once(slices)
        results.append(out)
        print(f"{slices} slice(s): single RMSE {out['single_rmse']:.4f}, multi RMSE {out['multi_rmse']:.4f}, "
              f"time {out['time_single_s']:.1f}s / {out['time_multi_s']:.1f}s")

    # Write a simple JSONL report
    import json

    out_path = Path("benchmark_thickness.jsonl")
    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
