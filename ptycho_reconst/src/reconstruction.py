"""iterative reconstruction engine for ptychography."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union
import time

import numpy as np
import torch
import torch.optim as optim

from .forward_models import SingleSlicePtychography, MultislicePtychography

__all__ = ["ReconstructionEngine", "ReconstructionResult"]


@dataclass
class ReconstructionResult:
    """results from ptychographic reconstruction."""

    reconstruction: Union[torch.Tensor, List[torch.Tensor]]
    loss_history: List[float]
    time_per_iteration: List[float]
    total_time: float
    final_loss: float
    converged: bool
    diagnostics: dict = field(default_factory=dict)


class ReconstructionEngine:
    """gradient-based reconstruction using Adam optimizer."""

    def __init__(
        self,
        model: Union[SingleSlicePtychography, MultislicePtychography],
        lr_object: float = 0.5,
        lr_probe: float = 0.1,
        loss_type: str = "amp_mse",
    ):
        self.model = model
        self.device = model.config.device
        self.loss_type = loss_type

        # collect object parameters
        obj_params = []
        if isinstance(model, MultislicePtychography):
            obj_params += list(model.slice_amp) + list(model.slice_phase)
        else:
            obj_params += [model.object_amp, model.object_phase]

        # collect probe parameters
        probe_params = list(model.probe.parameters())

        # setup optimizer with parameter groups
        self.optimizer = optim.Adam([
            {"params": obj_params, "lr": lr_object},
            {"params": probe_params, "lr": lr_probe},
        ])

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "intensity_mse":
            # compare intensities directly
            return torch.mean((torch.abs(pred) ** 2 - target) ** 2)
        else:
            # compare amplitudes (default)
            return torch.mean((torch.abs(pred) - torch.sqrt(target + 1e-10)) ** 2)

    def reconstruct(
        self,
        measured: torch.Tensor,
        num_iterations: int = 200,
        batch_size: int = 64,
        min_iterations: int = 20,
        convergence_tol: float = 1e-4,
        verbose: bool = True,
    ) -> ReconstructionResult:
        """run iterative reconstruction."""
        measured = measured.to(self.device)
        n = measured.shape[0]
        loss_history: List[float] = []
        time_history: List[float] = []
        converged = False

        for it in range(num_iterations):
            t0 = time.time()

            # shuffle pattern order each epoch
            order = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            num_batches = 0

            # process in batches
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]

                self.optimizer.zero_grad()
                psi = self.model(idx)
                loss = self._compute_loss(psi, measured[idx])
                loss.backward()

                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append(avg_loss)
            time_history.append(time.time() - t0)

            if verbose:
                print(f"iter {it:03d} loss {avg_loss:.5e}")

            # check convergence
            if it + 1 >= min_iterations and len(loss_history) >= 2:
                delta = abs(loss_history[-1] - loss_history[-2])
                if delta < convergence_tol:
                    converged = True
                    if verbose:
                        print(f"Converged at iter {it} (delta={delta:.2e})")
                    break

        # extract final reconstruction
        with torch.no_grad():
            if isinstance(self.model, MultislicePtychography):
                recon = [
                    (amp * torch.exp(1j * phase)).detach().cpu()
                    for amp, phase in zip(self.model.slice_amp, self.model.slice_phase)
                ]
            else:
                recon = self.model.obj.detach().cpu()

        return ReconstructionResult(
            reconstruction=recon,
            loss_history=loss_history,
            time_per_iteration=time_history,
            total_time=float(np.sum(time_history)),
            final_loss=loss_history[-1] if loss_history else float("inf"),
            converged=converged,
            diagnostics={"loss_type": self.loss_type, "num_positions": n},
        )
