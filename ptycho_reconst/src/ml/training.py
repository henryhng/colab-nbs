"""training utilities for ML-based ptychography."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .losses import AmplitudePhaseLoss

__all__ = ["TrainingConfig", "TrainingResult", "create_datasets", "Trainer"]


@dataclass
class TrainingConfig:
    """configuration for model training."""

    DEFAULT_LR = 1e-3
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_EPOCHS = 30

    learning_rate: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    num_epochs: int = DEFAULT_EPOCHS
    weight_decay: float = 1e-5
    _device: str = field(default="auto", repr=False)
    checkpoint_dir: str = "checkpoints"

    @property
    def device(self) -> torch.device:
        if self._device != "auto":
            return torch.device(self._device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @device.setter
    def device(self, val: str) -> None:
        self._device = val


@dataclass
class TrainingResult:
    """results from model training."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_time: float = 0.0


def create_datasets(
    diffraction: torch.Tensor,
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    train_ratio: float = 0.8,
) -> Tuple[TensorDataset, TensorDataset]:
    """create train/validation datasets from tensors."""
    n = len(diffraction)
    perm = torch.randperm(n)
    n_train = int(n * train_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # add channel dimension if needed
    def add_channel(x):
        return x.unsqueeze(1) if x.dim() == 3 else x

    train_ds = TensorDataset(
        add_channel(diffraction[train_idx]),
        add_channel(amplitude[train_idx]),
        add_channel(phase[train_idx]),
    )
    val_ds = TensorDataset(
        add_channel(diffraction[val_idx]),
        add_channel(amplitude[val_idx]),
        add_channel(phase[val_idx]),
    )

    return train_ds, val_ds


class Trainer:
    """simple training loop for amplitude/phase prediction models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: nn.Module | None = None,
    ):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)

        self.loss_fn = loss_fn or AmplitudePhaseLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(exist_ok=True)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        """run training loop."""
        result = TrainingResult()

        for epoch in range(self.config.num_epochs):
            # training phase
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                diff, amp, phase = [b.to(self.device) for b in batch]

                self.optimizer.zero_grad()
                pred_amp, pred_phase = self.model(diff)
                loss = self.loss_fn(pred_amp, pred_phase, amp, phase)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= max(1, len(train_loader))
            result.train_losses.append(epoch_loss)

            # validation phase
            val_loss = self._evaluate(val_loader)
            result.val_losses.append(val_loss)

            # save best model
            if val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                result.best_epoch = epoch
                torch.save(self.model.state_dict(), self.ckpt_dir / "best.pt")

        return result

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in loader:
            diff, amp, phase = [b.to(self.device) for b in batch]
            pred_amp, pred_phase = self.model(diff)
            loss = self.loss_fn(pred_amp, pred_phase, amp, phase)
            total_loss += loss.item()

        return total_loss / max(1, len(loader))
