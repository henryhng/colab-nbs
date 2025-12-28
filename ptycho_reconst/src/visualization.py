"""Optional plotting helpers (requires matplotlib)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for visualization") from exc

__all__ = ["plot_complex_object"]


def plot_complex_object(obj: torch.Tensor, title: str = "Reconstruction", figsize: Tuple[int, int] = (10, 4),
                        save_path: Optional[str] = None):
    arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else obj
    amp = np.abs(arr)
    phase = np.angle(arr)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    im1 = ax1.imshow(amp, cmap="gray")
    ax1.set_title(f"{title} amplitude")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    im2 = ax2.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax2.set_title(f"{title} phase")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
