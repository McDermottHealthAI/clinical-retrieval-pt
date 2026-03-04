"""Training metrics used by Lightning modules."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ClassificationMetrics(nn.Module):
    """Simple per-batch classification metrics."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)

    def forward(self, logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape (B, C), got {tuple(logits.shape)}")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected logits second dim == num_classes ({self.num_classes}), got {logits.shape[1]}"
            )

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == targets).float().mean()
        return {"accuracy": acc}
