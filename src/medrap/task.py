"""Task abstractions for supervised MedRAP training.

These classes own supervision semantics (target preparation, loss, metrics, postprocessing) so the Lightning
module can remain task-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class TaskStepOutput:
    """Per-step task computation output."""

    loss: Tensor
    metrics: dict[str, Tensor]
    predictions: Tensor
    targets: Tensor


class BinaryAccuracy(nn.Module):
    """Binary accuracy computed from logits."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        preds = probs >= self.threshold
        refs = targets >= 0.5
        return (preds == refs).float().mean()


class CategoricalAccuracy(nn.Module):
    """Categorical accuracy computed from class logits."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        preds = torch.argmax(logits, dim=-1)
        return (preds == targets).float().mean()


class MeanSquaredErrorMetric(nn.Module):
    """Mean squared error metric."""

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return torch.mean((preds - targets) ** 2)


class MeanAbsoluteErrorMetric(nn.Module):
    """Mean absolute error metric."""

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return torch.mean(torch.abs(preds - targets))


class SupervisedTask(nn.Module, ABC):
    """Abstract task interface used by ``MedRAPLightningModule``."""

    def __init__(
        self,
        *,
        label_field: str,
        output_dim: int,
        loss: nn.Module,
        metrics: dict[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.label_field = label_field
        self.output_dim = int(output_dim)
        self.loss_fn = loss
        self.metrics = nn.ModuleDict(dict(metrics or {}))

    @abstractmethod
    def prepare_predictions(self, predictions: Tensor) -> Tensor:
        """Shape/cast model outputs for the task loss/metrics."""

    @abstractmethod
    def prepare_targets(self, targets: Tensor) -> Tensor:
        """Shape/cast raw label tensors for the task loss/metrics."""

    def step(self, predictions: Tensor, targets: Tensor) -> TaskStepOutput:
        """Compute task loss and metrics for one optimization step."""
        preds = self.prepare_predictions(predictions)
        refs = self.prepare_targets(targets)
        loss = self.loss_fn(preds, refs)
        metrics = {name: metric(preds, refs) for name, metric in self.metrics.items()}
        return TaskStepOutput(loss=loss, metrics=metrics, predictions=preds, targets=refs)


class BinaryClassificationTask(SupervisedTask):
    """Boolean label task using binary logits and BCE loss."""

    def __init__(
        self,
        *,
        label_field: str = "boolean_value",
        output_dim: int = 1,
        loss: nn.Module | None = None,
        metrics: dict[str, nn.Module] | None = None,
    ) -> None:
        super().__init__(
            label_field=label_field,
            output_dim=output_dim,
            loss=loss if loss is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics if metrics is not None else {"accuracy": BinaryAccuracy()},
        )
        if self.output_dim != 1:
            raise ValueError(f"BinaryClassificationTask requires output_dim=1, got {self.output_dim}")

    def prepare_predictions(self, predictions: Tensor) -> Tensor:
        x = predictions.float()
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x.squeeze(1)
        raise ValueError(f"Expected binary logits shaped (B,) or (B,1), got {tuple(x.shape)}")

    def prepare_targets(self, targets: Tensor) -> Tensor:
        y = targets.float()
        if y.ndim == 1:
            return y
        if y.ndim == 2 and y.shape[1] == 1:
            return y.squeeze(1)
        raise ValueError(f"Expected binary targets shaped (B,) or (B,1), got {tuple(y.shape)}")


class CategoricalClassificationTask(SupervisedTask):
    """Categorical class-index task using cross-entropy loss."""

    def __init__(
        self,
        *,
        num_classes: int,
        label_field: str = "categorical_value",
        output_dim: int | None = None,
        loss: nn.Module | None = None,
        metrics: dict[str, nn.Module] | None = None,
    ) -> None:
        n_classes = int(num_classes)
        if n_classes < 2:
            raise ValueError("num_classes must be >= 2 for categorical classification")
        out_dim = int(output_dim) if output_dim is not None else n_classes
        if out_dim != n_classes:
            raise ValueError(f"output_dim ({out_dim}) must equal num_classes ({n_classes})")

        super().__init__(
            label_field=label_field,
            output_dim=out_dim,
            loss=loss if loss is not None else nn.CrossEntropyLoss(),
            metrics=metrics if metrics is not None else {"accuracy": CategoricalAccuracy()},
        )
        self.num_classes = n_classes

    def prepare_predictions(self, predictions: Tensor) -> Tensor:
        x = predictions.float()
        if x.ndim != 2 or x.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected categorical logits shaped (B,{self.num_classes}), got {tuple(x.shape)}"
            )
        return x

    def prepare_targets(self, targets: Tensor) -> Tensor:
        y = targets.long()
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        if y.ndim != 1:
            raise ValueError(f"Expected categorical targets shaped (B,) or (B,1), got {tuple(y.shape)}")
        return y


class RegressionTask(SupervisedTask):
    """Float target regression task."""

    def __init__(
        self,
        *,
        label_field: str = "float_value",
        output_dim: int = 1,
        loss: nn.Module | None = None,
        metrics: dict[str, nn.Module] | None = None,
    ) -> None:
        super().__init__(
            label_field=label_field,
            output_dim=output_dim,
            loss=loss if loss is not None else nn.MSELoss(),
            metrics=(
                metrics
                if metrics is not None
                else {
                    "mse": MeanSquaredErrorMetric(),
                    "mae": MeanAbsoluteErrorMetric(),
                }
            ),
        )
        if self.output_dim != 1:
            raise ValueError(f"RegressionTask requires output_dim=1, got {self.output_dim}")

    def prepare_predictions(self, predictions: Tensor) -> Tensor:
        x = predictions.float()
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x.squeeze(1)
        raise ValueError(f"Expected regression predictions shaped (B,) or (B,1), got {tuple(x.shape)}")

    def prepare_targets(self, targets: Tensor) -> Tensor:
        y = targets.float()
        if y.ndim == 1:
            return y
        if y.ndim == 2 and y.shape[1] == 1:
            return y.squeeze(1)
        raise ValueError(f"Expected regression targets shaped (B,) or (B,1), got {tuple(y.shape)}")
