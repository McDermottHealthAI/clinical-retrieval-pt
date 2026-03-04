"""Batch adapter for semi-synthetic experiment dataloader batches."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from medrap.batch_adapter import AdaptedSupervisedBatch


class SemiSyntheticBatchAdapter:
    """Adapt semi-synthetic dataloader dict batches for core Lightning training."""

    def __init__(self, *, label_field: str = "target") -> None:
        self.label_field = str(label_field)

    def adapt(self, batch: dict[str, Any]) -> AdaptedSupervisedBatch:
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch for semi-synthetic adapter, got {type(batch)!r}")
        targets = batch.get("targets")
        if not isinstance(targets, Tensor):
            raise ValueError("Semi-synthetic batch must contain Tensor key 'targets'.")

        return AdaptedSupervisedBatch(
            model_input=batch,
            targets=targets,
            label_field=self.label_field,
            metadata={
                "example_ids": list(batch.get("example_ids", [])),
                "subject_ids": list(batch.get("subject_ids", [])),
            },
        )

    def __call__(self, batch: dict[str, Any]) -> AdaptedSupervisedBatch:
        return self.adapt(batch)
