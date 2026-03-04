"""Adapters for converting MEDS batches into supervised task batches."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

from meds_torchdata import MEDSTorchBatch
from torch import Tensor

LABEL_FIELDS: tuple[str, ...] = (
    "boolean_value",
    "integer_value",
    "float_value",
    "categorical_value",
)


@dataclass(slots=True)
class MEDSSupervisedBatch:
    """Wrapper for MEDS model inputs plus LabelSchema-compatible target fields."""

    batch: MEDSTorchBatch
    boolean_value: Tensor | None = None
    integer_value: Tensor | None = None
    float_value: Tensor | None = None
    categorical_value: Tensor | None = None


@dataclass(slots=True)
class AdaptedSupervisedBatch:
    """Task-ready supervised batch output produced by adapters."""

    model_input: MEDSTorchBatch
    targets: Tensor
    label_field: str
    metadata: dict[str, object] = dataclass_field(default_factory=dict)


class MEDSSupervisedBatchAdapter:
    """Extract model input + task targets from MEDS-native labeled batches."""

    def __init__(self, *, label_field: str | None = None) -> None:
        if label_field is not None and label_field not in LABEL_FIELDS:
            raise ValueError(f"Unsupported label_field={label_field!r}. Expected one of {LABEL_FIELDS!r}")
        self.label_field = label_field

    def _from_meds_batch(self, batch: MEDSTorchBatch) -> tuple[MEDSTorchBatch, dict[str, Tensor]]:
        labels: dict[str, Tensor] = {}
        for label_field_name in LABEL_FIELDS:
            value = getattr(batch, label_field_name, None)
            if isinstance(value, Tensor):
                labels[label_field_name] = value
        return batch, labels

    def _from_supervised_wrapper(
        self, batch: MEDSSupervisedBatch
    ) -> tuple[MEDSTorchBatch, dict[str, Tensor]]:
        labels: dict[str, Tensor] = {}
        for label_field_name in LABEL_FIELDS:
            value = getattr(batch, label_field_name)
            if isinstance(value, Tensor):
                labels[label_field_name] = value

        # Preserve MEDSTorchBatch-embedded label(s) if present.
        for label_field_name in LABEL_FIELDS:
            value = getattr(batch.batch, label_field_name, None)
            if isinstance(value, Tensor):
                labels.setdefault(label_field_name, value)

        return batch.batch, labels

    def adapt(self, batch: MEDSTorchBatch | MEDSSupervisedBatch) -> AdaptedSupervisedBatch:
        """Return task-ready labels from a MEDS batch structure."""
        if isinstance(batch, MEDSSupervisedBatch):
            model_input, labels = self._from_supervised_wrapper(batch)
        elif isinstance(batch, MEDSTorchBatch):
            model_input, labels = self._from_meds_batch(batch)
        else:
            raise TypeError(
                "Expected batch to be MEDSTorchBatch or MEDSSupervisedBatch for supervised training, "
                f"got {type(batch)!r}."
            )

        if self.label_field is not None:
            target = labels.get(self.label_field)
            if target is None:
                raise ValueError(
                    f"Configured label_field={self.label_field!r} not present in batch labels. "
                    f"Observed: {sorted(labels)}"
                )
            return AdaptedSupervisedBatch(
                model_input=model_input,
                targets=target,
                label_field=self.label_field,
                metadata={"available_label_fields": sorted(labels)},
            )

        if not labels:
            raise ValueError(
                f"No label tensors found in batch. Expected exactly one of: {', '.join(LABEL_FIELDS)}"
            )

        if len(labels) > 1:
            raise ValueError(
                "Multiple label tensors present in batch with no explicit label_field configured: "
                f"{sorted(labels)}. Set batch_adapter.label_field to disambiguate."
            )

        field, target = next(iter(labels.items()))
        return AdaptedSupervisedBatch(
            model_input=model_input,
            targets=target,
            label_field=field,
            metadata={"available_label_fields": sorted(labels)},
        )

    def __call__(self, batch: MEDSTorchBatch | MEDSSupervisedBatch) -> AdaptedSupervisedBatch:
        return self.adapt(batch)
