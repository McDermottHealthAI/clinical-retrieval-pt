"""Tiny synthetic LightningDataModule used only by tests."""

from __future__ import annotations

import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor
from torch.utils.data import DataLoader

from medrap.batch_adapter import LABEL_FIELDS, MEDSSupervisedBatch

try:
    import lightning
except ModuleNotFoundError:
    lightning = None  # type: ignore[assignment]


def _missing_lightning() -> ModuleNotFoundError:
    return ModuleNotFoundError("lightning is required for DemoMedRAPDataModule")


if lightning is None:

    class DemoMedRAPDataModule:  # type: ignore[no-redef]
        """Placeholder when Lightning is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise _missing_lightning()

else:

    class DemoMedRAPDataModule(lightning.LightningDataModule):
        """In-memory synthetic datamodule for smoke-level tests."""

        def __init__(
            self,
            *,
            batch_size: int = 2,
            seq_len: int = 3,
            vocab_size: int = 128,
            label_field: str = "boolean_value",
            label_cardinality: int = 3,
            num_train_batches: int = 4,
            num_val_batches: int = 2,
            num_test_batches: int = 2,
            seed: int = 13,
        ) -> None:
            super().__init__()
            if label_field not in LABEL_FIELDS:
                raise ValueError(f"label_field must be one of {LABEL_FIELDS!r}, got {label_field!r}")

            self.batch_size = int(batch_size)
            self.seq_len = int(seq_len)
            self.vocab_size = int(vocab_size)
            self.label_field = label_field
            self.label_cardinality = int(label_cardinality)
            self.num_train_batches = int(num_train_batches)
            self.num_val_batches = int(num_val_batches)
            self.num_test_batches = int(num_test_batches)
            self.seed = int(seed)

            self._train_batches: list[MEDSSupervisedBatch] = []
            self._val_batches: list[MEDSSupervisedBatch] = []
            self._test_batches: list[MEDSSupervisedBatch] = []

        def _make_targets(self, generator: torch.Generator) -> Tensor:
            if self.label_field == "boolean_value":
                return torch.randint(
                    low=0,
                    high=2,
                    size=(self.batch_size,),
                    generator=generator,
                    dtype=torch.bool,
                )
            if self.label_field in {"integer_value", "categorical_value"}:
                if self.label_cardinality < 2:
                    raise ValueError("label_cardinality must be >= 2 for integer/categorical labels")
                return torch.randint(
                    low=0,
                    high=self.label_cardinality,
                    size=(self.batch_size,),
                    generator=generator,
                    dtype=torch.long,
                )
            if self.label_field == "float_value":
                return torch.rand((self.batch_size,), generator=generator, dtype=torch.float32)
            raise ValueError(f"Unsupported label_field: {self.label_field!r}")

        def _make_batch(self, generator: torch.Generator) -> MEDSSupervisedBatch:
            code = torch.randint(
                low=1,
                high=self.vocab_size,
                size=(self.batch_size, self.seq_len),
                generator=generator,
                dtype=torch.long,
            )
            batch = MEDSTorchBatch(
                code=code,
                numeric_value=torch.zeros((self.batch_size, self.seq_len), dtype=torch.float32),
                numeric_value_mask=torch.zeros((self.batch_size, self.seq_len), dtype=torch.bool),
                time_delta_days=torch.zeros((self.batch_size, self.seq_len), dtype=torch.float32),
                boolean_value=self._make_targets(generator) if self.label_field == "boolean_value" else None,
            )

            labels = {
                "boolean_value": None,
                "integer_value": None,
                "float_value": None,
                "categorical_value": None,
            }
            if self.label_field != "boolean_value":
                labels[self.label_field] = self._make_targets(generator)

            return MEDSSupervisedBatch(batch=batch, **labels)

        def setup(self, stage: str | None = None) -> None:
            del stage
            g = torch.Generator().manual_seed(self.seed)

            self._train_batches = [self._make_batch(g) for _ in range(self.num_train_batches)]
            self._val_batches = [self._make_batch(g) for _ in range(self.num_val_batches)]
            self._test_batches = [self._make_batch(g) for _ in range(self.num_test_batches)]

        @staticmethod
        def _loader(data: list[MEDSSupervisedBatch]) -> DataLoader:
            return DataLoader(data, batch_size=None, shuffle=False)

        def train_dataloader(self) -> DataLoader:
            return self._loader(self._train_batches)

        def val_dataloader(self) -> DataLoader:
            return self._loader(self._val_batches)

        def test_dataloader(self) -> DataLoader:
            return self._loader(self._test_batches)
