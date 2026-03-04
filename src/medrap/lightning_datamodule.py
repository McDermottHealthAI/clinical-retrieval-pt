"""Tiny synthetic LightningDataModule for MedRAP scaffolding."""

from __future__ import annotations

import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor
from torch.utils.data import DataLoader

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
        """In-memory synthetic data module for fast training/eval smoke runs."""

        def __init__(
            self,
            *,
            batch_size: int = 2,
            seq_len: int = 3,
            vocab_size: int = 128,
            num_classes: int = 2,
            num_train_batches: int = 4,
            num_val_batches: int = 2,
            num_test_batches: int = 2,
            seed: int = 13,
        ) -> None:
            super().__init__()
            self.batch_size = int(batch_size)
            self.seq_len = int(seq_len)
            self.vocab_size = int(vocab_size)
            self.num_classes = int(num_classes)
            self.num_train_batches = int(num_train_batches)
            self.num_val_batches = int(num_val_batches)
            self.num_test_batches = int(num_test_batches)
            self.seed = int(seed)

            self._train_batches: list[tuple[MEDSTorchBatch, Tensor]] = []
            self._val_batches: list[tuple[MEDSTorchBatch, Tensor]] = []
            self._test_batches: list[tuple[MEDSTorchBatch, Tensor]] = []

        def _make_batch(self, generator: torch.Generator) -> tuple[MEDSTorchBatch, Tensor]:
            code = torch.randint(
                low=1,
                high=self.vocab_size,
                size=(self.batch_size, self.seq_len),
                generator=generator,
                dtype=torch.long,
            )
            targets = torch.randint(
                low=0,
                high=self.num_classes,
                size=(self.batch_size,),
                generator=generator,
                dtype=torch.long,
            )
            batch = MEDSTorchBatch(
                code=code,
                numeric_value=torch.zeros((self.batch_size, self.seq_len), dtype=torch.float32),
                numeric_value_mask=torch.zeros((self.batch_size, self.seq_len), dtype=torch.bool),
                time_delta_days=torch.zeros((self.batch_size, self.seq_len), dtype=torch.float32),
            )
            return batch, targets

        def setup(self, stage: str | None = None) -> None:
            del stage
            g = torch.Generator().manual_seed(self.seed)

            self._train_batches = [self._make_batch(g) for _ in range(self.num_train_batches)]
            self._val_batches = [self._make_batch(g) for _ in range(self.num_val_batches)]
            self._test_batches = [self._make_batch(g) for _ in range(self.num_test_batches)]

        @staticmethod
        def _loader(data: list[tuple[MEDSTorchBatch, Tensor]]) -> DataLoader:
            # Each dataset item is already a complete batch tuple.
            return DataLoader(data, batch_size=None, shuffle=False)

        def train_dataloader(self) -> DataLoader:
            return self._loader(self._train_batches)

        def val_dataloader(self) -> DataLoader:
            return self._loader(self._val_batches)

        def test_dataloader(self) -> DataLoader:
            return self._loader(self._test_batches)
