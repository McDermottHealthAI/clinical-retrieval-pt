"""DataModule for the MIMIC-IV demo semi-synthetic retrieval experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    import lightning
except ModuleNotFoundError:
    lightning = None  # type: ignore[assignment]


def _missing_lightning() -> ModuleNotFoundError:
    return ModuleNotFoundError("lightning is required for SemiSyntheticDataModule")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class _SemiSyntheticDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


if lightning is None:

    class SemiSyntheticDataModule:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise _missing_lightning()

else:

    class SemiSyntheticDataModule(lightning.LightningDataModule):
        """Loads prepared examples + synthetic corpus artifacts and builds train/val/test loaders."""

        def __init__(
            self,
            *,
            prepared_dir: str,
            corpus_dir: str,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_examples_missing_docs: bool = True,
        ) -> None:
            super().__init__()
            self.prepared_dir = Path(prepared_dir).expanduser().resolve()
            self.corpus_dir = Path(corpus_dir).expanduser().resolve()
            self.batch_size = int(batch_size)
            self.num_workers = int(num_workers)
            self.pin_memory = bool(pin_memory)
            self.drop_examples_missing_docs = bool(drop_examples_missing_docs)

            self.doc_features: Tensor | None = None
            self.doc_labels: Tensor | None = None
            self.doc_ids: Tensor | None = None
            self.doc_drug_names: list[str] = []
            self.patient_dim: int | None = None

            self._train: _SemiSyntheticDataset | None = None
            self._val: _SemiSyntheticDataset | None = None
            self._test: _SemiSyntheticDataset | None = None

        def _build_rows(self, examples: list[dict[str, Any]], example_ids: set[str]) -> list[dict[str, Any]]:
            if self.doc_features is None or self.doc_labels is None or self.doc_ids is None:
                raise RuntimeError("Corpus tensors are not loaded")

            vocab = _read_json(self.prepared_dir / "drug_vocab.json")
            vocab_index = {name: idx for idx, name in enumerate(vocab)}
            self.patient_dim = len(vocab_index)
            doc_by_drug = {drug_name: idx for idx, drug_name in enumerate(self.doc_drug_names)}

            rows: list[dict[str, Any]] = []
            for example in examples:
                if example["example_id"] not in example_ids:
                    continue

                feature = torch.zeros((len(vocab_index),), dtype=torch.float32)
                for drug_name in example["drug_names"]:
                    if drug_name in vocab_index:
                        feature[vocab_index[drug_name]] = 1.0

                true_doc_ids = [doc_by_drug[drug] for drug in example["drug_names"] if drug in doc_by_drug]
                true_positive_doc_ids = [
                    doc_by_drug[drug]
                    for drug in example.get("positive_drug_names", [])
                    if drug in doc_by_drug
                ]

                if self.drop_examples_missing_docs and not true_doc_ids:
                    continue

                rows.append(
                    {
                        "example_id": example["example_id"],
                        "subject_id": str(example["subject_id"]),
                        "drug_names": list(example["drug_names"]),
                        "patient_features": feature,
                        "target": torch.tensor(float(example["target"]), dtype=torch.float32),
                        "true_doc_ids": true_doc_ids,
                        "true_positive_doc_ids": true_positive_doc_ids,
                    }
                )
            return rows

        @staticmethod
        def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "example_ids": [row["example_id"] for row in batch],
                "subject_ids": [row["subject_id"] for row in batch],
                "drug_names": [row["drug_names"] for row in batch],
                "patient_features": torch.stack([row["patient_features"] for row in batch], dim=0),
                "targets": torch.stack([row["target"] for row in batch], dim=0),
                "true_doc_ids": [row["true_doc_ids"] for row in batch],
                "true_positive_doc_ids": [row["true_positive_doc_ids"] for row in batch],
            }

        def setup(self, stage: str | None = None) -> None:
            del stage
            examples = _read_jsonl(self.prepared_dir / "examples.jsonl")
            splits = _read_json(self.prepared_dir / "splits.json")
            corpus_payload = torch.load(self.corpus_dir / "features.pt", map_location="cpu")

            self.doc_features = corpus_payload["doc_features"].float()
            self.doc_labels = corpus_payload["doc_labels"].long()
            self.doc_ids = corpus_payload["doc_ids"].long()
            self.doc_drug_names = list(corpus_payload["drug_names"])

            train_rows = self._build_rows(examples, set(splits["train"]))
            val_rows = self._build_rows(examples, set(splits["val"]))
            test_rows = self._build_rows(examples, set(splits["test"]))

            self._train = _SemiSyntheticDataset(train_rows)
            self._val = _SemiSyntheticDataset(val_rows)
            self._test = _SemiSyntheticDataset(test_rows)

        def train_dataloader(self) -> DataLoader:
            if self._train is None:
                raise RuntimeError("setup() must be called before requesting dataloaders")
            return DataLoader(
                self._train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate,
            )

        def val_dataloader(self) -> DataLoader:
            if self._val is None:
                raise RuntimeError("setup() must be called before requesting dataloaders")
            return DataLoader(
                self._val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate,
            )

        def test_dataloader(self) -> DataLoader:
            if self._test is None:
                raise RuntimeError("setup() must be called before requesting dataloaders")
            return DataLoader(
                self._test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate,
            )
