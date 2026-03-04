"""Synthetic corpus utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .core import CorpusDocument


@dataclass(slots=True)
class SyntheticCorpus:
    """In-memory synthetic corpus for retrieval experiments."""

    documents: list[CorpusDocument]

    @property
    def keys(self) -> Tensor:
        return torch.stack([doc.key for doc in self.documents], dim=0)

    @property
    def values(self) -> Tensor:
        return torch.stack([doc.value for doc in self.documents], dim=0)

    @property
    def doc_ids(self) -> Tensor:
        return torch.LongTensor([doc.doc_id for doc in self.documents])


def build_toy_corpus(num_docs: int, dim: int, seed: int = 0) -> SyntheticCorpus:
    """Build a deterministic synthetic corpus with unique keys and values."""
    g = torch.Generator().manual_seed(seed)
    docs: list[CorpusDocument] = []
    for idx in range(num_docs):
        key = torch.randn(dim, generator=g)
        value = torch.randn(dim, generator=g)
        docs.append(
            CorpusDocument(
                doc_id=idx,
                key=key,
                value=value,
                payload={"drug_id": idx, "binary_label": int(idx % 2 == 0), "continuous_target": float(idx)},
            )
        )
    return SyntheticCorpus(documents=docs)
