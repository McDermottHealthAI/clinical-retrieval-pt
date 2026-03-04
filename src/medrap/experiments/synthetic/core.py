"""Core synthetic experiment data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(slots=True)
class CorpusDocument:
    """Single synthetic corpus document."""

    doc_id: int
    key: Tensor
    value: Tensor
    payload: dict[str, object]


@dataclass(slots=True)
class PatientSample:
    """Single synthetic patient sample used for retrieval experiments."""

    sample_id: int
    query: Tensor
    metadata: dict[str, object]


@dataclass(slots=True)
class RetrievalResult:
    """Retriever output for one sample."""

    doc_ids: Tensor
    scores: Tensor


@dataclass(slots=True)
class RetrievalExperimentBatch:
    """Batch of synthetic retrieval queries and target metadata."""

    queries: Tensor
    target_doc_ids: Tensor
    metadata: dict[str, Tensor]
