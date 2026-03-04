"""Evaluation helpers for synthetic retrieval behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from medrap.retrieval_eval import retrieval_doc_ids_are_sample_dependent, top1_recall_at_1
from medrap.task import MeanAbsoluteErrorMetric

if TYPE_CHECKING:
    from torch import Tensor

    from .core import RetrievalResult


_MAE = MeanAbsoluteErrorMetric()


def retrieval_is_sample_dependent(result: RetrievalResult) -> bool:
    """Return True if retrieval varies across samples."""
    return retrieval_doc_ids_are_sample_dependent(result.doc_ids)


def oracle_recall_at_1(result: RetrievalResult, target_doc_ids: Tensor) -> Tensor:
    """Compute recall@1 against known target doc ids."""
    return top1_recall_at_1(result.doc_ids[:, 0], target_doc_ids)


def mean_absolute_error(preds: Tensor, targets: Tensor) -> Tensor:
    """Mean absolute error utility for synthetic regression checks."""
    return _MAE(preds, targets)
