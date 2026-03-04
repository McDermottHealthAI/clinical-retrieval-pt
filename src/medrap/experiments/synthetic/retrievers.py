"""Synthetic retriever baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from medrap.search import DotProductTopKSearch

from .core import RetrievalResult

if TYPE_CHECKING:
    from .corpus import SyntheticCorpus


@dataclass(slots=True)
class OracleRetriever:
    """Retriever that returns the provided target document exactly."""

    top_k: int = 1

    def retrieve(self, *, target_doc_ids: Tensor) -> RetrievalResult:
        if self.top_k != 1:
            raise ValueError("OracleRetriever currently supports top_k=1 only")
        doc_ids = target_doc_ids.long().unsqueeze(-1)
        scores = torch.ones_like(doc_ids, dtype=torch.float32)
        return RetrievalResult(doc_ids=doc_ids, scores=scores)


class LearnedKeyQueryRetriever(nn.Module):
    """Trainable query->key retriever over a fixed synthetic corpus."""

    def __init__(self, *, corpus: SyntheticCorpus, query_dim: int, top_k: int = 1) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.query_proj = nn.Linear(query_dim, corpus.keys.shape[1], bias=False)
        self.register_buffer("corpus_keys", corpus.keys.clone().float())
        self.register_buffer("corpus_doc_ids", corpus.doc_ids.clone().long())
        self._search = DotProductTopKSearch(self.corpus_keys, doc_ids=self.corpus_doc_ids)

    def score(self, queries: Tensor) -> Tensor:
        projected = self.query_proj(queries.float())
        return projected @ self.corpus_keys.T

    def retrieve(self, queries: Tensor) -> RetrievalResult:
        projected = self.query_proj(queries.float())
        top_doc_ids, top_scores = self._search.search(projected, top_k=self.top_k)
        return RetrievalResult(doc_ids=top_doc_ids, scores=top_scores)


class CorruptedKeyRetriever(nn.Module):
    """Retriever that applies a key-space corruption before retrieval."""

    def __init__(self, *, base: LearnedKeyQueryRetriever, corruption: Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("corruption", corruption.float())
        self.register_buffer("corrupted_keys", self.base.corpus_keys @ self.corruption)
        self._search = DotProductTopKSearch(
            self.corrupted_keys,
            doc_ids=self.base.corpus_doc_ids,
        )

    def retrieve(self, queries: Tensor) -> RetrievalResult:
        projected = self.base.query_proj(queries.float())
        top_doc_ids, top_scores = self._search.search(projected, top_k=self.base.top_k)
        return RetrievalResult(doc_ids=top_doc_ids, scores=top_scores)
