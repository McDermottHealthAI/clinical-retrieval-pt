from __future__ import annotations

from typing import Any

from .types import RetrieverOutput


class StaticRetriever:
    """Retriever that returns pre-specified retrieval outputs."""

    def __init__(
        self,
        *,
        doc_tokens: Any,
        doc_attention_mask: Any,
        doc_scores: Any | None = None,
        doc_ids: Any | None = None,
        doc_key_embeddings: Any | None = None,
    ) -> None:
        self.doc_tokens = doc_tokens
        self.doc_attention_mask = doc_attention_mask
        self.doc_scores = doc_scores
        self.doc_ids = doc_ids
        self.doc_key_embeddings = doc_key_embeddings

    def retrieve(self, query_embeddings: Any) -> RetrieverOutput:
        """Ignore query embeddings and return fixed retrieval output."""
        del query_embeddings
        return RetrieverOutput(
            doc_tokens=self.doc_tokens,
            doc_attention_mask=self.doc_attention_mask,
            doc_scores=self.doc_scores,
            doc_ids=self.doc_ids,
            doc_key_embeddings=self.doc_key_embeddings,
        )
