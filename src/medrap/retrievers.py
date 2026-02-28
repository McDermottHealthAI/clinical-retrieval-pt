from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .types import RetrieverOutput


class StaticRetriever(nn.Module):
    """Retriever that returns pre-specified retrieval outputs."""

    def __init__(
        self,
        *,
        doc_tokens: Tensor | Sequence[Sequence[int]],
        doc_attention_mask: Tensor | Sequence[Sequence[int]],
        doc_scores: Tensor | Sequence[Sequence[float]] | None = None,
        doc_ids: Tensor | Sequence[Sequence[int]] | None = None,
        doc_key_embeddings: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.doc_tokens = (
            doc_tokens if isinstance(doc_tokens, Tensor) else torch.as_tensor(doc_tokens, dtype=torch.long)
        )
        self.doc_attention_mask = (
            doc_attention_mask
            if isinstance(doc_attention_mask, Tensor)
            else torch.as_tensor(doc_attention_mask, dtype=torch.bool)
        )
        self.doc_scores = (
            doc_scores
            if isinstance(doc_scores, Tensor) or doc_scores is None
            else torch.as_tensor(doc_scores, dtype=torch.float32)
        )
        self.doc_ids = (
            doc_ids
            if isinstance(doc_ids, Tensor) or doc_ids is None
            else torch.as_tensor(doc_ids, dtype=torch.long)
        )
        self.doc_key_embeddings = doc_key_embeddings

    def retrieve(self, query_embeddings: Tensor) -> RetrieverOutput:
        """Ignore query embeddings and return fixed retrieval output."""
        del query_embeddings
        return RetrieverOutput(
            doc_tokens=self.doc_tokens,
            doc_attention_mask=self.doc_attention_mask,
            doc_scores=self.doc_scores,
            doc_ids=self.doc_ids,
            doc_key_embeddings=self.doc_key_embeddings,
        )

    def forward(self, query_embeddings: Tensor) -> RetrieverOutput:
        """Call ``retrieve``."""
        return self.retrieve(query_embeddings)
