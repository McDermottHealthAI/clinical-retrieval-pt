from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import RetrieverOutput


class Retriever:
    """Base retriever interface."""

    def retrieve(self, query_embeddings: Any) -> RetrieverOutput:
        """Retrieve documents from query embeddings.

        Raises:
            NotImplementedError: Always for base class.
        """
        raise NotImplementedError("Retriever.retrieve is not implemented in base class.")
