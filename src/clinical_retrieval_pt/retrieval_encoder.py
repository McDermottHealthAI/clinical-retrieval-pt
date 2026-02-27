from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import RetrievalEncoderOutput, RetrieverOutput


class RetrievalEncoder:
    """Base retrieval encoder interface."""

    def encode(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Encode retrieved documents into retrieval memory.

        Raises:
            NotImplementedError: Always for base class.
        """
        raise NotImplementedError("RetrievalEncoder.encode is not implemented.")
