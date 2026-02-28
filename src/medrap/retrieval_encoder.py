from torch import nn

from .types import RetrievalEncoderOutput, RetrieverOutput


class IdentityRetrievalEncoder(nn.Module):
    """Concrete retrieval encoder that reuses retrieved tokens as memory."""

    def __init__(self) -> None:
        super().__init__()

    def encode(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Return retrieval.doc_tokens as retrieval memory."""
        return RetrievalEncoderOutput(retrieval_memory=retrieval.doc_tokens)

    def forward(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Call ``encode``."""
        return self.encode(retrieval)
