from .types import RetrievalEncoderOutput, RetrieverOutput


class IdentityRetrievalEncoder:
    """Concrete retrieval encoder that reuses retrieved tokens as memory."""

    def encode(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Return retrieval.doc_tokens as retrieval memory."""
        return RetrievalEncoderOutput(retrieval_memory=retrieval.doc_tokens)
