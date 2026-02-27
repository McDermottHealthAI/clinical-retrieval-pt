from __future__ import annotations

from typing import Any

from .types import QueryOutput


class QueryProjector:
    """Base query projection interface for RAP API v2."""

    def project(self, patient_state: Any, attention_mask: Any | None = None) -> QueryOutput:
        """Project patient_state into retrieval query space.

        Raises:
            NotImplementedError: Always for base class.
        """
        raise NotImplementedError("QueryProjector.project is not implemented base class.")


class IdentityQueryProjector(QueryProjector):
    """Concrete projector that forwards patient_state as query embeddings."""

    def project(self, patient_state: Any, attention_mask: Any | None = None) -> QueryOutput:
        """Return patient_state as query embeddings with no retrieval-step mapping."""
        return QueryOutput(query_embeddings=patient_state, retrieval_step_ids=None)
