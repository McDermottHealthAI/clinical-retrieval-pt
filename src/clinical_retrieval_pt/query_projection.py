from __future__ import annotations

from typing import Any

from .types import QueryOutput


class IdentityQueryProjector:
    """Concrete projector that forwards patient_state as query embeddings."""

    def project(self, patient_state: Any) -> QueryOutput:
        """Return patient_state as query embeddings with no retrieval-step mapping."""
        return QueryOutput(query_embeddings=patient_state, retrieval_step_ids=None)
