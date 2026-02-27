from torch import Tensor, nn

from .types import QueryOutput


class IdentityQueryProjector(nn.Module):
    """Concrete projector that forwards patient_state as query embeddings."""

    def __init__(self) -> None:
        super().__init__()

    def project(self, patient_state: Tensor) -> QueryOutput:
        """Return patient_state as query embeddings with no retrieval-step mapping."""
        return QueryOutput(query_embeddings=patient_state, retrieval_step_ids=None)

    def forward(self, patient_state: Tensor) -> QueryOutput:
        """Call ``project``."""
        return self.project(patient_state)
