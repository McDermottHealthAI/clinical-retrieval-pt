from torch import Tensor, nn

from .types import FusionOutput


class ReplaceFusion(nn.Module):
    """Fusion that outputs retrieval memory and discards patient state."""

    def __init__(self) -> None:
        super().__init__()

    def fuse(
        self,
        *,
        patient_state: Tensor,
        retrieval_memory: Tensor,
        retrieval_step_ids: Tensor | None = None,
        doc_attention_mask: Tensor | None = None,
    ) -> FusionOutput:
        """Return retrieval memory and ignore patient state."""
        del patient_state, retrieval_step_ids, doc_attention_mask
        return FusionOutput(fused_state=retrieval_memory)

    def forward(
        self,
        *,
        patient_state: Tensor,
        retrieval_memory: Tensor,
        retrieval_step_ids: Tensor | None = None,
        doc_attention_mask: Tensor | None = None,
    ) -> FusionOutput:
        """Call ``fuse``."""
        return self.fuse(
            patient_state=patient_state,
            retrieval_memory=retrieval_memory,
            retrieval_step_ids=retrieval_step_ids,
            doc_attention_mask=doc_attention_mask,
        )
