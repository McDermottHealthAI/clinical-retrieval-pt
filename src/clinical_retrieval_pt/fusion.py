from __future__ import annotations

from typing import Any

from .types import FusionOutput


class ReplaceFusion:
    """Fusion that outputs retrieval memory and discards patient state."""

    def fuse(
        self,
        *,
        patient_state: Any,
        retrieval_memory: Any,
        retrieval_step_ids: Any | None = None,
        doc_attention_mask: Any | None = None,
    ) -> FusionOutput:
        """Return retrieval memory and ignore patient state."""
        del patient_state, retrieval_step_ids, doc_attention_mask
        return FusionOutput(fused_state=retrieval_memory)
