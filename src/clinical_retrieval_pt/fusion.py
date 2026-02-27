from __future__ import annotations

from typing import Any

from .types import FusionOutput


class Fusion:
    """Base fusion interface for RAP API v2."""

    def fuse(
        self,
        *,
        patient_state: Any,
        retrieval_memory: Any,
        retrieval_step_ids: Any | None = None,
        doc_attention_mask: Any | None = None,
    ) -> FusionOutput:
        """Fuse patient_state with retrieval memory.

        Raises:
            NotImplementedError: Always for base class.
        """
        raise NotImplementedError("Fusion.fuse is not implemented for base class.")


class ReplaceFusion(Fusion):
    """Concrete fusion for replacement mode (Config C-style behavior)."""

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
