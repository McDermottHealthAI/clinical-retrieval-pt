from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
