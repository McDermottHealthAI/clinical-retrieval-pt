from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import EncoderOutput


class PatientEncoder:
    """Base encoder interface for RAP API v2."""

    def encode(self, batch: Any) -> EncoderOutput:
        """Encode a MEDS-like batch into patient_state.

        Raises:
            NotImplementedError: Always for base class.
        """
        raise NotImplementedError("PatientEncoder.encode is not implemented for base class.")
