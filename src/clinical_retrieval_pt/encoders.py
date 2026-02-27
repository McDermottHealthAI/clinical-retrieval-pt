from __future__ import annotations

from typing import Any

from .types import EncoderOutput


class PatientEncoder:
    """Base encoder interface for RAP API v2."""

    def encode(self, batch: Any) -> EncoderOutput:
        """Encode a MEDS-like batch into patient_state.

        Raises:
            NotImplementedError: Always for base class.
        """
        raise NotImplementedError("PatientEncoder.encode is not implemented for base class.")


class IdentityEncoder(PatientEncoder):
    """Concrete encoder that treats input batch as already-encoded patient state."""

    def encode(self, batch: Any) -> EncoderOutput:
        """Return the input batch directly as patient state."""
        return EncoderOutput(patient_state=batch, attention_mask=None)
