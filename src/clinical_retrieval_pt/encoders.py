from __future__ import annotations

from typing import Any

from .types import EncoderOutput


class IdentityEncoder:
    """Concrete encoder that treats input batch as already-encoded patient state."""

    def encode(self, batch: Any) -> EncoderOutput:
        """Return the input batch directly as patient state."""
        return EncoderOutput(patient_state=batch)
