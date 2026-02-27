from typing import Any

from meds_torchdata import MEDSTorchBatch

from .types import EncoderOutput


class IdentityEncoder:
    """Concrete encoder that treats input batch as already-encoded patient state."""

    def encode(self, batch: Any) -> EncoderOutput:
        """Return the input batch directly as patient state."""
        return EncoderOutput(patient_state=batch)


class MEDSCodeEncoder:
    """Simple encoder that forwards ``batch.code`` as patient state.

    This is a minimal concrete encoder for MEDS-style batches (for example
    ``meds_torchdata.MEDSTorchBatch`` in SM mode).
    """

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Return ``batch.code`` as the encoded patient state."""
        return EncoderOutput(patient_state=batch.code)
