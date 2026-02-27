from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from .types import EncoderOutput


class IdentityEncoder(nn.Module):
    """Concrete encoder that treats input batch as already-encoded patient state."""

    def __init__(self) -> None:
        super().__init__()

    def encode(self, batch: Tensor) -> EncoderOutput:
        """Return the input batch directly as patient state."""
        return EncoderOutput(patient_state=batch)

    def forward(self, batch: Tensor) -> EncoderOutput:
        """Call ``encode``."""
        return self.encode(batch)


class MEDSCodeEncoder(nn.Module):
    """Simple encoder that forwards ``batch.code`` as patient state.

    This is a minimal concrete encoder for MEDS-style batches (for example
    ``meds_torchdata.MEDSTorchBatch`` in SM mode).
    """

    def __init__(self) -> None:
        super().__init__()

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Return ``batch.code`` as the encoded patient state."""
        return EncoderOutput(patient_state=batch.code)

    def forward(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Call ``encode``."""
        return self.encode(batch)
