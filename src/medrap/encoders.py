"""Patient-side encoder modules for retrieval-augmented modeling.

These components convert MEDS batch inputs into dense patient representation used by query projection and
downstream fusion.
"""

from meds_torchdata import MEDSTorchBatch
from torch import nn

from .types import EncoderOutput


class MEDSCodeEncoder(nn.Module):
    """Simple encoder that forwards ``batch.code`` as patient state.

    This is a minimal scaffold encoder for MEDS-style batches in
    Subjec-Measurement mode. It performs no learned transformation. It reads
    ``batch.code`` from a ``MEDSTorchBatch`` and returns it unchanged as the
    encoded patient representation.
    """

    def __init__(self) -> None:
        super().__init__()

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Return ``batch.code`` as the encoded patient state.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, S_ehr)`` and is equal to ``batch.code``.

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> encoder = MEDSCodeEncoder()
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[11, 22, 0], [7, 3, 1]]),
            ...     numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            ...     numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
            ...     time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 3)
            >>> torch.equal(out.patient_state, batch.code)
            True
        """
        if batch.code is None:
            raise ValueError("MEDSCodeEncoder requires batch code to be present.")
        return EncoderOutput(patient_state=batch.code)

    def forward(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Call ``encode``."""
        return self.encode(batch)


class TokenEmbeddingEncoder(nn.Module):
    """Sequence encoder that maps ``batch.code`` to learned token embeddings.

    This is a minimal learned sequence encoder for MEDS-style batches. It reads
    ``batch.code`` from a ``MEDSTorchBatch`` and maps each code id to an
    embedding vector, producing a dense patient representation.

    Args:
        vocab_size: Size of the EHR code vocabulary.
        embedding_dim: Output hidden size ``D_ehr``.
    """

    def __init__(self, *, vocab_size: int = 1024, embedding_dim: int = 4) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Embed ``batch.code`` into a sequence hidden state.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, S_ehr, D_ehr)``.

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> encoder = TokenEmbeddingEncoder(vocab_size=8, embedding_dim=2)
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 0], [3, 4, 5]]),
            ...     numeric_value=torch.zeros((2, 3), dtype=torch.float32),
            ...     numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
            ...     time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 3, 2)
            >>> out.patient_state.dtype
            torch.float32
        """
        if batch.code is None:
            raise ValueError("TokenEmbeddingEncoder requires batch code to be present.")
        return EncoderOutput(patient_state=self.embedding(batch.code.long()))

    def forward(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Call ``encode``."""
        return self.encode(batch)
