"""Patient-side encoder modules for retrieval-augmented modeling.

These components convert MEDS batch inputs into dense patient representation used by query projection and
downstream fusion.
"""

from abc import ABC, abstractmethod

from meds_torchdata import MEDSTorchBatch
from torch import nn

from .types import EncoderOutput


class PatientEncoder(nn.Module, ABC):
    """Abstract base for all patient encoders.

    Subclasses must implement :meth:`encode`, which maps a ``MEDSTorchBatch``
    to an ``EncoderOutput``.  The ``forward`` method delegates to ``encode``
    so that the encoder can be used as a standard ``nn.Module``.
    """

    @abstractmethod
    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Encode a patient batch into a dense representation.

        Args:
            batch: A ``MEDSTorchBatch``.

        Returns:
            An ``EncoderOutput`` whose ``patient_state`` has shape
            ``(B, S_ehr, D_ehr)``.
        """

    def forward(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Call ``encode``."""
        return self.encode(batch)


class MEDSCodeEncoder(PatientEncoder):
    """Scaffold sequence encoder that casts ``batch.code`` to a float representation.

    This is a minimal, non-learned encoder for MEDS-style batches. It converts
    the integer code ids to floats and unsqueezes a trailing dimension so the
    output satisfies the sequence-mode shape contract ``(B, S_ehr, D_ehr)``
    with ``D_ehr = 1``.
    """

    def __init__(self) -> None:
        super().__init__()

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Return ``batch.code`` as a float tensor with a trailing embedding dim.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, S_ehr, 1)``.

        Examples:
            >>> encoder = MEDSCodeEncoder()
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[11, 22, 0], [7, 3, 1]]),
            ...     numeric_value=torch.zeros(2, 3),
            ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 3),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 3, 1)
            >>> out.patient_state.dtype
            torch.float32
        """
        return EncoderOutput(patient_state=batch.code.float().unsqueeze(-1))


class TokenEmbeddingEncoder(PatientEncoder):
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
            >>> encoder = TokenEmbeddingEncoder(vocab_size=8, embedding_dim=2)
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 0], [3, 4, 5]]),
            ...     numeric_value=torch.zeros(2, 3),
            ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 3),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 3, 2)
            >>> out.patient_state.dtype
            torch.float32
        """
        return EncoderOutput(patient_state=self.embedding(batch.code.long()))


class TabularEncoder(PatientEncoder):
    """Tabular encoder that pools a code sequence into a single patient vector.

    Embeds ``batch.code`` via a learned embedding table and mean-pools across the
    sequence dimension to produce a ``(B, 1, D_ehr)`` patient representation.

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
        """Embed and mean-pool ``batch.code`` into a tabular patient state.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, 1, D_ehr)``.

        Examples:
            >>> encoder = TabularEncoder(vocab_size=8, embedding_dim=2)
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 0], [3, 4, 5]]),
            ...     numeric_value=torch.zeros(2, 3),
            ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 3),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 1, 2)
            >>> out.patient_state.dtype
            torch.float32
            >>> tuple(encoder(batch).patient_state.shape)
            (2, 1, 2)
        """
        embedded = self.embedding(batch.code.long())  # (B, S_ehr, D_ehr)
        pooled = embedded.mean(dim=1, keepdim=True)  # (B, 1, D_ehr)
        return EncoderOutput(patient_state=pooled)
