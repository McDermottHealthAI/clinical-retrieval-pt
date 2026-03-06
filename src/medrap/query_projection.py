"""Query projection modules for mapping patient state into retrieval space.

These components convert encoded patient representations into retrieval queries.
"""

from torch import Tensor, nn

from .types import QueryOutput


class LinearQueryProjector(nn.Module):
    """Tabular query projector producing one retrieval query per patient.

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def project(self, patient_state: Tensor) -> QueryOutput:
        """Project a tabular patient state into a single retrieval query.

        Args:
            patient_state: Tensor with shape ``(B, D_ehr)``.

        Returns:
            ``QueryOutput`` with:
                - ``query_embeddings`` shaped ``(B, 1, D_ret)``
                - ``retrieval_step_ids=None``

        Examples:
            >>> import torch
            >>> projector = LinearQueryProjector(in_dim=2, out_dim=3)
            >>> patient_state = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
            >>> out = projector.project(patient_state)
            >>> tuple(out.query_embeddings.shape)
            (2, 1, 3)
            >>> out.query_embeddings.dtype
            torch.float32
        """
        if patient_state.ndim != 2:
            raise ValueError(
                "LinearQueryProjector expects patient_state shaped (B, D_ehr), "
                f"got {tuple(patient_state.shape)}"
            )
        return QueryOutput(query_embeddings=self.linear(patient_state.float()).unsqueeze(1))

    def forward(self, patient_state: Tensor) -> QueryOutput:
        """Call ``project``."""
        return self.project(patient_state)


class SequenceMeanQueryProjector(nn.Module):
    """Sequence query projector that mean-pools over the EHR sequence.

    This is a minimal sequence baseline. It reduces ``patient_state`` across the
    sequence dimension and emits a single retrieval query per patient.

    Args:
        out_dim: Retrieval query size ``D_ret``.
    """

    def __init__(self, *, out_dim: int) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.linear = nn.LazyLinear(self.out_dim)

    def project(self, patient_state: Tensor) -> QueryOutput:
        """Mean-pool over sequence positions and emit one query per sample.

        Args:
            patient_state: Encoded patient tensor with shape ``(B, S_ehr)`` or
                ``(B, S_ehr, D_ehr)``.

        Returns:
            ``QueryOutput`` with:
                - ``query_embeddings`` shaped ``(B, 1, D_ret)``
                - ``retrieval_step_ids=None``

        Examples:
            >>> import torch
            >>> projector = SequenceMeanQueryProjector(out_dim=2)
            >>> patient_state = torch.FloatTensor(
            ...     [
            ...         [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ...         [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
            ...     ]
            ... )
            >>> out = projector.project(patient_state)
            >>> tuple(out.query_embeddings.shape)
            (2, 1, 2)
            >>> out.query_embeddings.dtype
            torch.float32
        """
        x = patient_state.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError(
                "SequenceMeanQueryProjector expects patient_state shaped "
                f"(B, S_ehr) or (B, S_ehr, D_ehr), got {tuple(patient_state.shape)}"
            )
        pooled = x.mean(dim=1)
        return QueryOutput(query_embeddings=self.linear(pooled).unsqueeze(1))

    def forward(self, patient_state: Tensor) -> QueryOutput:
        """Call ``project``."""
        return self.project(patient_state)
