"""Fusion modules for combining patient and retrieval representations.

These components define how retrieved memory contributes to downstream model states before pooling and
prediction.
"""

import torch
from torch import nn

from .types import FusionInput, FusionOutput


class ReplaceFusion(nn.Module):
    """Fusion module that uses retrieved memory as the fused output."""

    def __init__(self) -> None:
        super().__init__()

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Return retrieval memory as fused state.

        Args:
            fusion_input: ``FusionInput`` containing patient and retrieval
                tensors. In the minimal replacement case,
                ``fusion_input.retrieval_memory`` is typically shaped
                ``(B, D_mem)`` or ``(B, *M)``.

        Returns:
            ``FusionOutput`` with ``fused_state`` equal to
            ``fusion_input.retrieval_memory`` and therefore shaped ``(B, *M)``.

        Examples:
            >>> import torch
            >>> from medrap.types import FusionInput
            >>> fusion = ReplaceFusion()
            >>> fusion_input = FusionInput(
            ...     patient_state=torch.randn(2, 3),
            ...     retrieval_memory=torch.randn(2, 1, 4),
            ... )
            >>> out = fusion.fuse(fusion_input)
            >>> out.fused_state.shape
            torch.Size([2, 1, 4])
            >>> torch.equal(out.fused_state, fusion_input.retrieval_memory)
            True
        """
        return FusionOutput(fused_state=fusion_input.retrieval_memory)

    def forward(self, fusion_input: FusionInput) -> FusionOutput:
        """Call ``fuse``."""
        return self.fuse(fusion_input)


class ConcatFusion(nn.Module):
    """Tabular fusion module that concatenates patient state with retrieval memory."""

    def __init__(self) -> None:
        super().__init__()

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Concatenate tabular patient and retrieval representations.

        Args:
            fusion_input: ``FusionInput`` with:
                - ``patient_state`` shaped ``(B, D_ehr)``
                - ``retrieval_memory`` shaped ``(B, D_mem)``

        Returns:
            ``FusionOutput`` where ``fused_state`` has shape
            ``(B, D_ehr + D_mem)``.

        Examples:
            >>> import torch
            >>> fusion = ConcatFusion()
            >>> fusion_input = FusionInput(
            ...     patient_state=torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]]),
            ...     retrieval_memory=torch.FloatTensor([[10.0, 20.0], [30.0, 40.0]]),
            ... )
            >>> out = fusion.fuse(fusion_input)
            >>> out.fused_state.tolist()
            [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]
        """
        patient_state = fusion_input.patient_state
        retrieval_memory = fusion_input.retrieval_memory
        if patient_state.ndim != 2 or retrieval_memory.ndim != 2:
            raise ValueError(
                "ConcatFusion expects tabular inputs shaped (B, D). "
                f"Got patient_state={tuple(patient_state.shape)}, "
                f"retrieval_memory={tuple(retrieval_memory.shape)}"
            )
        return FusionOutput(fused_state=torch.cat((patient_state.float(), retrieval_memory.float()), dim=-1))

    def forward(self, fusion_input: FusionInput) -> FusionOutput:
        """Call ``fuse``."""
        return self.fuse(fusion_input)
