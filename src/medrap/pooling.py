"""Pooling modules for reducing fused representations before prediction."""

from torch import Tensor, nn


class IdentityPooling(nn.Module):
    """Tabular pooling module that returns fused state unchanged.

    For this no-op pooling module, ``D_pool = D_fused``.
    """

    def __init__(self) -> None:
        super().__init__()

    def pool(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return fused_state unchanged.

        Args:
            fused_state: Tensor with shape ``(B, D_fused)``.
            attention_mask: Optional mask tensor, ignored by this module.

        Returns:
            The unchanged ``fused_state`` tensor, with shape ``(B, D_fused)``.

        Examples:
            >>> import torch
            >>> pooling = IdentityPooling()
            >>> fused_state = torch.randn(2, 4)
            >>> out = pooling.pool(fused_state)
            >>> out.shape
            torch.Size([2, 4])
            >>> torch.equal(out, fused_state)
            True
        """
        return fused_state

    def forward(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Call ``pool``."""
        return self.pool(fused_state, attention_mask=attention_mask)


class MaskedMeanPooling(nn.Module):
    """Sequence pooling module that averages over the sequence dimension."""

    def __init__(self) -> None:
        super().__init__()

    def pool(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Average over the sequence dimension with an optional mask.

        Args:
            fused_state: Tensor with shape ``(B, S_fused, D_fused)``.
            attention_mask: Optional boolean mask with shape ``(B, S_fused)``.

        Returns:
            A tensor with shape ``(B, D_fused)``. For this module,
            ``D_pool = D_fused``.

        Examples:
            >>> import torch
            >>> pooling = MaskedMeanPooling()
            >>> fused_state = torch.FloatTensor(
            ...     [
            ...         [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            ...         [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            ...     ]
            ... )
            >>> attention_mask = torch.BoolTensor([[True, True, False], [True, False, True]])
            >>> out = pooling.pool(fused_state, attention_mask=attention_mask)
            >>> out.tolist()
            [[2.0, 3.0], [7.0, 8.0]]
        """
        if fused_state.ndim != 3:
            raise ValueError(
                f"MaskedMeanPooling expects fused_state shaped (B, S, D), got {tuple(fused_state.shape)}"
            )
        if attention_mask is None:
            return fused_state.float().mean(dim=1)

        mask = attention_mask.bool()
        if mask.shape != fused_state.shape[:2]:
            raise ValueError(
                "attention_mask must match the first two fused_state dimensions. "
                f"Got mask={tuple(mask.shape)}, fused_state={tuple(fused_state.shape)}"
            )
        expanded_mask = mask.unsqueeze(-1)
        counts = expanded_mask.sum(dim=1).clamp_min(1).to(dtype=fused_state.dtype)
        return (fused_state.float() * expanded_mask).sum(dim=1) / counts

    def forward(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Call ``pool``."""
        return self.pool(fused_state, attention_mask=attention_mask)
