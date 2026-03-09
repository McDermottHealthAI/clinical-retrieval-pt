"""Prediction head modules mapping pooled states to task outputs.

These components consume pooled fused representations and produce model logits.
"""

from torch import Tensor, nn


class LinearHead(nn.Module):
    """Linear prediction head mapping pooled representations to task logits.

    Args:
        in_dim: Input pooled representation size ``D_pool``.
        out_dim: Output logit size ``C``.
    """

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def predict(self, pooled_state: Tensor) -> Tensor:
        """Apply the linear prediction head.

        Args:
            pooled_state: Tensor with shape ``(B, D_pool)``.

        Returns:
            Tensor with shape ``(B, C)``.

        Examples:
            >>> import torch
            >>> head = LinearHead(in_dim=2, out_dim=3)
            >>> pooled_state = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
            >>> out = head.predict(pooled_state)
            >>> tuple(out.shape)
            (2, 3)
            >>> out.dtype
            torch.float32
        """
        return self.linear(pooled_state.float())

    def forward(self, pooled_state: Tensor) -> Tensor:
        """Call ``predict``."""
        return self.predict(pooled_state)
