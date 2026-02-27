from torch import Tensor, nn


class IdentityPooling(nn.Module):
    """Concrete pooling that returns fused_state unchanged."""

    def __init__(self) -> None:
        super().__init__()

    def pool(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return fused_state unchanged."""
        del attention_mask
        return fused_state

    def forward(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Call ``pool``."""
        return self.pool(fused_state, attention_mask=attention_mask)
