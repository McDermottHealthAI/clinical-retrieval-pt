from torch import Tensor, nn


class IdentityHead(nn.Module):
    """Concrete head that returns pooled state unchanged."""

    def __init__(self) -> None:
        super().__init__()

    def predict(self, pooled_state: Tensor) -> Tensor:
        """Return pooled_state unchanged."""
        return pooled_state

    def forward(self, pooled_state: Tensor) -> Tensor:
        """Call ``predict``."""
        return self.predict(pooled_state)
