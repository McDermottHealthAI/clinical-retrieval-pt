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


class LinearHead(nn.Module):
    """Trainable linear head over flattened pooled state."""

    def __init__(self, out_features: int = 2) -> None:
        super().__init__()
        self.out_features = int(out_features)
        self.linear = nn.LazyLinear(self.out_features)

    def predict(self, pooled_state: Tensor) -> Tensor:
        """Flatten pooled state to ``(B, F)`` and apply a linear classifier."""
        if pooled_state.ndim < 2:
            raise ValueError(f"Expected pooled_state with at least 2 dims, got {tuple(pooled_state.shape)}")
        flat = pooled_state.float().reshape(pooled_state.shape[0], -1)
        return self.linear(flat)

    def forward(self, pooled_state: Tensor) -> Tensor:
        """Call ``predict``."""
        return self.predict(pooled_state)
