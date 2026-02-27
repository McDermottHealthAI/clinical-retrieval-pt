from typing import Any


class IdentityPooling:
    """Concrete pooling that returns fused_state unchanged."""

    def pool(self, fused_state: Any, attention_mask: Any | None = None) -> Any:
        """Return fused_state unchanged."""
        del attention_mask
        return fused_state
