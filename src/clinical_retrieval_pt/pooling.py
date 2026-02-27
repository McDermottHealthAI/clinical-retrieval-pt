from __future__ import annotations

from typing import Any


class Pooling:
    """Base pooling interface for RAP API v2."""

    def pool(self, fused_state: Any, attention_mask: Any | None = None) -> Any:
        """Pool fused state to a fixed-size vector.

        Raises:
            NotImplementedError: Always for base class.
        """
        raise NotImplementedError("Pooling.pool is not implemented for base class.")


class IdentityPooling(Pooling):
    """Concrete pooling that returns fused_state unchanged."""

    def pool(self, fused_state: Any, attention_mask: Any | None = None) -> Any:
        """Return fused_state unchanged."""
        del attention_mask
        return fused_state
