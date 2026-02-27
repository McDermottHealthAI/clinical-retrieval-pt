from __future__ import annotations

from typing import Any


class IdentityHead:
    """Concrete head that returns pooled state unchanged."""

    def predict(self, pooled_state: Any) -> Any:
        """Return pooled_state unchanged."""
        return pooled_state
