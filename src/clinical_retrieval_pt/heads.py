from __future__ import annotations

from typing import Any


class PredictionHead:
    """Base task head interface for RAP API v2."""

    def predict(self, pooled_state: Any) -> Any:
        """Predict logits from pooled state.

        Raises:
            NotImplementedError: Always for base class.
        """
        raise NotImplementedError("PredictionHead.predict is not implemented for base class.")
