"""Runtime helpers for building RAP models from composed configs."""

from typing import Any

from .configs import instantiate_model


def build_model_from_cfg(cfg: Any):
    """Build a ``RetrievalAugmentedModel`` from Hydra-style component config."""
    return instantiate_model(cfg)
