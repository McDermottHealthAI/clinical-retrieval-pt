"""Runtime helpers for smoke-running the RAP scaffold."""

import torch
from meds_torchdata import MEDSTorchBatch

from .configs import instantiate_model


def build_model_from_cfg(cfg: object):
    """Build a ``RetrievalAugmentedModel`` from Hydra-style component config."""
    return instantiate_model(cfg)


def build_example_batch() -> MEDSTorchBatch:
    """Return a tiny valid MEDSTorchBatch for smoke runs."""
    return MEDSTorchBatch(
        code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
