import importlib.util

import pytest
import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn

from medrap.batch_adapter import MEDSSupervisedBatch
from medrap.lightning_module import MedRAPLightningModule
from medrap.task import RegressionTask
from medrap.types import ModelOutput

HAS_LIGHTNING = bool(importlib.util.find_spec("lightning")) or bool(
    importlib.util.find_spec("pytorch_lightning")
)


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
        logits = self.linear(batch.code.float())
        return ModelOutput(logits=logits)


def test_training_signal_improves_on_toy_regression() -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    module = MedRAPLightningModule(
        model=TinyRegressor(),
        task=RegressionTask(label_field="float_value"),
    )
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    code = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    target = torch.tensor([0.2, 0.4, 0.6, 1.2], dtype=torch.float32)

    batch = MEDSSupervisedBatch(
        batch=MEDSTorchBatch(
            code=code.long(),
            numeric_value=torch.zeros_like(code),
            numeric_value_mask=torch.zeros_like(code, dtype=torch.bool),
            time_delta_days=torch.zeros_like(code),
        ),
        float_value=target,
    )

    with torch.no_grad():
        initial_loss = float(module.training_step(batch, batch_idx=0))

    for _ in range(80):
        optimizer.zero_grad()
        loss = module.training_step(batch, batch_idx=0)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = float(module.training_step(batch, batch_idx=0))

    assert final_loss < initial_loss
