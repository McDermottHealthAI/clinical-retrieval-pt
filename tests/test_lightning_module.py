import importlib.util
from functools import partial

import pytest
import torch
from torch import nn

from medrap.lightning_module import MedRAPLightningModule
from medrap.training_metrics import ClassificationMetrics
from medrap.types import ModelOutput

HAS_LIGHTNING = bool(importlib.util.find_spec("lightning")) or bool(
    importlib.util.find_spec("pytorch_lightning")
)


class DummyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, batch: object) -> ModelOutput:
        x = torch.as_tensor(batch, dtype=torch.float32)
        return ModelOutput(logits=self.linear(x))


def test_lightning_module_training_step_and_optimizer() -> None:
    if not HAS_LIGHTNING:
        with pytest.raises(ModuleNotFoundError, match="lightning"):
            MedRAPLightningModule(model=DummyClassifier())
        return

    module = MedRAPLightningModule(
        model=DummyClassifier(),
        metrics=ClassificationMetrics(num_classes=2),
        optimizer=partial(torch.optim.AdamW, lr=1e-2, weight_decay=1e-3),
    )

    batch = (torch.randn(4, 3), torch.LongTensor([0, 1, 0, 1]))
    loss = module.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    configured = module.configure_optimizers()
    assert isinstance(configured, torch.optim.Optimizer)
    optimizer = configured
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.defaults["lr"] == pytest.approx(1e-2)
    assert optimizer.defaults["weight_decay"] == pytest.approx(1e-3)


def test_lightning_module_accepts_mapping_batches() -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    module = MedRAPLightningModule(model=DummyClassifier(), metrics=ClassificationMetrics(num_classes=2))

    batch = {
        "batch": torch.randn(3, 3),
        "target": torch.LongTensor([0, 1, 0]),
    }

    val_loss = module.validation_step(batch, batch_idx=0)
    assert isinstance(val_loss, torch.Tensor)
    assert val_loss.ndim == 0


def test_lightning_module_configures_scheduler_when_provided() -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    module = MedRAPLightningModule(
        model=DummyClassifier(),
        metrics=ClassificationMetrics(num_classes=2),
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-2),
        lr_scheduler=partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0, total_iters=1),
    )
    configured = module.configure_optimizers()

    assert isinstance(configured, dict)
    assert "optimizer" in configured
    assert "lr_scheduler" in configured


def test_lightning_module_treats_empty_scheduler_mapping_as_none() -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    module = MedRAPLightningModule(
        model=DummyClassifier(),
        metrics=ClassificationMetrics(num_classes=2),
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-2),
        lr_scheduler={},
    )
    configured = module.configure_optimizers()

    assert isinstance(configured, torch.optim.Optimizer)
