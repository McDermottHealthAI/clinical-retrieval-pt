import importlib.util
from functools import partial

import pytest
import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn

from medrap.batch_adapter import MEDSSupervisedBatch, MEDSSupervisedBatchAdapter
from medrap.lightning_module import MedRAPLightningModule
from medrap.task import BinaryClassificationTask, CategoricalClassificationTask, RegressionTask
from medrap.types import ModelOutput

HAS_LIGHTNING = bool(importlib.util.find_spec("lightning")) or bool(
    importlib.util.find_spec("pytorch_lightning")
)


class DummyPredictor(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3, out_features)

    def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
        x = batch.code.float()
        return ModelOutput(logits=self.linear(x))


def _base_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]),
        numeric_value=torch.zeros((3, 3), dtype=torch.float32),
        numeric_value_mask=torch.zeros((3, 3), dtype=torch.bool),
        time_delta_days=torch.zeros((3, 3), dtype=torch.float32),
    )


def test_lightning_binary_task_with_labeled_meds_batch() -> None:
    if not HAS_LIGHTNING:
        with pytest.raises(ModuleNotFoundError, match="lightning"):
            MedRAPLightningModule(model=DummyPredictor(out_features=1))
        return

    module = MedRAPLightningModule(
        model=DummyPredictor(out_features=1),
        task=BinaryClassificationTask(label_field="boolean_value"),
        batch_adapter=MEDSSupervisedBatchAdapter(label_field="boolean_value"),
        optimizer=partial(torch.optim.AdamW, lr=1e-2, weight_decay=1e-3),
    )

    batch = MEDSTorchBatch(
        code=_base_batch().code,
        numeric_value=_base_batch().numeric_value,
        numeric_value_mask=_base_batch().numeric_value_mask,
        time_delta_days=_base_batch().time_delta_days,
        boolean_value=torch.BoolTensor([True, False, True]),
    )

    loss = module.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_lightning_categorical_task_with_supervised_wrapper_batch() -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    module = MedRAPLightningModule(
        model=DummyPredictor(out_features=3),
        task=CategoricalClassificationTask(num_classes=3, label_field="categorical_value"),
        batch_adapter=MEDSSupervisedBatchAdapter(label_field="categorical_value"),
    )

    wrapped = MEDSSupervisedBatch(
        batch=_base_batch(),
        categorical_value=torch.LongTensor([0, 1, 2]),
    )
    val_loss = module.validation_step(wrapped, batch_idx=0)
    assert isinstance(val_loss, torch.Tensor)
    assert val_loss.ndim == 0


def test_lightning_regression_task_does_not_require_num_classes() -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    module = MedRAPLightningModule(
        model=DummyPredictor(out_features=1),
        task=RegressionTask(label_field="float_value"),
        batch_adapter=MEDSSupervisedBatchAdapter(label_field="float_value"),
    )

    wrapped = MEDSSupervisedBatch(
        batch=_base_batch(),
        float_value=torch.FloatTensor([0.1, 0.9, 0.4]),
    )
    test_loss = module.test_step(wrapped, batch_idx=0)
    assert isinstance(test_loss, torch.Tensor)
    assert test_loss.ndim == 0


def test_batch_adapter_rejects_ambiguous_multiple_label_fields() -> None:
    adapter = MEDSSupervisedBatchAdapter()
    wrapped = MEDSSupervisedBatch(
        batch=_base_batch(),
        float_value=torch.FloatTensor([0.1, 0.9, 0.4]),
        categorical_value=torch.LongTensor([0, 1, 2]),
    )

    with pytest.raises(ValueError, match="Multiple label tensors"):
        adapter(wrapped)


def test_lightning_module_configures_scheduler_when_provided() -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    module = MedRAPLightningModule(
        model=DummyPredictor(out_features=1),
        task=BinaryClassificationTask(label_field="boolean_value"),
        batch_adapter=MEDSSupervisedBatchAdapter(label_field="boolean_value"),
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
        model=DummyPredictor(out_features=1),
        task=BinaryClassificationTask(label_field="boolean_value"),
        batch_adapter=MEDSSupervisedBatchAdapter(label_field="boolean_value"),
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-2),
        lr_scheduler={},
    )
    configured = module.configure_optimizers()

    assert isinstance(configured, torch.optim.Optimizer)
