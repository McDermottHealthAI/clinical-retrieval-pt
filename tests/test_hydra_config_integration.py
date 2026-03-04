import importlib.util

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from meds_torchdata import MEDSTorchBatch

from medrap.configs import RAPAppConfig
from medrap.lightning_datamodule import DemoMedRAPDataModule
from medrap.lightning_module import MedRAPLightningModule
from medrap.model import RetrievalAugmentedModel
from medrap.runtime import build_model_from_cfg

HAS_LIGHTNING = bool(importlib.util.find_spec("lightning")) or bool(
    importlib.util.find_spec("pytorch_lightning")
)


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_train_config_composes_and_instantiates_model() -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(config_name="_train")

    model = build_model_from_cfg(cfg)

    assert isinstance(model, RetrievalAugmentedModel)
    out = model.forward(_example_batch())
    assert isinstance(out.logits, torch.Tensor)
    assert out.logits.shape[1] == 2


def test_train_config_instantiates_lightning_stack() -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(config_name="_train")

    datamodule = instantiate(cfg.datamodule)
    lightning_module = instantiate(cfg.lightning_module, model=build_model_from_cfg(cfg))

    assert isinstance(datamodule, DemoMedRAPDataModule)
    assert isinstance(lightning_module, MedRAPLightningModule)


def test_app_config_registers_with_hydra_config_store() -> None:
    RAPAppConfig.add_to_config_store(group="medrap")
    cs = ConfigStore.instance()

    assert "medrap" in cs.repo
    assert "RAPAppConfig.yaml" in cs.repo["medrap"]


def test_train_config_supports_meds_torchdata_datamodule_override(tmp_path) -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(
            config_name="_train",
            overrides=[
                "datamodule=meds_torchdata",
                f"datamodule.config.tensorized_cohort_dir={tmp_path}",
            ],
        )

    datamodule = instantiate(cfg.datamodule)
    assert datamodule.__class__.__name__ == "Datamodule"
    assert str(datamodule.config.tensorized_cohort_dir) == str(tmp_path)
