import torch
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from meds_torchdata import MEDSTorchBatch

from clinical_retrieval_pt.configs import RAPAppConfig
from clinical_retrieval_pt.model import RetrievalAugmentedModel
from clinical_retrieval_pt.runtime import build_model_from_cfg


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_train_config_composes_and_instantiates_model() -> None:
    with initialize_config_module(version_base=None, config_module="clinical_retrieval_pt.conf"):
        cfg = compose(config_name="_train")

    model = build_model_from_cfg(cfg)

    assert isinstance(model, RetrievalAugmentedModel)
    out = model.forward(_example_batch())
    assert out.logits == [[1.0, 2.0]]


def test_app_config_registers_with_hydra_config_store() -> None:
    RAPAppConfig.add_to_config_store(group="medrap")
    cs = ConfigStore.instance()

    assert "medrap" in cs.repo
    assert "RAPAppConfig.yaml" in cs.repo["medrap"]
