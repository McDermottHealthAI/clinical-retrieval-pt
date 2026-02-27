import torch
from meds_torchdata import MEDSTorchBatch

from clinical_retrieval_pt.configs import (
    PipelineConfig,
    StaticRetrieverConfig,
    default_pipeline_config,
    instantiate_model,
)
from clinical_retrieval_pt.encoders import MEDSCodeEncoder
from clinical_retrieval_pt.fusion import ReplaceFusion
from clinical_retrieval_pt.heads import IdentityHead
from clinical_retrieval_pt.model import RetrievalAugmentedModel
from clinical_retrieval_pt.pooling import IdentityPooling
from clinical_retrieval_pt.query_projection import IdentityQueryProjector
from clinical_retrieval_pt.retrieval_encoder import IdentityRetrievalEncoder
from clinical_retrieval_pt.retrievers import StaticRetriever


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_default_pipeline_config_instantiates_default_components() -> None:
    cfg = default_pipeline_config()

    model = instantiate_model(cfg)

    assert isinstance(model, RetrievalAugmentedModel)
    assert isinstance(model.encoder, MEDSCodeEncoder)
    assert isinstance(model.query_projector, IdentityQueryProjector)
    assert isinstance(model.retriever, StaticRetriever)
    assert isinstance(model.retrieval_encoder, IdentityRetrievalEncoder)
    assert isinstance(model.fusion, ReplaceFusion)
    assert isinstance(model.pooling, IdentityPooling)
    assert isinstance(model.head, IdentityHead)


def test_pipeline_config_allows_overriding_retriever_values() -> None:
    cfg = PipelineConfig(
        retriever=StaticRetrieverConfig(
            doc_tokens=[[9, 8]],
            doc_attention_mask=[[1, 1]],
        )
    )
    model = instantiate_model(cfg)

    out = model.forward(_example_batch())

    assert torch.equal(out.logits, torch.LongTensor([[9, 8]]))
