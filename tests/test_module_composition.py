import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn

from clinical_retrieval_pt.configs import default_pipeline_config, instantiate_model
from clinical_retrieval_pt.encoders import MEDSCodeEncoder
from clinical_retrieval_pt.fusion import ReplaceFusion
from clinical_retrieval_pt.heads import IdentityHead
from clinical_retrieval_pt.model import RetrievalAugmentedModel
from clinical_retrieval_pt.pooling import IdentityPooling
from clinical_retrieval_pt.query_projection import IdentityQueryProjector
from clinical_retrieval_pt.retrieval_encoder import IdentityRetrievalEncoder
from clinical_retrieval_pt.retrievers import StaticRetriever
from clinical_retrieval_pt.types import RetrieverOutput


class TrainableHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def predict(self, pooled_state: torch.Tensor) -> torch.Tensor:
        return self.linear(pooled_state)

    def forward(self, pooled_state: torch.Tensor) -> torch.Tensor:
        return self.predict(pooled_state)


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[11, 22, 0], [7, 3, 1]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_default_pipeline_builds_nn_module_stages() -> None:
    model = instantiate_model(default_pipeline_config())

    assert isinstance(model, nn.Module)
    assert isinstance(model.encoder, nn.Module)
    assert isinstance(model.query_projector, nn.Module)
    assert isinstance(model.retrieval_encoder, nn.Module)
    assert isinstance(model.fusion, nn.Module)
    assert isinstance(model.pooling, nn.Module)
    assert isinstance(model.head, nn.Module)


def test_trainable_stage_parameters_are_registered_on_model() -> None:
    model = RetrievalAugmentedModel(
        encoder=MEDSCodeEncoder(),
        query_projector=IdentityQueryProjector(),
        retriever=StaticRetriever(
            doc_tokens=torch.FloatTensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]),
            doc_attention_mask=torch.LongTensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
        ),
        retrieval_encoder=IdentityRetrievalEncoder(),
        fusion=ReplaceFusion(),
        pooling=IdentityPooling(),
        head=TrainableHead(),
    )

    out = model(_example_batch())

    assert isinstance(out.logits, torch.Tensor)
    assert out.logits.shape == (2, 2)
    assert "head.linear.weight" in model.state_dict()


def test_stage_forward_aliases_named_methods() -> None:
    batch = _example_batch()
    encoder = MEDSCodeEncoder()
    enc_via_method = encoder.encode(batch)
    enc_via_forward = encoder(batch)
    assert torch.equal(enc_via_method.patient_state, enc_via_forward.patient_state)

    projector = IdentityQueryProjector()
    q_via_method = projector.project(enc_via_method.patient_state)
    q_via_forward = projector(enc_via_method.patient_state)
    assert torch.equal(q_via_method.query_embeddings, q_via_forward.query_embeddings)

    retrieval_out = RetrieverOutput(
        doc_tokens=torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]]),
        doc_attention_mask=torch.LongTensor([[1, 1], [1, 1]]),
    )
    retrieval_encoder = IdentityRetrievalEncoder()
    r_via_method = retrieval_encoder.encode(retrieval_out)
    r_via_forward = retrieval_encoder(retrieval_out)
    assert torch.equal(r_via_method.retrieval_memory, r_via_forward.retrieval_memory)

    fusion = ReplaceFusion()
    f_via_method = fusion.fuse(
        patient_state=enc_via_method.patient_state,
        retrieval_memory=r_via_method.retrieval_memory,
    )
    f_via_forward = fusion(
        patient_state=enc_via_method.patient_state,
        retrieval_memory=r_via_method.retrieval_memory,
    )
    assert torch.equal(f_via_method.fused_state, f_via_forward.fused_state)

    pooling = IdentityPooling()
    p_via_method = pooling.pool(f_via_method.fused_state)
    p_via_forward = pooling(f_via_method.fused_state)
    assert torch.equal(p_via_method, p_via_forward)

    head = IdentityHead()
    h_via_method = head.predict(p_via_method)
    h_via_forward = head(p_via_method)
    assert torch.equal(h_via_method, h_via_forward)


def test_static_retriever_coerces_list_inputs_to_expected_tensor_dtypes() -> None:
    retriever = StaticRetriever(
        doc_tokens=[[1, 2, 3]],
        doc_attention_mask=[[1, 0, 1]],
        doc_scores=[[0.2, 0.8]],
        doc_ids=[[10, 11]],
    )

    out = retriever.retrieve(torch.zeros((1, 3)))

    assert out.doc_tokens.dtype == torch.long
    assert out.doc_attention_mask.dtype == torch.bool
    assert out.doc_scores is not None and out.doc_scores.dtype == torch.float32
    assert out.doc_ids is not None and out.doc_ids.dtype == torch.long
