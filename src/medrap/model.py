"""RAP API model orchestration."""

from torch import nn

from .types import ModelOutput


class RetrievalAugmentedModel(nn.Module):
    """Composable pipeline orchestrator for RAP.

    This class wires the 4-stage pipeline only; stage behavior is delegated to injected components.

    Examples:
        >>> import torch
        >>> from medrap.encoders import MEDSCodeEncoder
        >>> from medrap.fusion import ReplaceFusion
        >>> from medrap.heads import IdentityHead
        >>> from meds_torchdata import MEDSTorchBatch
        >>> from medrap.pooling import IdentityPooling
        >>> from medrap.query_projection import IdentityQueryProjector
        >>> from medrap.retrieval_encoder import IdentityRetrievalEncoder
        >>> from medrap.retrievers import StaticRetriever
        >>> model = RetrievalAugmentedModel(
        ...     encoder=MEDSCodeEncoder(),
        ...     query_projector=IdentityQueryProjector(),
        ...     retriever=StaticRetriever(doc_tokens=[[1, 2]], doc_attention_mask=[[1, 1]]),
        ...     retrieval_encoder=IdentityRetrievalEncoder(),
        ...     fusion=ReplaceFusion(),
        ...     pooling=IdentityPooling(),
        ...     head=IdentityHead(),
        ... )
        >>> batch = MEDSTorchBatch(
        ...     code=torch.LongTensor([[101, 0], [42, 7]]),
        ...     numeric_value=torch.zeros((2, 2), dtype=torch.float32),
        ...     numeric_value_mask=torch.zeros((2, 2), dtype=torch.bool),
        ...     time_delta_days=torch.zeros((2, 2), dtype=torch.float32),
        ... )
        >>> out = model.forward(batch=batch)
        >>> out.logits
        tensor([[1, 2]])
        >>> sorted(out.metadata)
        ['fusion_output', 'query_output', 'retrieval_encoder_output', 'retriever_output']
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        query_projector: nn.Module,
        retriever: nn.Module,
        retrieval_encoder: nn.Module,
        fusion: nn.Module,
        pooling: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.query_projector = query_projector
        self.retriever = retriever
        self.retrieval_encoder = retrieval_encoder
        self.fusion = fusion
        self.pooling = pooling
        self.head = head

    def forward(self, batch: object) -> ModelOutput:
        """Run the end-to-end RAP pipeline."""
        encoder_out = self.encoder(batch)
        query_out = self.query_projector(encoder_out.patient_state)
        retrieval_out = self.retriever(query_out.query_embeddings)
        retrieval_encoded = self.retrieval_encoder(retrieval_out)
        fusion_out = self.fusion(
            patient_state=encoder_out.patient_state,
            retrieval_memory=retrieval_encoded.retrieval_memory,
            retrieval_step_ids=query_out.retrieval_step_ids,
            doc_attention_mask=retrieval_out.doc_attention_mask,
        )
        pooled = self.pooling(fusion_out.fused_state)
        logits = self.head(pooled)

        return ModelOutput(
            logits=logits,
            metadata={
                "query_output": query_out,
                "retriever_output": retrieval_out,
                "retrieval_encoder_output": retrieval_encoded,
                "fusion_output": fusion_out,
            },
        )
