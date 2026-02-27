"""RAP API model orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .encoders import PatientEncoder
    from .fusion import Fusion
    from .heads import PredictionHead
    from .pooling import Pooling
    from .query_projection import QueryProjector
    from .retrieval_encoder import RetrievalEncoder
    from .retrievers import Retriever

from .types import ModelOutput


class RetrievalAugmentedModel:
    """Composable pipeline orchestrator for RAP.

    This class wires the 4-stage pipeline only; stage behavior is delegated to injected components.

    Examples:
        >>> from clinical_retrieval_pt.encoders import IdentityEncoder
        >>> from clinical_retrieval_pt.fusion import ReplaceFusion
        >>> from clinical_retrieval_pt.heads import IdentityHead
        >>> from clinical_retrieval_pt.pooling import IdentityPooling
        >>> from clinical_retrieval_pt.query_projection import IdentityQueryProjector
        >>> from clinical_retrieval_pt.retrieval_encoder import IdentityRetrievalEncoder
        >>> from clinical_retrieval_pt.retrievers import StaticRetriever
        >>> model = RetrievalAugmentedModel(
        ...     encoder=IdentityEncoder(),
        ...     query_projector=IdentityQueryProjector(),
        ...     retriever=StaticRetriever(doc_tokens=[[1.0, 2.0]], doc_attention_mask=[[1, 1]]),
        ...     retrieval_encoder=IdentityRetrievalEncoder(),
        ...     fusion=ReplaceFusion(),
        ...     pooling=IdentityPooling(),
        ...     head=IdentityHead(),
        ... )
        >>> out = model.forward(batch={"not_used_yet": True})
        >>> out.logits
        [[1.0, 2.0]]
        >>> sorted(out.metadata)
        ['fusion_output', 'query_output', 'retrieval_encoder_output', 'retriever_output']
    """

    def __init__(
        self,
        *,
        encoder: PatientEncoder,
        query_projector: QueryProjector,
        retriever: Retriever,
        retrieval_encoder: RetrievalEncoder,
        fusion: Fusion,
        pooling: Pooling,
        head: PredictionHead,
    ) -> None:
        self.encoder = encoder
        self.query_projector = query_projector
        self.retriever = retriever
        self.retrieval_encoder = retrieval_encoder
        self.fusion = fusion
        self.pooling = pooling
        self.head = head

    def forward(self, batch: Any) -> ModelOutput:
        """Run the end-to-end RAP pipeline."""
        encoder_out = self.encoder.encode(batch)
        query_out = self.query_projector.project(
            encoder_out.patient_state,
            attention_mask=encoder_out.attention_mask,
        )
        retrieval_out = self.retriever.retrieve(query_out.query_embeddings)
        retrieval_encoded = self.retrieval_encoder.encode(retrieval_out)
        fusion_out = self.fusion.fuse(
            patient_state=encoder_out.patient_state,
            retrieval_memory=retrieval_encoded.retrieval_memory,
            retrieval_step_ids=query_out.retrieval_step_ids,
            doc_attention_mask=retrieval_out.doc_attention_mask,
        )
        pooled = self.pooling.pool(fusion_out.fused_state, attention_mask=encoder_out.attention_mask)
        logits = self.head.predict(pooled)

        return ModelOutput(
            logits=logits,
            metadata={
                "query_output": query_out,
                "retriever_output": retrieval_out,
                "retrieval_encoder_output": retrieval_encoded,
                "fusion_output": fusion_out,
            },
        )
