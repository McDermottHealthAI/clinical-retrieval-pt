"""Shared API contract types.

This module defines the stage-by-stage outputs used by the RAP pipeline.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EncoderOutput:
    """Output of encode.

    Attributes:
        patient_state: Encoded patient representation with shape ``(B, *P)``.
    """

    patient_state: Any


@dataclass(slots=True)
class QueryOutput:
    """Output of query projection.

    Attributes:
        query_embeddings: Query tensor with shape ``(B, R, D_ret)``.
        retrieval_step_ids: Optional retrieval step mapping ``(B, S_ehr)``.
            Present in sequence mode (Config A), absent in tabular modes (B/C).
    """

    query_embeddings: Any
    retrieval_step_ids: Any | None = None


@dataclass(slots=True)
class RetrieverOutput:
    """Output of retrieval.

    Attributes:
        doc_tokens: Retrieved token ids with shape ``(B, R, K, S_doc)``.
        doc_attention_mask: Retrieved token mask with shape ``(B, R, K, S_doc)``.
        doc_scores: Optional retrieval scores with shape ``(B, R, K)``.
        doc_ids: Optional retrieval identifiers with shape ``(B, R, K)``.
        doc_key_embeddings: Optional retrieved key embeddings with shape
            ``(B, R, K, D_ret)`` for in-graph query-key scoring losses.
    """

    doc_tokens: Any
    doc_attention_mask: Any
    doc_scores: Any | None = None
    doc_ids: Any | None = None
    doc_key_embeddings: Any | None = None


@dataclass(slots=True)
class RetrievalEncoderOutput:
    """Output of retrieval encoding.

    Attributes:
        retrieval_memory: Encoded retrieval memory with shape
            ``(B, R, K, S_doc, D_mem)``.
    """

    retrieval_memory: Any


@dataclass(slots=True)
class FusionOutput:
    """Output of fusion.

    Attributes:
        fused_state: Fused representation with shape ``(B, *F)``.
    """

    fused_state: Any


@dataclass(slots=True)
class ModelOutput:
    """Output of full model forward pass.

    Attributes:
        logits: Task logits with shape ``(B, C)``.
        metadata: Optional side outputs (for diagnostics and training utilities).
    """

    logits: Any
    metadata: dict[str, Any] = field(default_factory=dict)
