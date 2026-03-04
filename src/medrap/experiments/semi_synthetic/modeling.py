"""Model components for the MIMIC demo semi-synthetic experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .backend import DotProductTopKSearch
from .components import BinaryLogitHead, LinearProjection, MLPEncoder


@dataclass(slots=True)
class SemiSyntheticForwardOutput:
    """Forward outputs including retrieval diagnostics."""

    logits: Tensor
    top_doc_ids: Tensor | None = None
    top_doc_scores: Tensor | None = None
    top_doc_labels: Tensor | None = None
    all_doc_weights: Tensor | None = None


class SemiSyntheticRetrievalModel(nn.Module):
    """Baseline model family for semi-synthetic retrieval experiments."""

    def __init__(
        self,
        *,
        mode: str,
        patient_dim: int,
        doc_features: Tensor,
        doc_labels: Tensor,
        hidden_dim: int = 128,
        encoder_depth: int = 2,
        top_k: int = 1,
        random_seed: int = 17,
    ) -> None:
        super().__init__()
        if mode not in {"learned", "oracle", "oracle_positive", "random", "no_retrieval"}:
            raise ValueError(f"Unsupported mode={mode!r}")

        self.mode = mode
        self.top_k = int(top_k)
        self.register_buffer("doc_features", doc_features.float())
        self.register_buffer("doc_labels", doc_labels.long())

        doc_dim = int(self.doc_features.shape[1])
        self.patient_encoder = MLPEncoder(
            in_dim=patient_dim,
            hidden_dim=hidden_dim,
            depth=encoder_depth,
        )
        self.query_projector = LinearProjection(in_dim=hidden_dim, out_dim=doc_dim)
        self.retrieval_head = BinaryLogitHead(in_dim=doc_dim, hidden_dim=hidden_dim)
        self.no_retrieval_head = BinaryLogitHead(in_dim=patient_dim, hidden_dim=hidden_dim)

        self.backend = DotProductTopKSearch(self.doc_features)
        self._rng = torch.Generator().manual_seed(random_seed)

    def _pool_retrieved_features(self, top_doc_ids: Tensor, top_doc_scores: Tensor) -> Tensor:
        gathered = self.doc_features[top_doc_ids]  # (B, K, D)
        weights = torch.softmax(top_doc_scores, dim=-1).unsqueeze(-1)
        return (gathered * weights).sum(dim=1)

    def _oracle_doc_ids(self, true_doc_ids: list[list[int]], device: torch.device) -> Tensor:
        chosen: list[int] = []
        for docs in true_doc_ids:
            if not docs:
                chosen.append(0)
            else:
                chosen.append(int(docs[0]))
        return torch.tensor(chosen, dtype=torch.long, device=device)

    def _oracle_positive_doc_ids(
        self,
        true_doc_ids: list[list[int]],
        true_positive_doc_ids: list[list[int]],
        device: torch.device,
    ) -> Tensor:
        chosen: list[int] = []
        for docs, positive_docs in zip(true_doc_ids, true_positive_doc_ids, strict=True):
            if positive_docs:
                chosen.append(int(positive_docs[0]))
            elif docs:
                chosen.append(int(docs[0]))
            else:
                chosen.append(0)
        return torch.tensor(chosen, dtype=torch.long, device=device)

    def _random_doc_ids(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(
            low=0,
            high=self.doc_features.shape[0],
            size=(batch_size,),
            generator=self._rng,
            dtype=torch.long,
            device=device,
        )

    def forward(
        self,
        batch: dict[str, Any] | None = None,
        *,
        patient_features: Tensor | None = None,
        true_doc_ids: list[list[int]] | None = None,
        true_positive_doc_ids: list[list[int]] | None = None,
    ) -> SemiSyntheticForwardOutput:
        if batch is not None:
            if not isinstance(batch, dict):
                raise TypeError(f"Expected dict batch input, got {type(batch)!r}")
            if patient_features is None:
                raw_features = batch.get("patient_features")
                if not isinstance(raw_features, Tensor):
                    raise ValueError("Semi-synthetic batch must contain Tensor key 'patient_features'.")
                patient_features = raw_features
            if true_doc_ids is None:
                raw_doc_ids = batch.get("true_doc_ids")
                if not isinstance(raw_doc_ids, list):
                    raise ValueError("Semi-synthetic batch must contain key 'true_doc_ids'.")
                true_doc_ids = [[int(item) for item in row] for row in raw_doc_ids]
            if true_positive_doc_ids is None:
                raw_positive_doc_ids = batch.get("true_positive_doc_ids")
                if isinstance(raw_positive_doc_ids, list):
                    true_positive_doc_ids = [[int(item) for item in row] for row in raw_positive_doc_ids]

        if patient_features is None:
            raise ValueError("patient_features is required.")

        if self.mode == "no_retrieval":
            logits = self.no_retrieval_head(patient_features)
            return SemiSyntheticForwardOutput(logits=logits)

        if true_doc_ids is None:
            raise ValueError("true_doc_ids is required for retrieval baselines.")

        batch_size = int(patient_features.shape[0])
        device = patient_features.device

        if self.mode == "learned":
            patient_embedding = self.patient_encoder(patient_features)
            query = self.query_projector(patient_embedding)
            top_doc_ids, top_doc_scores = self.backend.search(query, top_k=self.top_k)
            top_doc_ids = top_doc_ids.to(device)
            top_doc_scores = top_doc_scores.to(device)
            full_scores = self.backend.score_all(query).to(device)
            doc_weights = torch.softmax(full_scores, dim=-1)
            pooled_doc_features = doc_weights @ self.doc_features
        elif self.mode == "oracle":
            top_doc_ids = self._oracle_doc_ids(true_doc_ids, device=device).unsqueeze(-1)
            top_doc_scores = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
            pooled_doc_features = self._pool_retrieved_features(top_doc_ids, top_doc_scores)
            doc_weights = None
        elif self.mode == "oracle_positive":
            if true_positive_doc_ids is None:
                raise ValueError("true_positive_doc_ids is required for mode='oracle_positive'")
            top_doc_ids = self._oracle_positive_doc_ids(
                true_doc_ids,
                true_positive_doc_ids,
                device=device,
            ).unsqueeze(-1)
            top_doc_scores = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
            pooled_doc_features = self._pool_retrieved_features(top_doc_ids, top_doc_scores)
            doc_weights = None
        elif self.mode == "random":
            top_doc_ids = self._random_doc_ids(batch_size, device=device).unsqueeze(-1)
            top_doc_scores = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
            pooled_doc_features = self._pool_retrieved_features(top_doc_ids, top_doc_scores)
            doc_weights = None
        else:
            raise RuntimeError(f"Unexpected mode={self.mode!r}")

        logits = self.retrieval_head(pooled_doc_features)
        doc_labels = self.doc_labels[top_doc_ids[:, 0]]

        return SemiSyntheticForwardOutput(
            logits=logits,
            top_doc_ids=top_doc_ids[:, 0],
            top_doc_scores=top_doc_scores[:, 0],
            top_doc_labels=doc_labels,
            all_doc_weights=doc_weights if self.mode == "learned" else None,
        )
