"""Core dense top-k search primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class TopKSearchBackend(ABC):
    """Top-k retrieval interface with optional per-query candidate restriction."""

    @abstractmethod
    def search(
        self,
        queries: Tensor,
        *,
        top_k: int,
        candidate_doc_ids: list[list[int]] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(doc_ids, scores)`` with shape ``(B, K)`` each."""


class DotProductTopKSearch(TopKSearchBackend):
    """Dense dot-product top-k search over fixed key vectors."""

    def __init__(self, keys: Tensor, *, doc_ids: Tensor | None = None) -> None:
        if keys.ndim != 2:
            raise ValueError(f"Expected keys shape (N, D), got {tuple(keys.shape)}")
        self.keys = keys.float()
        if doc_ids is None:
            self.doc_ids = torch.arange(self.keys.shape[0], dtype=torch.long)
        else:
            if doc_ids.ndim != 1:
                raise ValueError(f"Expected doc_ids shape (N,), got {tuple(doc_ids.shape)}")
            if doc_ids.shape[0] != self.keys.shape[0]:
                raise ValueError(
                    f"doc_ids length must match keys rows. got {doc_ids.shape[0]} vs {self.keys.shape[0]}"
                )
            self.doc_ids = doc_ids.long().clone()
        self._doc_id_to_pos = {int(doc_id.item()): idx for idx, doc_id in enumerate(self.doc_ids)}

    def score_all(self, queries: Tensor) -> Tensor:
        """Return dense score matrix with shape ``(B, N_docs)``."""
        if queries.ndim != 2:
            raise ValueError(f"Expected queries shape (B, D), got {tuple(queries.shape)}")
        return queries.float() @ self.keys.T

    def _candidate_positions_tensor(
        self,
        candidate_doc_ids: list[int],
        *,
        device: torch.device,
    ) -> Tensor:
        if not candidate_doc_ids:
            raise ValueError("Each candidate set must contain at least one document id")
        try:
            positions = [self._doc_id_to_pos[int(doc_id)] for doc_id in candidate_doc_ids]
        except KeyError as exc:
            raise ValueError(f"Unknown candidate doc id: {exc.args[0]}") from exc
        return torch.tensor(positions, dtype=torch.long, device=device)

    def search(
        self,
        queries: Tensor,
        *,
        top_k: int,
        candidate_doc_ids: list[list[int]] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if queries.ndim != 2:
            raise ValueError(f"Expected queries shape (B, D), got {tuple(queries.shape)}")

        query_matrix = queries.float()
        if candidate_doc_ids is None:
            scores = self.score_all(query_matrix)
            top_scores, top_idx = torch.topk(scores, k=top_k, dim=-1)
            return self.doc_ids.to(top_idx.device)[top_idx], top_scores

        if len(candidate_doc_ids) != query_matrix.shape[0]:
            raise ValueError(
                "candidate_doc_ids length must equal batch size. "
                f"got {len(candidate_doc_ids)} vs {query_matrix.shape[0]}"
            )

        out_ids: list[Tensor] = []
        out_scores: list[Tensor] = []
        doc_ids_device = self.doc_ids.to(query_matrix.device)
        for row_idx, candidates in enumerate(candidate_doc_ids):
            candidate_pos = self._candidate_positions_tensor(candidates, device=query_matrix.device)
            candidate_keys = self.keys[candidate_pos]
            row_scores = query_matrix[row_idx : row_idx + 1] @ candidate_keys.T
            take_k = min(top_k, candidate_pos.numel())
            top_scores, local_idx = torch.topk(row_scores, k=take_k, dim=-1)
            selected_pos = candidate_pos[local_idx.squeeze(0)]
            selected_doc_ids = doc_ids_device[selected_pos]

            if take_k < top_k:
                pad_count = top_k - take_k
                selected_doc_ids = torch.cat(
                    [selected_doc_ids, selected_doc_ids[-1].repeat(pad_count)], dim=0
                )
                top_scores = torch.cat([top_scores.squeeze(0), top_scores[0, -1].repeat(pad_count)], dim=0)
            else:
                top_scores = top_scores.squeeze(0)

            out_ids.append(selected_doc_ids)
            out_scores.append(top_scores)

        return torch.stack(out_ids, dim=0), torch.stack(out_scores, dim=0)
