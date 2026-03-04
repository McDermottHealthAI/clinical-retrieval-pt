"""Reusable retrieval-evaluation utilities for experiment callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def binary_accuracy_from_logits(logits: Tensor, targets: Tensor) -> float:
    """Compute binary accuracy from logits and binary targets."""
    if logits.numel() == 0:
        return 0.0
    preds = torch.sigmoid(logits) >= 0.5
    refs = targets >= 0.5
    return float((preds == refs).float().mean().item())


def top1_recall_at_1(top1_doc_ids: Tensor, target_doc_ids: Tensor) -> Tensor:
    """Compute recall@1 from top-1 retrieved doc ids and target doc ids."""
    if top1_doc_ids.ndim != 1:
        raise ValueError(f"Expected top1_doc_ids shape (B,), got {tuple(top1_doc_ids.shape)}")
    refs = target_doc_ids.long()
    if refs.ndim != 1:
        raise ValueError(f"Expected target_doc_ids shape (B,), got {tuple(refs.shape)}")
    if refs.shape[0] != top1_doc_ids.shape[0]:
        raise ValueError(
            "top1_doc_ids and target_doc_ids must have same batch size. "
            f"got {top1_doc_ids.shape[0]} vs {refs.shape[0]}"
        )
    return (top1_doc_ids.long() == refs).float().mean()


def retrieval_doc_ids_are_sample_dependent(doc_ids: Tensor) -> bool:
    """Return ``True`` when retrieval doc ids vary across rows."""
    if doc_ids.ndim < 2:
        raise ValueError(f"Expected retrieval doc_ids with shape (B, K), got {tuple(doc_ids.shape)}")
    if doc_ids.shape[0] < 2:
        return False
    first_row = doc_ids[0]
    return any(not torch.equal(row, first_row) for row in doc_ids[1:])


def compute_retrieval_batch_metrics(
    *,
    top1_doc_ids: Tensor,
    top1_doc_labels: Tensor,
    targets: Tensor,
    true_doc_ids: list[list[int]],
    true_positive_doc_ids: list[list[int]],
) -> dict[str, Tensor]:
    """Compute hard retrieval metrics that expose label-equivalence failure modes."""
    if top1_doc_ids.ndim != 1:
        raise ValueError(f"Expected top1_doc_ids shape (B,), got {tuple(top1_doc_ids.shape)}")

    batch_size = top1_doc_ids.shape[0]
    any_hits: list[float] = []
    positive_hits: list[float] = []
    positive_doc_label_hits: list[float] = []
    label_consistent_patient_inconsistent: list[float] = []

    for idx in range(batch_size):
        retrieved = int(top1_doc_ids[idx].item())
        retrieved_label = int(top1_doc_labels[idx].item())
        target = int(targets[idx].item())
        patient_docs = set(true_doc_ids[idx])
        positive_docs = set(true_positive_doc_ids[idx])

        any_hits.append(float(retrieved in patient_docs))
        positive_doc_label_hits.append(float(retrieved_label == 1))

        if target == 1 and positive_docs:
            positive_hits.append(float(retrieved in positive_docs))

        consistent = retrieved_label == target
        patient_inconsistent = retrieved not in patient_docs
        label_consistent_patient_inconsistent.append(float(consistent and patient_inconsistent))

    top1_hit_any = torch.tensor(any_hits, dtype=torch.float32).mean()
    top1_is_positive_doc_label = torch.tensor(positive_doc_label_hits, dtype=torch.float32).mean()
    if positive_hits:
        top1_hit_positive_patient_set = torch.tensor(positive_hits, dtype=torch.float32).mean()
    else:
        top1_hit_positive_patient_set = torch.tensor(0.0, dtype=torch.float32)

    lcpi_rate = torch.tensor(label_consistent_patient_inconsistent, dtype=torch.float32).mean()
    return {
        "retrieval/top1_hit_any_true_drug": top1_hit_any,
        "retrieval/top1_is_positive_doc_label": top1_is_positive_doc_label,
        "retrieval/top1_hit_any_positive_drug_in_patient_set": top1_hit_positive_patient_set,
        "retrieval/top1_hit_any_positive_drug": top1_hit_positive_patient_set,
        "retrieval/label_consistent_patient_inconsistent_rate": lcpi_rate,
    }


def empty_hard_retrieval_metrics() -> dict[str, float]:
    """Return a zero-valued hard retrieval metric payload."""
    return {
        "retrieval/top1_hit_any_true_drug": 0.0,
        "retrieval/top1_is_positive_doc_label": 0.0,
        "retrieval/top1_hit_any_positive_drug_in_patient_set": 0.0,
        "retrieval/top1_hit_any_positive_drug": 0.0,
        "retrieval/label_consistent_patient_inconsistent_rate": 0.0,
    }


@dataclass(slots=True)
class RetrievalEvalState:
    """Mutable state helper for per-epoch retrieval diagnostics."""

    logits: list[Tensor] = field(default_factory=list)
    targets: list[Tensor] = field(default_factory=list)
    top_doc_ids: list[Tensor] = field(default_factory=list)
    top_doc_labels: list[Tensor] = field(default_factory=list)
    true_doc_ids: list[list[int]] = field(default_factory=list)
    true_positive_doc_ids: list[list[int]] = field(default_factory=list)
    pred_stats: dict[str, float] = field(
        default_factory=lambda: {
            "count": 0.0,
            "target_positive_sum": 0.0,
            "pred_positive_sum": 0.0,
        }
    )
    soft_mass: dict[str, float] = field(
        default_factory=lambda: {
            "soft_mass_any_patient_sum": 0.0,
            "soft_mass_any_patient_count": 0.0,
            "soft_mass_positive_patient_sum": 0.0,
            "soft_mass_positive_patient_count": 0.0,
        }
    )

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.logits.clear()
        self.targets.clear()
        self.top_doc_ids.clear()
        self.top_doc_labels.clear()
        self.true_doc_ids.clear()
        self.true_positive_doc_ids.clear()
        self.pred_stats = {
            "count": 0.0,
            "target_positive_sum": 0.0,
            "pred_positive_sum": 0.0,
        }
        self.soft_mass = {
            "soft_mass_any_patient_sum": 0.0,
            "soft_mass_any_patient_count": 0.0,
            "soft_mass_positive_patient_sum": 0.0,
            "soft_mass_positive_patient_count": 0.0,
        }

    def update_predictions(self, *, logits: Tensor, targets: Tensor) -> None:
        """Record predictions/targets and update thresholded prediction stats."""
        probs = torch.sigmoid(logits)
        self.pred_stats["count"] += float(targets.numel())
        self.pred_stats["target_positive_sum"] += float(targets.float().sum().item())
        self.pred_stats["pred_positive_sum"] += float((probs >= 0.5).float().sum().item())
        self.logits.append(logits.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def update_retrieval(
        self,
        *,
        top_doc_ids: Tensor,
        top_doc_labels: Tensor,
        true_doc_ids: list[list[int]],
        true_positive_doc_ids: list[list[int]],
        all_doc_weights: Tensor | None = None,
    ) -> None:
        """Record retrieval outputs and optional soft retrieval mass diagnostics."""
        self.top_doc_ids.append(top_doc_ids.detach().cpu())
        self.top_doc_labels.append(top_doc_labels.detach().cpu())
        self.true_doc_ids.extend([[int(item) for item in row] for row in true_doc_ids])
        self.true_positive_doc_ids.extend([[int(item) for item in row] for row in true_positive_doc_ids])

        if all_doc_weights is None:
            return

        for row_idx in range(all_doc_weights.shape[0]):
            row_weights = all_doc_weights[row_idx]
            all_docs = {int(v) for v in true_doc_ids[row_idx]}
            positive_docs = {int(v) for v in true_positive_doc_ids[row_idx]}
            if all_docs:
                self.soft_mass["soft_mass_any_patient_sum"] += float(row_weights[list(all_docs)].sum().item())
                self.soft_mass["soft_mass_any_patient_count"] += 1.0
            if positive_docs:
                self.soft_mass["soft_mass_positive_patient_sum"] += float(
                    row_weights[list(positive_docs)].sum().item()
                )
                self.soft_mass["soft_mass_positive_patient_count"] += 1.0

    def stacked_logits_targets(self) -> tuple[Tensor, Tensor]:
        """Return concatenated logits/targets (or empty tensors when no batches observed)."""
        logits = torch.cat(self.logits, dim=0) if self.logits else torch.empty(0)
        targets = torch.cat(self.targets, dim=0) if self.targets else torch.empty(0)
        return logits, targets

    def prediction_rate_metrics(self) -> dict[str, float]:
        """Return target/prediction rate diagnostics."""
        count = self.pred_stats["count"]
        if count <= 0:
            return {"pos_rate": 0.0, "pred_pos_rate_at_0_5": 0.0}
        return {
            "pos_rate": self.pred_stats["target_positive_sum"] / count,
            "pred_pos_rate_at_0_5": self.pred_stats["pred_positive_sum"] / count,
        }

    def soft_mass_metrics(self) -> dict[str, float]:
        """Return soft retrieval mass diagnostics."""
        soft_any_count = self.soft_mass["soft_mass_any_patient_count"]
        soft_positive_count = self.soft_mass["soft_mass_positive_patient_count"]
        return {
            "retrieval/soft_mass_on_any_patient_docs": (
                self.soft_mass["soft_mass_any_patient_sum"] / soft_any_count if soft_any_count > 0 else 0.0
            ),
            "retrieval/soft_mass_on_patient_positive_docs": (
                self.soft_mass["soft_mass_positive_patient_sum"] / soft_positive_count
                if soft_positive_count > 0
                else 0.0
            ),
        }

    def hard_retrieval_metrics(self) -> dict[str, float]:
        """Return hard top-1 retrieval diagnostics."""
        if not self.top_doc_ids:
            return empty_hard_retrieval_metrics()
        metric_tensors = compute_retrieval_batch_metrics(
            top1_doc_ids=torch.cat(self.top_doc_ids, dim=0),
            top1_doc_labels=torch.cat(self.top_doc_labels, dim=0),
            targets=torch.cat(self.targets, dim=0),
            true_doc_ids=self.true_doc_ids,
            true_positive_doc_ids=self.true_positive_doc_ids,
        )
        return {name: float(value.item()) for name, value in metric_tensors.items()}
