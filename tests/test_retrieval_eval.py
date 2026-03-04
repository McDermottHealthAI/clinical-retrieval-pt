from __future__ import annotations

import pytest
import torch

from medrap.retrieval_eval import (
    RetrievalEvalState,
    binary_accuracy_from_logits,
    compute_retrieval_batch_metrics,
    empty_hard_retrieval_metrics,
    retrieval_doc_ids_are_sample_dependent,
    top1_recall_at_1,
)


def test_binary_accuracy_from_logits() -> None:
    logits = torch.tensor([3.0, -3.0, 0.1, -0.1])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert binary_accuracy_from_logits(logits, targets) == 1.0


def test_compute_retrieval_batch_metrics_flags_equivalence_failure() -> None:
    metrics = compute_retrieval_batch_metrics(
        top1_doc_ids=torch.tensor([2]),
        top1_doc_labels=torch.tensor([1]),
        targets=torch.tensor([1.0]),
        true_doc_ids=[[0, 1]],
        true_positive_doc_ids=[[0]],
    )
    assert float(metrics["retrieval/top1_hit_any_true_drug"]) == 0.0
    assert float(metrics["retrieval/top1_is_positive_doc_label"]) == 1.0
    assert float(metrics["retrieval/top1_hit_any_positive_drug_in_patient_set"]) == 0.0
    assert float(metrics["retrieval/label_consistent_patient_inconsistent_rate"]) == 1.0


def test_top1_recall_at_1() -> None:
    recall = top1_recall_at_1(
        torch.tensor([0, 2, 1]),
        torch.tensor([0, 1, 1]),
    )
    assert float(recall) == pytest.approx(2.0 / 3.0)


def test_retrieval_doc_ids_are_sample_dependent() -> None:
    assert retrieval_doc_ids_are_sample_dependent(torch.tensor([[1], [2], [3]]))
    assert not retrieval_doc_ids_are_sample_dependent(torch.tensor([[1], [1], [1]]))


def test_retrieval_eval_state_aggregates_and_resets() -> None:
    state = RetrievalEvalState()
    logits = torch.tensor([1.0, -1.0])
    targets = torch.tensor([1.0, 0.0])
    state.update_predictions(logits=logits, targets=targets)

    state.update_retrieval(
        top_doc_ids=torch.tensor([0, 2]),
        top_doc_labels=torch.tensor([1, 0]),
        true_doc_ids=[[0, 1], [2, 3]],
        true_positive_doc_ids=[[0], []],
        all_doc_weights=torch.tensor([[0.7, 0.2, 0.1, 0.0], [0.1, 0.1, 0.8, 0.0]]),
    )

    pred_rates = state.prediction_rate_metrics()
    assert pred_rates["pos_rate"] == 0.5
    assert pred_rates["pred_pos_rate_at_0_5"] == 0.5

    soft = state.soft_mass_metrics()
    assert soft["retrieval/soft_mass_on_any_patient_docs"] > 0.0
    assert soft["retrieval/soft_mass_on_patient_positive_docs"] > 0.0

    hard = state.hard_retrieval_metrics()
    assert hard["retrieval/top1_hit_any_true_drug"] == 1.0

    state.reset()
    assert state.hard_retrieval_metrics() == empty_hard_retrieval_metrics()
    assert state.prediction_rate_metrics() == {"pos_rate": 0.0, "pred_pos_rate_at_0_5": 0.0}
