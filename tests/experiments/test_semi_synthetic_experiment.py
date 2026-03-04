from __future__ import annotations

import json

import polars as pl
import pytest
import torch

from medrap.experiments.common.reporting import collect_numeric_metric_rows
from medrap.experiments.semi_synthetic.backend import DotProductTopKSearch
from medrap.experiments.semi_synthetic.corpus import build_synthetic_drug_corpus
from medrap.experiments.semi_synthetic.modeling import SemiSyntheticRetrievalModel
from medrap.experiments.semi_synthetic.preprocessing import prepare_mimic_demo
from medrap.retrieval_eval import compute_retrieval_batch_metrics


def _write_mapping(path) -> None:
    payload = {
        "metoprolol": 1,
        "aspirin": 0,
        "furosemide": 0,
        "atenolol": 1,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_tiny_meds_dataset(root) -> None:
    data_train = root / "data" / "train"
    data_tuning = root / "data" / "tuning"
    data_held_out = root / "data" / "held_out"
    metadata = root / "metadata"
    data_train.mkdir(parents=True, exist_ok=True)
    data_tuning.mkdir(parents=True, exist_ok=True)
    data_held_out.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    train_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "hadm_id": [100, 100, 200, 200],
            "code": [
                "MEDICATION//START//Metoprolol",
                "MEDICATION//START//Aspirin",
                "MEDICATION//START//Furosemide",
                "LAB//50912//mg/dL",
            ],
        }
    )
    held_out_df = pl.DataFrame(
        {
            "subject_id": [3],
            "hadm_id": [300],
            "code": ["MEDICATION//START//Atenolol"],
        }
    )
    train_df.write_parquet(data_train / "0.parquet")
    held_out_df.write_parquet(data_held_out / "0.parquet")
    pl.DataFrame(
        {
            "subject_id": pl.Series([], dtype=pl.Int64),
            "hadm_id": pl.Series([], dtype=pl.Int64),
            "code": pl.Series([], dtype=pl.Utf8),
        }
    ).write_parquet(data_tuning / "0.parquet")
    pl.DataFrame({"subject_id": [1, 2, 3], "split": ["train", "train", "held_out"]}).write_parquet(
        metadata / "subject_splits.parquet"
    )


def test_preprocessing_outputs_and_subject_split_reproducibility(tmp_path) -> None:
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path)
    meds_root = tmp_path / "meds_demo"
    _write_tiny_meds_dataset(meds_root)

    cfg = {
        "meds_root": str(meds_root),
        "label_mapping_path": str(mapping_path),
        "output_dir": str(tmp_path / "prepared_a"),
        "unresolved_policy": "exclude",
        "split_seed": 7,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
    }
    summary_a = prepare_mimic_demo(cfg)

    cfg_b = dict(cfg)
    cfg_b["output_dir"] = str(tmp_path / "prepared_b")
    prepare_mimic_demo(cfg_b)

    assert summary_a["num_examples"] > 0
    assert (tmp_path / "prepared_a" / "examples.jsonl").exists()
    assert (tmp_path / "prepared_a" / "splits.json").exists()
    assert (tmp_path / "prepared_a" / "drug_vocab.json").exists()

    splits_a = json.loads((tmp_path / "prepared_a" / "splits.json").read_text(encoding="utf-8"))
    splits_b = json.loads((tmp_path / "prepared_b" / "splits.json").read_text(encoding="utf-8"))
    assert splits_a == splits_b

    train_ids = set(splits_a["train"])
    val_ids = set(splits_a["val"])
    test_ids = set(splits_a["test"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_preprocessing_reads_meds_root_parquet_shards(tmp_path) -> None:
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path)

    meds_root = tmp_path / "meds_demo"
    _write_tiny_meds_dataset(meds_root)

    cfg = {
        "meds_root": str(meds_root),
        "label_mapping_path": str(mapping_path),
        "output_dir": str(tmp_path / "prepared"),
        "unresolved_policy": "exclude",
        "split_seed": 7,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
    }
    summary = prepare_mimic_demo(cfg)
    assert summary["input_mode"] == "meds"
    assert summary["num_examples"] >= 2
    assert summary["num_unique_drugs"] >= 3


def test_preprocessing_requires_meds_root(tmp_path) -> None:
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path)
    with pytest.raises(ValueError, match="preprocessing.meds_root must be set"):
        prepare_mimic_demo(
            {
                "label_mapping_path": str(mapping_path),
                "output_dir": str(tmp_path / "prepared"),
            }
        )


def test_corpus_construction_from_prepared_vocab(tmp_path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir(parents=True)
    (prepared_dir / "drug_vocab.json").write_text(json.dumps(["metoprolol", "aspirin"]), encoding="utf-8")

    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path)

    summary = build_synthetic_drug_corpus(
        {
            "prepared_dir": str(prepared_dir),
            "output_dir": str(tmp_path / "corpus"),
            "label_mapping_path": str(mapping_path),
            "category_name": "beta blocker",
            "exclude_unresolved": True,
        }
    )

    assert summary["num_documents"] == 2
    features = torch.load(tmp_path / "corpus" / "features.pt", map_location="cpu")
    assert features["doc_features"].shape[0] == 2
    assert features["doc_labels"].tolist() == [1, 0]


def test_dense_backend_search_shapes_and_candidate_restriction() -> None:
    doc_keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    backend = DotProductTopKSearch(doc_keys)

    queries = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
    doc_ids, scores = backend.search(queries, top_k=1)
    assert tuple(doc_ids.shape) == (2, 1)
    assert tuple(scores.shape) == (2, 1)

    restricted_ids, _ = backend.search(queries, top_k=1, candidate_doc_ids=[[1], [2]])
    assert restricted_ids.tolist() == [[1], [2]]


def _train_accuracy(model: SemiSyntheticRetrievalModel, epochs: int = 80) -> float:
    patient_features = torch.eye(4)
    true_doc_ids = [[0], [1], [2], [3]]
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(patient_features=patient_features, true_doc_ids=true_doc_ids)
        loss = criterion(outputs.logits, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        outputs = model(patient_features=patient_features, true_doc_ids=true_doc_ids)
        preds = (torch.sigmoid(outputs.logits) >= 0.5).float()
        return float((preds == targets).float().mean().item())


def test_learned_retrieval_beats_random_on_tiny_fixture() -> None:
    doc_features = torch.eye(4)
    doc_labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    learned = SemiSyntheticRetrievalModel(
        mode="learned",
        patient_dim=4,
        doc_features=doc_features,
        doc_labels=doc_labels,
        hidden_dim=32,
        encoder_depth=1,
        top_k=1,
        random_seed=3,
    )
    random_model = SemiSyntheticRetrievalModel(
        mode="random",
        patient_dim=4,
        doc_features=doc_features,
        doc_labels=doc_labels,
        hidden_dim=32,
        encoder_depth=1,
        top_k=1,
        random_seed=3,
    )

    learned_acc = _train_accuracy(learned, epochs=80)
    random_acc = _train_accuracy(random_model, epochs=80)

    assert learned_acc > random_acc


def test_equivalence_class_metric_flags_label_consistent_patient_inconsistent() -> None:
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
    assert float(metrics["retrieval/top1_hit_any_positive_drug"]) == 0.0
    assert float(metrics["retrieval/label_consistent_patient_inconsistent_rate"]) == 1.0


def test_oracle_positive_prefers_positive_patient_doc() -> None:
    model = SemiSyntheticRetrievalModel(
        mode="oracle_positive",
        patient_dim=4,
        doc_features=torch.eye(4),
        doc_labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        hidden_dim=16,
        encoder_depth=1,
        top_k=1,
        random_seed=1,
    )
    patient_features = torch.eye(2, 4)
    outputs = model(
        patient_features=patient_features,
        true_doc_ids=[[0, 1], [2, 3]],
        true_positive_doc_ids=[[1], [3]],
    )
    assert outputs.top_doc_ids is not None
    assert outputs.top_doc_ids.tolist() == [1, 3]


def test_collect_metric_rows_adds_experiment_and_baseline_context() -> None:
    rows = collect_numeric_metric_rows(
        experiment_name="semi_synthetic_mimic_demo",
        experiment_results={
            "learned": {
                "test_summary": {"test/auroc": 0.9, "example_cases": {}},
                "callback_metrics": {"test/loss": 0.5},
            }
        },
    )

    assert rows == [
        {
            "experiment": "semi_synthetic_mimic_demo",
            "baseline": "learned",
            "source": "test_summary",
            "metric": "test/auroc",
            "value": 0.9,
        },
        {
            "experiment": "semi_synthetic_mimic_demo",
            "baseline": "learned",
            "source": "callback_metrics",
            "metric": "test/loss",
            "value": 0.5,
        },
    ]
