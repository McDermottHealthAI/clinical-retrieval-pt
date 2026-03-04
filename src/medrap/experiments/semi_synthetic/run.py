"""Hydra entrypoint for the first MIMIC-IV demo semi-synthetic retrieval experiment."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from medrap.lightning_module import MedRAPLightningModule
from medrap.task import BinaryClassificationTask

from ..common.reporting import (
    collect_numeric_metric_rows,
    print_run_summary,
    write_metric_rows_csv,
)
from .batch_adapter import SemiSyntheticBatchAdapter
from .callbacks import SemiSyntheticEvalCallback
from .corpus import build_synthetic_drug_corpus
from .datamodule import SemiSyntheticDataModule
from .modeling import SemiSyntheticRetrievalModel
from .preprocessing import prepare_mimic_demo


def _as_dict(node: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(node, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config node, got {type(payload)!r}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_from_cfg(cfg: DictConfig) -> int:
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "semi_synthetic_resolved.yaml", resolve=True)

    prepared_dir = output_dir / "prepared"
    corpus_dir = output_dir / "corpus"

    preprocessing_cfg = _as_dict(cfg.preprocessing)
    preprocessing_cfg["output_dir"] = str(prepared_dir)
    prepare_mimic_demo(preprocessing_cfg)

    corpus_cfg = _as_dict(cfg.corpus)
    corpus_cfg["prepared_dir"] = str(prepared_dir)
    corpus_cfg["output_dir"] = str(corpus_dir)
    corpus_summary = build_synthetic_drug_corpus(corpus_cfg)

    baselines = list(cfg.experiment.baselines)
    if not baselines:
        raise ValueError("At least one baseline must be configured")

    experiment_name = str(cfg.get("experiment_name", "semi_synthetic_mimic_demo"))
    experiment_results: dict[str, Any] = {}
    for baseline_name in baselines:
        baseline = str(baseline_name)
        print(f"[medrap] running baseline={baseline} for experiment={experiment_name}")
        datamodule_kwargs = _as_dict(cfg.datamodule)
        datamodule = SemiSyntheticDataModule(
            prepared_dir=str(prepared_dir),
            corpus_dir=str(corpus_dir),
            **datamodule_kwargs,
        )
        datamodule.setup("fit")

        if datamodule.doc_features is None or datamodule.doc_labels is None or datamodule.patient_dim is None:
            raise RuntimeError("SemiSyntheticDataModule did not initialize required tensors")

        model_kwargs = _as_dict(cfg.model)
        model = SemiSyntheticRetrievalModel(
            mode=baseline,
            patient_dim=int(datamodule.patient_dim),
            doc_features=datamodule.doc_features,
            doc_labels=datamodule.doc_labels,
            hidden_dim=int(model_kwargs.get("hidden_dim", 128)),
            encoder_depth=int(model_kwargs.get("encoder_depth", 2)),
            top_k=int(cfg.retriever.top_k),
            random_seed=int(cfg.experiment.seed),
        )

        lightning_module = MedRAPLightningModule(
            model=model,
            task=BinaryClassificationTask(label_field="target", metrics={}),
            batch_adapter=SemiSyntheticBatchAdapter(label_field="target"),
            optimizer=partial(
                torch.optim.AdamW,
                lr=float(cfg.optimizer.learning_rate),
                weight_decay=float(cfg.optimizer.weight_decay),
            ),
        )
        eval_callback = SemiSyntheticEvalCallback(output_dir=str(output_dir), run_name=baseline)
        trainer = instantiate(cfg.trainer)
        trainer.callbacks.append(eval_callback)
        trainer.fit(lightning_module, datamodule=datamodule)
        trainer.test(lightning_module, datamodule=datamodule)

        callback_metrics = {
            key: float(value.item()) if hasattr(value, "item") else float(value)
            for key, value in trainer.callback_metrics.items()
        }
        test_summary: dict[str, Any] = eval_callback.summary
        if not test_summary:
            raise RuntimeError("SemiSyntheticEvalCallback did not produce a test summary.")
        experiment_results[baseline] = {
            "experiment": experiment_name,
            "baseline": baseline,
            "test_summary": test_summary,
            "callback_metrics": callback_metrics,
        }
        print(f"[medrap] completed baseline={baseline}")

    metric_rows = collect_numeric_metric_rows(
        experiment_name=experiment_name,
        experiment_results=experiment_results,
    )
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = reports_dir / "metrics_by_baseline.json"
    metrics_csv_path = reports_dir / "metrics_by_baseline.csv"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2, sort_keys=True)
    write_metric_rows_csv(metrics_csv_path, metric_rows)

    cohort_summary_path = prepared_dir / "summary.json"
    report = {
        "experiment": {"name": experiment_name, "baselines": baselines},
        "cohort": json.loads(cohort_summary_path.read_text(encoding="utf-8")),
        "corpus": corpus_summary,
        "baselines": experiment_results,
        "metric_rows_path": str(reports_dir / "metrics_by_baseline.csv"),
    }
    report_path = output_dir / "experiment_report.json"
    _write_json(report_path, report)
    print_run_summary(
        output_dir=output_dir,
        report_path=report_path,
        metrics_csv_path=metrics_csv_path,
        metrics_json_path=metrics_json_path,
        baselines=baselines,
        experiment_name=experiment_name,
    )
    return 0


@hydra.main(version_base=None, config_path="../../conf/experiments", config_name="semi_synthetic_mimic_demo")
def run_main(cfg: DictConfig) -> int:
    return run_from_cfg(cfg)
