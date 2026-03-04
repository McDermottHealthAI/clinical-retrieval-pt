"""Thin experiment callback for semi-synthetic retrieval diagnostics/reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torchmetrics.classification import BinaryAUROC

from medrap.retrieval_eval import RetrievalEvalState, binary_accuracy_from_logits

try:
    import lightning
except ModuleNotFoundError:
    lightning = None  # type: ignore[assignment]


def _missing_lightning() -> ModuleNotFoundError:
    return ModuleNotFoundError("lightning is required for SemiSyntheticEvalCallback")


if lightning is None:

    class SemiSyntheticEvalCallback:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise _missing_lightning()

else:

    class SemiSyntheticEvalCallback(lightning.Callback):
        """Compute experiment-specific retrieval diagnostics without custom Lightning module."""

        def __init__(self, *, output_dir: str, run_name: str) -> None:
            super().__init__()
            self.output_dir = Path(output_dir)
            self.run_name = str(run_name)
            self.summary: dict[str, Any] = {}

            self._val_auroc = BinaryAUROC()
            self._test_auroc = BinaryAUROC()
            self._val_state = RetrievalEvalState()
            self._test_state = RetrievalEvalState()
            self._test_examples = {
                "correct_retrieval_correct_classification": [],
                "incorrect_retrieval_correct_classification": [],
                "incorrect_classification": [],
            }

        def _collect_batch(
            self,
            *,
            stage: str,
            pl_module: lightning.LightningModule,
            batch: dict[str, Any],
        ) -> None:
            with torch.no_grad():
                output = pl_module.model(batch)

            if not hasattr(output, "logits"):
                return
            logits = output.logits
            if not isinstance(logits, Tensor):
                return

            targets = batch["targets"].float()
            probs = torch.sigmoid(logits)
            state = self._val_state if stage == "val" else self._test_state

            state.update_predictions(logits=logits, targets=targets)

            if stage == "val":
                self._val_auroc.update(probs, targets.long())
            else:
                self._test_auroc.update(probs, targets.long())

            if output.top_doc_ids is None or output.top_doc_labels is None:
                return

            state.update_retrieval(
                top_doc_ids=output.top_doc_ids,
                top_doc_labels=output.top_doc_labels,
                true_doc_ids=batch["true_doc_ids"],
                true_positive_doc_ids=batch["true_positive_doc_ids"],
                all_doc_weights=output.all_doc_weights,
            )

            if stage == "test":
                predicted = (torch.sigmoid(logits) >= 0.5).long()
                for idx, example_id in enumerate(batch["example_ids"]):
                    retrieved_doc_id = int(output.top_doc_ids[idx].item())
                    retrieved_doc_label = int(output.top_doc_labels[idx].item())
                    target = int(targets[idx].item())
                    pred = int(predicted[idx].item())
                    all_docs = {int(v) for v in batch["true_doc_ids"][idx]}
                    positive_docs = {int(v) for v in batch["true_positive_doc_ids"][idx]}

                    any_hit = retrieved_doc_id in all_docs
                    consistent = retrieved_doc_label == target
                    record = {
                        "example_id": str(example_id),
                        "target": target,
                        "predicted_label": pred,
                        "retrieved_doc_id": retrieved_doc_id,
                        "retrieved_doc_label": retrieved_doc_label,
                        "patient_doc_ids": sorted(all_docs),
                        "patient_positive_doc_ids": sorted(positive_docs),
                    }
                    if pred == target and any_hit:
                        self._test_examples["correct_retrieval_correct_classification"].append(record)
                    elif pred == target and (not any_hit) and consistent:
                        self._test_examples["incorrect_retrieval_correct_classification"].append(record)
                    elif pred != target:
                        self._test_examples["incorrect_classification"].append(record)

        def on_validation_epoch_start(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
        ) -> None:
            del trainer, pl_module
            self._val_state.reset()
            self._val_auroc.reset()

        def on_test_epoch_start(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
        ) -> None:
            del trainer, pl_module
            self._test_state.reset()
            self._test_auroc.reset()
            self._test_examples = {
                "correct_retrieval_correct_classification": [],
                "incorrect_retrieval_correct_classification": [],
                "incorrect_classification": [],
            }

        def on_validation_batch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            outputs: Any,
            batch: dict[str, Any],
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            del trainer, outputs, batch_idx, dataloader_idx
            self._collect_batch(stage="val", pl_module=pl_module, batch=batch)

        def on_test_batch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            outputs: Any,
            batch: dict[str, Any],
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            del trainer, outputs, batch_idx, dataloader_idx
            self._collect_batch(stage="test", pl_module=pl_module, batch=batch)

        def on_validation_epoch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
        ) -> None:
            del trainer
            logits, targets = self._val_state.stacked_logits_targets()
            val_metrics = {
                "val/auroc": float(torch.nan_to_num(self._val_auroc.compute(), nan=0.0).item()),
                "val/accuracy": binary_accuracy_from_logits(logits, targets),
            }
            pred_stats = self._val_state.prediction_rate_metrics()
            val_metrics["val/pos_rate"] = float(pred_stats["pos_rate"])
            val_metrics["val/pred_pos_rate_at_0_5"] = float(pred_stats["pred_pos_rate_at_0_5"])
            retrieval_metrics = {
                **self._val_state.hard_retrieval_metrics(),
                **self._val_state.soft_mass_metrics(),
            }
            for name, value in val_metrics.items():
                pl_module.log(name, value)
            for name, value in retrieval_metrics.items():
                pl_module.log(f"val/{name}", value)

        def on_test_epoch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
        ) -> None:
            del trainer
            logits, targets = self._test_state.stacked_logits_targets()
            test_metrics = {
                "test/auroc": float(torch.nan_to_num(self._test_auroc.compute(), nan=0.0).item()),
                "test/accuracy": binary_accuracy_from_logits(logits, targets),
            }
            pred_stats = self._test_state.prediction_rate_metrics()
            test_metrics["test/pos_rate"] = float(pred_stats["pos_rate"])
            test_metrics["test/pred_pos_rate_at_0_5"] = float(pred_stats["pred_pos_rate_at_0_5"])

            retrieval_metrics = {
                **self._test_state.hard_retrieval_metrics(),
                **self._test_state.soft_mass_metrics(),
            }
            for name, value in test_metrics.items():
                pl_module.log(name, value)
            for name, value in retrieval_metrics.items():
                pl_module.log(f"test/{name}", value)

            self.summary = {
                **test_metrics,
                **{f"test/{k}": float(v) for k, v in retrieval_metrics.items()},
                "example_cases": {key: value[:5] for key, value in self._test_examples.items()},
            }

            report_dir = self.output_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            with (report_dir / f"{self.run_name}_test_summary.json").open("w", encoding="utf-8") as f:
                json.dump(self.summary, f, indent=2, sort_keys=True)
