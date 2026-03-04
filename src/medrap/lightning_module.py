"""PyTorch Lightning orchestration module for MedRAP training."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from functools import partial
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor, nn

from .batch_adapter import MEDSSupervisedBatchAdapter
from .task import BinaryClassificationTask, SupervisedTask
from .types import ModelOutput

try:
    import lightning
except ModuleNotFoundError:
    lightning = None  # type: ignore[assignment]


def _missing_lightning() -> ModuleNotFoundError:
    return ModuleNotFoundError("lightning is required for MedRAPLightningModule")


def _factory_to_dict(factory: partial | None) -> dict[str, Any] | None:
    """Serialize a ``functools.partial`` so it can be saved as hyperparameters."""
    if factory is None:
        return None
    if not isinstance(factory, partial):
        raise TypeError(f"Expected partial factory, got {type(factory)!r}")
    if factory.args:
        raise ValueError(f"Expected no positional args in factory, got {factory.args!r}")

    kwargs = dict(factory.keywords or {})
    for key, value in list(kwargs.items()):
        if isinstance(value, DictConfig | ListConfig):
            kwargs[key] = OmegaConf.to_container(value, resolve=True)
    if "_target_" in kwargs:
        raise ValueError("Factory kwargs must not contain reserved key '_target_'.")

    target = f"{factory.func.__module__}.{factory.func.__qualname__}"
    return {"_target_": target, **kwargs}


def _normalize_factory(factory: object, *, name: str) -> partial | None:
    """Normalize factory values from Python/Hydra inputs."""
    if factory is None:
        return None
    if isinstance(factory, partial):
        return factory
    if isinstance(factory, DictConfig):
        if len(factory) == 0:
            return None
        converted = OmegaConf.to_container(factory, resolve=True)
        if not isinstance(converted, dict):
            raise TypeError(f"{name} DictConfig must resolve to a mapping, got {type(converted)!r}")
        factory = converted
    if isinstance(factory, Mapping):
        if len(factory) == 0:
            return None
        if "_target_" not in factory:
            raise TypeError(f"{name} mapping must include '_target_' when not empty.")
        return hydra.utils.instantiate(dict(factory), _partial_=True)
    if callable(factory):
        return partial(factory)
    raise TypeError(f"Unsupported {name} type: {type(factory)!r}")


if lightning is None:

    class MedRAPLightningModule(nn.Module):  # type: ignore[no-redef]
        """Placeholder when Lightning is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            super().__init__()
            raise _missing_lightning()

else:

    class MedRAPLightningModule(lightning.LightningModule):
        """Task-agnostic Lightning orchestration for MedRAP."""

        def __init__(
            self,
            *,
            model: nn.Module,
            task: SupervisedTask | None = None,
            batch_adapter: MEDSSupervisedBatchAdapter | None = None,
            optimizer: object = None,
            lr_scheduler: object = None,
        ) -> None:
            super().__init__()
            self.model = model
            self.task = task if task is not None else BinaryClassificationTask()
            self.batch_adapter = (
                batch_adapter
                if batch_adapter is not None
                else MEDSSupervisedBatchAdapter(label_field=self.task.label_field)
            )
            self.optimizer_factory = _normalize_factory(
                optimizer or partial(torch.optim.AdamW, lr=1e-3, weight_decay=0.01),
                name="optimizer",
            )
            self.lr_scheduler_factory = _normalize_factory(lr_scheduler, name="lr_scheduler")

            self.save_hyperparameters(
                {
                    "optimizer": _factory_to_dict(self.optimizer_factory),
                    "lr_scheduler": _factory_to_dict(self.lr_scheduler_factory),
                    "task_target": self.task.label_field,
                }
            )

        def forward(self, batch: object) -> Tensor:
            out = self.model(batch)
            if isinstance(out, ModelOutput):
                return out.logits
            if isinstance(out, Tensor):
                return out
            if hasattr(out, "logits"):
                logits = out.logits
                if isinstance(logits, Tensor):
                    return logits
            raise TypeError(f"Unexpected model output type: {type(out)!r}")

        @staticmethod
        def _is_norm_bias_param(name: str) -> bool:
            return bool(re.search(r"(bias|layer(_?)norm(\d*)\.weight)", name, re.IGNORECASE))

        def _norm_bias_params(self) -> Iterator[nn.Parameter]:
            for name, _ in self.named_parameters():
                if self._is_norm_bias_param(name):
                    yield self.get_parameter(name)

        def _non_norm_bias_params(self) -> Iterator[nn.Parameter]:
            for name, _ in self.named_parameters():
                if not self._is_norm_bias_param(name):
                    yield self.get_parameter(name)

        def _run_stage(self, batch: object, *, stage: str) -> Tensor:
            adapted = self.batch_adapter(batch)
            if adapted.label_field != self.task.label_field:
                raise ValueError(
                    f"Batch adapter produced label field {adapted.label_field!r} but task expects "
                    f"{self.task.label_field!r}."
                )

            logits = self.forward(adapted.model_input)
            step_out = self.task.step(logits, adapted.targets)

            is_train = stage == "train"
            batch_size = int(step_out.targets.shape[0])

            self._safe_log(
                f"{stage}/loss",
                step_out.loss,
                on_step=is_train,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            self._safe_log_dict(
                {f"{stage}/{k}": v for k, v in step_out.metrics.items()},
                on_step=is_train,
                on_epoch=True,
                batch_size=batch_size,
            )
            return step_out.loss

        def training_step(self, batch: object, batch_idx: int) -> Tensor:
            del batch_idx
            return self._run_stage(batch, stage="train")

        def validation_step(self, batch: object, batch_idx: int) -> Tensor:
            del batch_idx
            return self._run_stage(batch, stage="val")

        def test_step(self, batch: object, batch_idx: int) -> Tensor:
            del batch_idx
            return self._run_stage(batch, stage="test")

        def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
            params = [
                {"params": self._non_norm_bias_params()},
                {"params": self._norm_bias_params(), "weight_decay": 0.0},
            ]
            if self.optimizer_factory is None:
                raise RuntimeError("optimizer_factory cannot be None.")
            optimizer = self.optimizer_factory(params)

            if self.lr_scheduler_factory is None:
                return optimizer

            scheduler = self.lr_scheduler_factory(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        def _safe_log(self, name: str, value: Tensor, **kwargs: Any) -> None:
            if getattr(self, "_trainer", None) is not None:
                self.log(name, value, **kwargs)

        def _safe_log_dict(self, values: dict[str, Tensor], **kwargs: Any) -> None:
            if getattr(self, "_trainer", None) is not None:
                self.log_dict(values, **kwargs)
