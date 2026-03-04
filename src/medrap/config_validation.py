"""Hydra config validation helpers."""

from __future__ import annotations

from omegaconf import DictConfig


def _normalize_seq_sampling_strategy(value: object) -> str:
    text = str(value)
    if "." in text:
        text = text.split(".")[-1]
    return text.strip().lower()


def validate_task_mode_config(cfg: DictConfig) -> None:
    """Validate supervised task-mode assumptions for meds-torch-data.

    If ``datamodule.config.task_labels_dir`` is set, we require
    ``datamodule.config.seq_sampling_strategy == to_end``.
    """

    datamodule_cfg = cfg.get("datamodule")
    if datamodule_cfg is None:
        return

    config = datamodule_cfg.get("config") if isinstance(datamodule_cfg, DictConfig) else None
    if config is None:
        return

    task_labels_dir = config.get("task_labels_dir")
    if task_labels_dir in (None, "null"):
        return

    seq_sampling = _normalize_seq_sampling_strategy(config.get("seq_sampling_strategy"))
    if seq_sampling != "to_end":
        raise ValueError(
            "Invalid supervised task-mode setup: when datamodule.config.task_labels_dir is set, "
            "datamodule.config.seq_sampling_strategy must be TO_END (end-of-sequence supervision)."
        )
