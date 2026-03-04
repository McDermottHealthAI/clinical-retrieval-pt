from pathlib import Path

import pytest
from omegaconf import OmegaConf

from medrap.config_validation import validate_task_mode_config


def test_validate_task_mode_allows_unsupervised_random_sampling() -> None:
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "config": {
                    "task_labels_dir": None,
                    "seq_sampling_strategy": "RANDOM",
                }
            }
        }
    )

    validate_task_mode_config(cfg)


def test_validate_task_mode_requires_to_end_when_task_labels_dir_set() -> None:
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "config": {
                    "task_labels_dir": str(Path("/tmp/task_labels")),
                    "seq_sampling_strategy": "RANDOM",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="seq_sampling_strategy"):
        validate_task_mode_config(cfg)


def test_validate_task_mode_accepts_to_end_when_task_labels_dir_set() -> None:
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "config": {
                    "task_labels_dir": str(Path("/tmp/task_labels")),
                    "seq_sampling_strategy": "TO_END",
                }
            }
        }
    )

    validate_task_mode_config(cfg)
