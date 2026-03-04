from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import medrap.cli as cli
from medrap.cli import eval_main, main, train_main

HAS_LIGHTNING = bool(importlib.util.find_spec("lightning")) or bool(
    importlib.util.find_spec("pytorch_lightning")
)


def test_medrap_train_cli_runs_with_overrides() -> None:
    assert main(["train", "run_smoke=true"]) == 0


def test_medrap_eval_cli_runs_with_overrides() -> None:
    assert main(["eval", "run_smoke=false", "datamodule=demo"]) == 0


def test_train_entrypoint_runs_with_hydra_overrides() -> None:
    assert train_main(["run_smoke=true"]) == 0


def test_eval_entrypoint_runs_with_hydra_overrides() -> None:
    assert eval_main(["run_smoke=false", "datamodule=demo"]) == 0


def test_semi_synthetic_subcommand_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_run(overrides: list[str] | None = None) -> int:
        called["overrides"] = list(overrides or [])
        return 0

    monkeypatch.setattr(cli, "semi_synthetic_mimic_main", fake_run)
    assert main(["semi-synthetic-mimic", "foo=bar"]) == 0
    assert called["overrides"] == ["foo=bar"]


@dataclass
class _RecorderTrainer:
    fit_kwargs: dict[str, Any] | None = None
    test_kwargs: dict[str, Any] | None = None
    validate_kwargs: dict[str, Any] | None = None

    def fit(self, **kwargs: Any) -> None:
        self.fit_kwargs = kwargs

    def test(self, **kwargs: Any) -> None:
        self.test_kwargs = kwargs

    def validate(self, **kwargs: Any) -> None:
        self.validate_kwargs = kwargs


def test_run_train_cfg_forwards_resume_ckpt_and_optional_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cfg = OmegaConf.create(
        {
            "run_smoke": False,
            "output_dir": str(tmp_path),
            "resume_from_checkpoint": "resume.ckpt",
            "run_test_after_fit": True,
            "test_ckpt_path": "best.ckpt",
            "datamodule": {"_target_": "dummy.datamodule"},
            "lightning_module": {"_target_": "dummy.module"},
            "trainer": {"_target_": "dummy.trainer"},
        }
    )
    model = object()
    datamodule = object()
    lightning_module = object()
    trainer = _RecorderTrainer()

    def fake_build_model_from_cfg(_: object) -> object:
        return model

    def fake_instantiate(node: object, **kwargs: Any) -> object:
        if node is cfg.datamodule:
            return datamodule
        if node is cfg.lightning_module:
            assert kwargs["model"] is model
            return lightning_module
        if node is cfg.trainer:
            return trainer
        raise AssertionError(f"unexpected instantiate node: {node!r}")

    monkeypatch.setattr(cli, "build_model_from_cfg", fake_build_model_from_cfg)
    monkeypatch.setattr(cli, "instantiate", fake_instantiate)

    assert cli._run_train_cfg(cfg) == 0
    assert trainer.fit_kwargs is not None
    assert trainer.fit_kwargs["model"] is lightning_module
    assert trainer.fit_kwargs["datamodule"] is datamodule
    assert trainer.fit_kwargs["ckpt_path"] == "resume.ckpt"

    assert trainer.test_kwargs is not None
    assert trainer.test_kwargs["model"] is lightning_module
    assert trainer.test_kwargs["datamodule"] is datamodule
    assert trainer.test_kwargs["ckpt_path"] == "best.ckpt"
    assert (tmp_path / "train_resolved.yaml").exists()


@pytest.mark.parametrize(
    ("eval_split", "expected_call"),
    [
        ("test", "test"),
        ("val", "validate"),
    ],
)
def test_run_eval_cfg_runs_selected_split(
    eval_split: str, expected_call: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cfg = OmegaConf.create(
        {
            "run_smoke": False,
            "output_dir": str(tmp_path),
            "checkpoint_path": "model.ckpt",
            "eval_split": eval_split,
            "datamodule": {"_target_": "dummy.datamodule"},
            "lightning_module": {"_target_": "dummy.module"},
            "trainer": {"_target_": "dummy.trainer"},
        }
    )
    model = object()
    datamodule = object()
    lightning_module = object()
    trainer = _RecorderTrainer()

    def fake_build_model_from_cfg(_: object) -> object:
        return model

    def fake_instantiate(node: object, **kwargs: Any) -> object:
        if node is cfg.datamodule:
            return datamodule
        if node is cfg.lightning_module:
            assert kwargs["model"] is model
            return lightning_module
        if node is cfg.trainer:
            return trainer
        raise AssertionError(f"unexpected instantiate node: {node!r}")

    monkeypatch.setattr(cli, "build_model_from_cfg", fake_build_model_from_cfg)
    monkeypatch.setattr(cli, "instantiate", fake_instantiate)

    assert cli._run_eval_cfg(cfg) == 0
    if expected_call == "test":
        assert trainer.test_kwargs is not None
        assert trainer.validate_kwargs is None
        assert trainer.test_kwargs["ckpt_path"] == "model.ckpt"
    else:
        assert trainer.validate_kwargs is not None
        assert trainer.test_kwargs is None
        assert trainer.validate_kwargs["ckpt_path"] == "model.ckpt"
    assert (tmp_path / "eval_resolved.yaml").exists()


def test_run_eval_cfg_rejects_unknown_split(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "run_smoke": False,
            "output_dir": str(tmp_path),
            "checkpoint_path": None,
            "eval_split": "unknown",
            "datamodule": {"_target_": "dummy.datamodule"},
            "lightning_module": {"_target_": "dummy.module"},
            "trainer": {"_target_": "dummy.trainer"},
        }
    )

    monkeypatch.setattr(cli, "build_model_from_cfg", lambda _: object())
    monkeypatch.setattr(cli, "instantiate", lambda *_args, **_kwargs: object())

    with pytest.raises(ValueError, match="eval_split"):
        cli._run_eval_cfg(cfg)


def test_train_and_eval_checkpoint_roundtrip(tmp_path) -> None:
    if not HAS_LIGHTNING:
        pytest.skip("lightning is not available")

    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        train_cfg = compose(
            config_name="_train",
            overrides=[
                "run_smoke=false",
                "datamodule=demo",
                f"output_dir={tmp_path}",
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=1",
                "trainer.limit_val_batches=1",
                "trainer.enable_checkpointing=true",
                "trainer.logger=false",
                "trainer.enable_model_summary=false",
            ],
        )
    assert cli._run_train_cfg(train_cfg) == 0

    checkpoints = sorted(tmp_path.rglob("*.ckpt"))
    assert checkpoints, "expected at least one checkpoint from trainer.fit"
    checkpoint = checkpoints[0]

    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        eval_cfg = compose(
            config_name="_eval",
            overrides=[
                "run_smoke=false",
                "datamodule=demo",
                f"output_dir={tmp_path}",
                f"checkpoint_path='{checkpoint}'",
                "eval_split=test",
                "trainer.limit_test_batches=1",
                "trainer.logger=false",
                "trainer.enable_checkpointing=false",
                "trainer.enable_model_summary=false",
                "trainer.callbacks=[]",
            ],
        )
    assert cli._run_eval_cfg(eval_cfg) == 0
