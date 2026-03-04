"""CLI entrypoints for medrap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .config_validation import validate_task_mode_config
from .experiments.semi_synthetic.run import run_main as _semi_synthetic_hydra_main
from .runtime import build_example_batch, build_model_from_cfg


def _persist_resolved_cfg(cfg: DictConfig, *, output_dir: str | Path, name: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path / name, resolve=True)


def _run_train_cfg(cfg: DictConfig) -> int:
    model = build_model_from_cfg(cfg)

    if cfg.get("run_smoke", False):
        out = model.forward(build_example_batch())
        print(out.logits)
        return 0

    validate_task_mode_config(cfg)
    _persist_resolved_cfg(cfg, output_dir=cfg.get("output_dir", "."), name="train_resolved.yaml")

    datamodule = instantiate(cfg.datamodule)
    lightning_module = instantiate(cfg.lightning_module, model=model)
    trainer = instantiate(cfg.trainer)
    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=cfg.get("resume_from_checkpoint"),
    )
    if cfg.get("run_test_after_fit", False):
        trainer.test(
            model=lightning_module,
            datamodule=datamodule,
            ckpt_path=cfg.get("test_ckpt_path"),
        )
    return 0


def _run_eval_cfg(cfg: DictConfig) -> int:
    model = build_model_from_cfg(cfg)
    if cfg.get("run_smoke", True):
        out = model.forward(build_example_batch())
        print(out.logits)
    else:
        validate_task_mode_config(cfg)
        _persist_resolved_cfg(cfg, output_dir=cfg.get("output_dir", "."), name="eval_resolved.yaml")

        datamodule = instantiate(cfg.datamodule)
        lightning_module = instantiate(cfg.lightning_module, model=model)
        trainer = instantiate(cfg.trainer)
        eval_split = str(cfg.get("eval_split", "test"))
        ckpt_path = cfg.get("checkpoint_path")
        if eval_split == "test":
            trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
        elif eval_split in {"val", "validate"}:
            trainer.validate(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            raise ValueError(f"Unsupported eval_split={eval_split!r}. Use one of: test, val, validate.")
    return 0


@hydra.main(version_base=None, config_path="conf", config_name="_train")
def _train_hydra(cfg: DictConfig) -> int:
    return _run_train_cfg(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="_eval")
def _eval_hydra(cfg: DictConfig) -> int:
    return _run_eval_cfg(cfg)


def train_main(overrides: list[str] | None = None) -> int:
    """Run the Hydra-native train entrypoint."""
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0] if old_argv else "medrap-train", *(list(overrides or []))]
        result = _train_hydra()
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def eval_main(overrides: list[str] | None = None) -> int:
    """Run the Hydra-native eval entrypoint."""
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0] if old_argv else "medrap-eval", *(list(overrides or []))]
        result = _eval_hydra()
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def semi_synthetic_mimic_main(overrides: list[str] | None = None) -> int:
    """Run the semi-synthetic MIMIC demo retrieval experiment."""
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0] if old_argv else "medrap-semi-synthetic-mimic", *(list(overrides or []))]
        result = _semi_synthetic_hydra_main()
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> int:
    """Dispatch medrap subcommands to Hydra-native entrypoints."""
    parser = argparse.ArgumentParser(prog="medrap")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd in ("train", "eval", "semi-synthetic-mimic"):
        sub = subparsers.add_parser(cmd)
        sub.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. run_smoke=false")

    args = parser.parse_args(argv)
    if args.command == "train":
        return train_main(args.overrides)
    if args.command == "eval":
        return eval_main(args.overrides)
    return semi_synthetic_mimic_main(args.overrides)
