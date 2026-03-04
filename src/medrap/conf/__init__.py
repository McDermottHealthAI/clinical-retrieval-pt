"""Hydra config package for medrap CLI."""

from hydra.core.config_store import ConfigStore
from meds_torchdata import MEDSTorchDataConfig


def _register_meds_torch_data_config() -> None:
    """Register MEDSTorchDataConfig in Hydra's config store once."""
    cs = ConfigStore.instance()
    group = "datamodule/config"
    try:
        existing = cs.repo.get("datamodule", {}).get("config", {})
    except Exception:  # pragma: no cover - defensive for Hydra internals.
        existing = {}

    if "MEDSTorchDataConfig.yaml" not in existing:
        MEDSTorchDataConfig.add_to_config_store(group=group)


_register_meds_torch_data_config()
