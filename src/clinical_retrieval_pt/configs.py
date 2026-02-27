"""Structured config objects and instantiation helpers.

This module provides a minimal hydra-zen based configuration layer for composing the scaffold RAP pipeline
from concrete components.
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from hydra_zen import builds, instantiate

from .encoders import MEDSCodeEncoder
from .fusion import ReplaceFusion
from .heads import IdentityHead
from .model import RetrievalAugmentedModel
from .pooling import IdentityPooling
from .query_projection import IdentityQueryProjector
from .retrieval_encoder import IdentityRetrievalEncoder
from .retrievers import StaticRetriever

MEDSCodeEncoderConfig = builds(
    MEDSCodeEncoder,
    zen_dataclass={"cls_name": "MEDSCodeEncoderConfig"},
)
IdentityQueryProjectorConfig = builds(
    IdentityQueryProjector,
    zen_dataclass={"cls_name": "IdentityQueryProjectorConfig"},
)
StaticRetrieverConfig = builds(
    StaticRetriever,
    populate_full_signature=True,
    zen_dataclass={"cls_name": "StaticRetrieverConfig"},
)
DemoStaticRetrieverConfig = builds(
    StaticRetriever,
    populate_full_signature=True,
    doc_tokens=[[1, 2]],
    doc_attention_mask=[[1, 1]],
    zen_dataclass={"cls_name": "DemoStaticRetrieverConfig"},
)
IdentityRetrievalEncoderConfig = builds(
    IdentityRetrievalEncoder,
    zen_dataclass={"cls_name": "IdentityRetrievalEncoderConfig"},
)
ReplaceFusionConfig = builds(
    ReplaceFusion,
    zen_dataclass={"cls_name": "ReplaceFusionConfig"},
)
IdentityPoolingConfig = builds(
    IdentityPooling,
    zen_dataclass={"cls_name": "IdentityPoolingConfig"},
)
IdentityHeadConfig = builds(
    IdentityHead,
    zen_dataclass={"cls_name": "IdentityHeadConfig"},
)


@dataclass
class PipelineConfig:
    """Configuration container for composing ``RetrievalAugmentedModel``."""

    # ``object`` keeps Hydra/OmegaConf structured-config compatibility while still
    # allowing stage-specific hydra-zen config objects.
    encoder: object = field(default_factory=MEDSCodeEncoderConfig)
    query_projector: object = field(default_factory=IdentityQueryProjectorConfig)
    retriever: object = field(default_factory=DemoStaticRetrieverConfig)
    retrieval_encoder: object = field(default_factory=IdentityRetrievalEncoderConfig)
    fusion: object = field(default_factory=ReplaceFusionConfig)
    pooling: object = field(default_factory=IdentityPoolingConfig)
    head: object = field(default_factory=IdentityHeadConfig)


@dataclass
class RAPAppConfig(PipelineConfig):
    """Top-level app config for CLI/Hydra composition."""

    run_smoke: bool = True

    @classmethod
    def add_to_config_store(cls, group: str | None = None) -> None:
        """Register this config in Hydra's ConfigStore.

        This follows the standard ``ConfigStore.store`` pattern used in
        MEDS ecosystem repos.
        """
        cs = ConfigStore.instance()
        cs.store(name=cls.__name__, group=group, node=cls)


def default_pipeline_config() -> PipelineConfig:
    """Return a default, fully-instantiable pipeline config."""
    return PipelineConfig()


def instantiate_model(config: PipelineConfig) -> RetrievalAugmentedModel:
    """Instantiate a ``RetrievalAugmentedModel`` from structured config."""
    return RetrievalAugmentedModel(
        encoder=instantiate(config.encoder),
        query_projector=instantiate(config.query_projector),
        retriever=instantiate(config.retriever),
        retrieval_encoder=instantiate(config.retrieval_encoder),
        fusion=instantiate(config.fusion),
        pooling=instantiate(config.pooling),
        head=instantiate(config.head),
    )
