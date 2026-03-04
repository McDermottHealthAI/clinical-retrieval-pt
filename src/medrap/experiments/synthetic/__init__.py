"""Synthetic retrieval experiment primitives."""

from .core import CorpusDocument, PatientSample, RetrievalExperimentBatch, RetrievalResult
from .corpus import SyntheticCorpus, build_toy_corpus
from .evaluation import mean_absolute_error, oracle_recall_at_1, retrieval_is_sample_dependent
from .retrievers import CorruptedKeyRetriever, LearnedKeyQueryRetriever, OracleRetriever
from .tasks import ContinuousTargetRecipe, LabelCollisionRecipe

__all__ = [
    "ContinuousTargetRecipe",
    "CorpusDocument",
    "CorruptedKeyRetriever",
    "LabelCollisionRecipe",
    "LearnedKeyQueryRetriever",
    "OracleRetriever",
    "PatientSample",
    "RetrievalExperimentBatch",
    "RetrievalResult",
    "SyntheticCorpus",
    "build_toy_corpus",
    "mean_absolute_error",
    "oracle_recall_at_1",
    "retrieval_is_sample_dependent",
]
