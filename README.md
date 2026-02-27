# clinical-retrieval-pt

[![Status: WIP](https://img.shields.io/badge/status-WIP-orange)](https://github.com/McDermottHealthAI/clinical-retrieval-pt)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/clinical-retrieval-pt)](https://pypi.org/project/clinical-retrieval-pt/)
[![Documentation Status](https://readthedocs.org/projects/clinical-retrieval-pt/badge/?version=latest)](https://clinical-retrieval-pt.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/McDermottHealthAI/clinical-retrieval-pt/actions/workflows/tests.yaml/badge.svg)](https://github.com/McDermottHealthAI/clinical-retrieval-pt/actions/workflows/tests.yaml)
[![Test Coverage](https://codecov.io/github/McDermottHealthAI/clinical-retrieval-pt/graph/badge.svg)](https://codecov.io/github/McDermottHealthAI/clinical-retrieval-pt)
[![Code Quality](https://github.com/McDermottHealthAI/clinical-retrieval-pt/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/McDermottHealthAI/clinical-retrieval-pt/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/McDermottHealthAI/clinical-retrieval-pt.svg)](https://github.com/McDermottHealthAI/clinical-retrieval-pt/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/McDermottHealthAI/clinical-retrieval-pt/pulls)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Retrieval-augmented pretraining (RAP) for MEDS-style EHR data.

## Status

This is a work-in-progress.

Implemented now:

- a concrete pipeline orchestrator (`RetrievalAugmentedModel`)
- simple concrete stage components for smoke usage and examples
- `MEDSCodeEncoder`, which consumes `batch.code` from MEDS-style batches
- a small end-to-end doctest example in `model.py`

## Quickstart (Synthetic MEDS Batch)

```python
import torch

from clinical_retrieval_pt.encoders import MEDSCodeEncoder
from clinical_retrieval_pt.fusion import ReplaceFusion
from clinical_retrieval_pt.heads import IdentityHead
from clinical_retrieval_pt.model import RetrievalAugmentedModel
from clinical_retrieval_pt.pooling import IdentityPooling
from clinical_retrieval_pt.query_projection import IdentityQueryProjector
from clinical_retrieval_pt.retrieval_encoder import IdentityRetrievalEncoder
from clinical_retrieval_pt.retrievers import StaticRetriever
from meds_torchdata import MEDSTorchBatch

model = RetrievalAugmentedModel(
    encoder=MEDSCodeEncoder(),
    query_projector=IdentityQueryProjector(),
    retriever=StaticRetriever(doc_tokens=[[1.0, 2.0]], doc_attention_mask=[[1, 1]]),
    retrieval_encoder=IdentityRetrievalEncoder(),
    fusion=ReplaceFusion(),
    pooling=IdentityPooling(),
    head=IdentityHead(),
)

batch = MEDSTorchBatch(
    code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
    numeric_value=torch.zeros((2, 3), dtype=torch.float32),
    numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
    time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
)
out = model.forward(batch)
print(out.logits)  # [[1.0, 2.0]]
```

## MEDS Batch Typing

`MEDSCodeEncoder` accepts `meds_torchdata.MEDSTorchBatch` directly.
