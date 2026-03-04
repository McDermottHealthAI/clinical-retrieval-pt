# medrap

[![Status: WIP](https://img.shields.io/badge/status-WIP-orange)](https://github.com/McDermottHealthAI/MedRAP)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/medrap)](https://pypi.org/project/medrap/)
[![Documentation Status](https://readthedocs.org/projects/MedRAP/badge/?version=latest)](https://medrap.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/McDermottHealthAI/MedRAP/actions/workflows/tests.yaml/badge.svg)](https://github.com/McDermottHealthAI/MedRAP/actions/workflows/tests.yaml)
[![Test Coverage](https://codecov.io/github/McDermottHealthAI/MedRAP/graph/badge.svg)](https://codecov.io/github/McDermottHealthAI/MedRAP)
[![Code Quality](https://github.com/McDermottHealthAI/MedRAP/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/McDermottHealthAI/MedRAP/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/McDermottHealthAI/MedRAP.svg)](https://github.com/McDermottHealthAI/MedRAP/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/McDermottHealthAI/MedRAP/pulls)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Retrieval-augmented pretraining/reasoning scaffold for MEDS-style EHR data.

## Scope

This package is organized to match MEDS ecosystem patterns:

- Data/config integration with `meds-torch-data`
- Hydra + Lightning app structure similar to `MEDS_EIC_AR`
- Task semantics based on MEDS LabelSchema fields:
    - `boolean_value`
    - `integer_value`
    - `float_value`
    - `categorical_value`

The Lightning module is task-agnostic orchestration. Task logic (loss, metrics, target handling) lives in task objects.

## Install

```bash
uv sync
```

## Config Layout

Top-level train/eval configs compose nested groups:

- `datamodule/`
- `lightning_module/`
- `model/`
- `optimizer/`
- `LR_scheduler/`
- `metrics/`
- `task/`
- `trainer/`
- `callbacks/`
- `logger/`

See [`src/medrap/conf`](src/medrap/conf).

## CLI

```bash
uv run medrap train ...
uv run medrap eval ...
```

`medrap` is a thin command dispatcher over Hydra-native train/eval entrypoints.

## Quick Smoke Run

```bash
uv run medrap train run_smoke=true
uv run medrap eval run_smoke=true
```

`run_smoke=true` uses an internal tiny batch builder (test-oriented), not the MEDS datamodule stack.

## Real MEDS Supervised Run

```bash
uv run medrap train \
	run_smoke=false \
	datamodule=task_supervised \
	datamodule.config.tensorized_cohort_dir=/abs/path/to/tensorized_cohort \
	datamodule.config.task_labels_dir=/abs/path/to/task_labels \
	task=binary \
	batch_adapter.label_field=boolean_value \
	model/head=linear
```

For task-label supervision, `seq_sampling_strategy=TO_END` is required and validated.

## Eval From Checkpoint

```bash
uv run medrap eval \
	run_smoke=false \
	checkpoint_path=/abs/path/to/checkpoints/epoch=0-step=123.ckpt \
	eval_split=test
```

Resolved train/eval configs are persisted to `output_dir` (`train_resolved.yaml`, `eval_resolved.yaml`).

## Synthetic Research Scaffolding

Synthetic retrieval-experiment primitives live under
[`src/medrap/experiments/synthetic`](src/medrap/experiments/synthetic) and include:

- oracle retriever
- learned key/query retriever
- corrupted-key retriever
- label-collision and continuous-target toy recipes

## Semi-Synthetic MIMIC-IV Demo Experiment

The first runnable experiment lives under
[`src/medrap/experiments/semi_synthetic`](src/medrap/experiments/semi_synthetic).

Design:

- input: patient drug sets from a MEDS-formatted dataset root (`data/*/*.parquet`)
- target: whether any patient drug is a beta blocker
- corpus: one synthetic document per normalized drug
- main model: retrieval-only downstream prediction (patient features are discarded after retrieval)

This setup intentionally exposes an under-specification/equivalence-class issue:
classification can be correct while top-1 retrieval is not patient-consistent.

### 1. Run the experiment

```bash
uv run medrap semi-synthetic-mimic \
	preprocessing.meds_root=/abs/path/to/MEDS_dataset_root \
	output_dir=/abs/path/to/output/dir
```

You can override any Hydra config in
[`src/medrap/conf/experiments/semi_synthetic_mimic_demo.yaml`](src/medrap/conf/experiments/semi_synthetic_mimic_demo.yaml).

### 2. What gets produced

`output_dir` will contain:

- `semi_synthetic_resolved.yaml`:
    - full resolved Hydra config used for the run
- `prepared/`:
    - `examples.jsonl`
    - `splits.json`
    - `drug_vocab.json`
    - `summary.json`
- `corpus/`:
    - `documents.jsonl`
    - `features.pt`
    - `summary.json`
- `reports/`:
    - `<baseline>_test_summary.json` per baseline
    - `metrics_by_baseline.csv`:
        - one row per metric with columns:
            - `experiment`
            - `baseline`
            - `source` (`test_summary` or `callback_metrics`)
            - `metric`
            - `value`
    - `metrics_by_baseline.json`: same content as JSON rows
- `experiment_report.json`:
    - cohort stats, corpus stats, and baseline metrics

### 3. Baselines included

- `learned`: trainable query encoder + dense retrieval
- `no_retrieval`: direct patient-vector classifier
- `oracle`: retrieval forced to true patient drug document
- `oracle_positive`: retrieval forced to true positive-category patient drug (when available)
- `random`: random document retrieval

### 4. Retrieval metrics reported

- top-1 hit against any true patient drug document
- top-1 positive-doc label rate (`top1_is_positive_doc_label`)
- top-1 hit against any true positive-category patient drug document
- label-consistent but patient-inconsistent retrieval rate
- learned-model soft retrieval mass on:
    - any patient docs
    - patient positive-category docs

Task reporting also includes `pos_rate` and `pred_pos_rate_at_0_5` for threshold calibration context.
