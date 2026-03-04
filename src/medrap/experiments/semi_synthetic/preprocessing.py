"""MIMIC-IV demo medication preprocessing for semi-synthetic retrieval experiments."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

NORMALIZE_RX = re.compile(r"[^a-z0-9]+")
DEFAULT_MEDICATION_CODE_REGEX = r"^MEDICATION//"
DEFAULT_MEDICATION_NAME_REGEX = r"^MEDICATION//(?:START|STOP)//(.+)$"


def normalize_drug_name(name: str) -> str:
    """Normalize drug names into stable lowercase canonical tokens."""
    stripped = NORMALIZE_RX.sub(" ", name.lower()).strip()
    return " ".join(stripped.split())


def _resolve_drug_to_mapping_key(normalized: str, mapping: dict[str, int]) -> str | None:
    """Resolve free-text normalized drug names onto explicit mapping keys when possible."""
    if normalized in mapping:
        return normalized

    # Prefer more specific keys first, e.g. "metoprolol succinate" before "metoprolol".
    for key in sorted(mapping, key=len, reverse=True):
        if key in normalized:
            return key
    return None


def _is_meds_root(path: Path) -> bool:
    return (path / "data").is_dir() and (path / "metadata").is_dir()


def _resolve_meds_root(path: Path) -> Path:
    """Resolve a MEDS dataset root that contains `data/` and `metadata/`."""
    candidate = path.expanduser().resolve()
    if _is_meds_root(candidate):
        return candidate

    meds_cohort = candidate / "MEDS_cohort"
    if _is_meds_root(meds_cohort):
        return meds_cohort

    raise FileNotFoundError(f"{path} does not look like a MEDS root. Expected to find data/ and metadata/.")


def _load_label_mapping(mapping_path: Path) -> dict[str, int]:
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping: dict[str, int] = {}
    for key, value in raw.items():
        mapped = int(value)
        if mapped not in {0, 1}:
            raise ValueError(f"Label mapping values must be 0/1. Got {value!r} for {key!r}")
        mapping[normalize_drug_name(key)] = mapped
    return mapping


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_medication_codes_from_meds(cfg: dict[str, Any], meds_root: Path) -> dict[tuple[str, str], set[str]]:
    """Load subject/admission-level medication names from MEDS parquet shards."""
    subject_col = str(cfg.get("subject_col", "subject_id"))
    admission_col = str(cfg.get("admission_col", "hadm_id"))
    code_col = str(cfg.get("code_col", "code"))
    medication_code_regex = str(cfg.get("medication_code_regex", DEFAULT_MEDICATION_CODE_REGEX))
    medication_name_regex = cfg.get("medication_name_regex", DEFAULT_MEDICATION_NAME_REGEX)
    split_names = cfg.get("split_names", ["train", "tuning", "held_out"])
    max_shards_per_split = cfg.get("max_shards_per_split")
    max_rows_per_shard = cfg.get("max_rows_per_shard")

    grouped_drugs: dict[tuple[str, str], set[str]] = defaultdict(set)

    data_dir = meds_root / "data"
    for split_name in split_names:
        split_dir = data_dir / str(split_name)
        if not split_dir.exists():
            continue

        shard_paths = sorted(split_dir.glob("*.parquet"))
        if max_shards_per_split is not None:
            shard_paths = shard_paths[: int(max_shards_per_split)]

        for shard_path in shard_paths:
            columns = [subject_col, code_col]
            schema = pl.read_parquet_schema(shard_path)
            if admission_col in schema:
                columns.append(admission_col)

            df = pl.read_parquet(shard_path, columns=columns)
            if max_rows_per_shard is not None:
                df = df.head(int(max_rows_per_shard))

            if subject_col not in df.columns or code_col not in df.columns:
                raise ValueError(
                    f"Missing required MEDS columns in {shard_path}: "
                    f"expected {subject_col!r} and {code_col!r}."
                )

            work = df.with_columns(
                [
                    pl.col(subject_col).cast(pl.Utf8).alias("__subject_id"),
                    pl.col(code_col).cast(pl.Utf8).alias("__code"),
                    (
                        pl.col(admission_col).cast(pl.Utf8)
                        if admission_col in df.columns
                        else pl.lit(None, dtype=pl.Utf8)
                    ).alias("__admission_id"),
                ]
            ).filter(pl.col("__code").is_not_null())
            work = work.filter(pl.col("__code").str.contains(medication_code_regex))

            if medication_name_regex:
                work = work.with_columns(
                    pl.col("__code").str.extract(str(medication_name_regex), group_index=1).alias("__drug")
                )
            else:
                work = work.with_columns(pl.col("__code").str.split("//").list.last().alias("__drug"))

            for subject_id, admission_id, drug in work.select(
                "__subject_id", "__admission_id", "__drug"
            ).iter_rows():
                if not subject_id or not drug:
                    continue
                normalized = normalize_drug_name(str(drug))
                if not normalized:
                    continue
                admission = str(admission_id) if admission_id not in {None, ""} else f"subject_{subject_id}"
                grouped_drugs[(str(subject_id), admission)].add(normalized)

    return grouped_drugs


def prepare_mimic_demo(cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepare reproducible admission-level examples from a MEDS dataset root."""
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    mapping_path = Path(cfg["label_mapping_path"]).expanduser().resolve()
    unresolved_policy = str(cfg.get("unresolved_policy", "exclude"))
    if unresolved_policy not in {"exclude", "fail"}:
        raise ValueError("unresolved_policy must be 'exclude' or 'fail'")

    split_seed = int(cfg.get("split_seed", 13))
    val_fraction = float(cfg.get("val_fraction", 0.2))
    test_fraction = float(cfg.get("test_fraction", 0.2))
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction/test_fraction must be >0 and sum to <1")

    mapping = _load_label_mapping(mapping_path)

    meds_root_cfg = cfg.get("meds_root")
    if not meds_root_cfg:
        raise ValueError("preprocessing.meds_root must be set to a MEDS dataset root.")
    meds_root = _resolve_meds_root(Path(str(meds_root_cfg)))
    grouped_drugs = _load_medication_codes_from_meds(cfg, meds_root)

    unresolved_drugs: set[str] = set()
    examples: list[dict[str, Any]] = []

    for (subject_id, admission_id), drugs in grouped_drugs.items():
        canonical_drugs = []
        for drug in sorted(drugs):
            canonical = _resolve_drug_to_mapping_key(drug, mapping)
            canonical_drugs.append(canonical if canonical is not None else drug)
        drug_list = sorted(set(canonical_drugs))
        unresolved_here = [drug for drug in drug_list if drug not in mapping]
        if unresolved_here:
            unresolved_drugs.update(unresolved_here)
            if unresolved_policy == "exclude":
                drug_list = [drug for drug in drug_list if drug in mapping]
                if not drug_list:
                    continue

        positives = [drug for drug in drug_list if mapping.get(drug, 0) == 1]
        target = int(len(positives) > 0)
        example_id = f"{subject_id}:{admission_id}"

        examples.append(
            {
                "example_id": example_id,
                "subject_id": subject_id,
                "admission_id": admission_id,
                "drug_names": drug_list,
                "positive_drug_names": positives,
                "target": target,
            }
        )

    if unresolved_policy == "fail" and unresolved_drugs:
        sample = sorted(unresolved_drugs)[:20]
        raise ValueError(
            "Found unresolved normalized drug names in labeling mapping. "
            f"Count={len(unresolved_drugs)} sample={sample}"
        )

    if not examples:
        raise ValueError(
            "No usable examples after preprocessing. Check unresolved policy and mapping coverage."
        )

    subjects = sorted({row["subject_id"] for row in examples})
    rng = random.Random(split_seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_test = max(1, round(n_subjects * test_fraction))
    n_val = max(1, round(n_subjects * val_fraction))
    n_train = n_subjects - n_val - n_test
    if n_train < 1:
        raise ValueError("Split fractions leave no train subjects.")

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])
    test_subjects = set(subjects[n_train + n_val :])

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for example in examples:
        subject_id = example["subject_id"]
        if subject_id in train_subjects:
            splits["train"].append(example["example_id"])
        elif subject_id in val_subjects:
            splits["val"].append(example["example_id"])
        elif subject_id in test_subjects:
            splits["test"].append(example["example_id"])
        else:
            raise RuntimeError(f"Subject {subject_id!r} did not land in any split")

    drug_vocab = sorted({drug for example in examples for drug in example["drug_names"]})

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "examples.jsonl", examples)
    _write_json(output_dir / "splits.json", splits)
    _write_json(output_dir / "drug_vocab.json", drug_vocab)
    _write_json(output_dir / "unresolved_drugs.json", sorted(unresolved_drugs))

    target_sum = sum(int(example["target"]) for example in examples)
    summary = {
        "input_mode": "meds",
        "input_source": str(meds_root),
        "num_examples": len(examples),
        "num_subjects": n_subjects,
        "num_unique_drugs": len(drug_vocab),
        "num_positive_examples": target_sum,
        "num_negative_examples": len(examples) - target_sum,
        "class_balance_positive": target_sum / len(examples),
        "num_unresolved_drugs": len(unresolved_drugs),
        "unresolved_policy": unresolved_policy,
        "split_sizes": {key: len(value) for key, value in splits.items()},
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
