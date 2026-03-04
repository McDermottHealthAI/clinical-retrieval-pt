"""Synthetic corpus construction for semi-synthetic retrieval experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

TOKEN_RX = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RX.findall(text.lower())


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_synthetic_drug_corpus(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build one synthetic document per observed drug in prepared data vocab."""
    prepared_dir = Path(cfg["prepared_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    mapping_path = Path(cfg["label_mapping_path"]).expanduser().resolve()
    category_name = str(cfg.get("category_name", "beta_blocker"))
    exclude_unresolved = bool(cfg.get("exclude_unresolved", True))

    drug_vocab: list[str] = list(_read_json(prepared_dir / "drug_vocab.json"))
    label_mapping_raw = _read_json(mapping_path)
    label_mapping = {str(key).strip().lower(): int(value) for key, value in label_mapping_raw.items()}

    docs: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for drug_name in drug_vocab:
        if drug_name not in label_mapping:
            unresolved.append(drug_name)
            if exclude_unresolved:
                continue
            label = 0
        else:
            label = label_mapping[drug_name]

        text = (
            f"Drug {drug_name} is a {category_name}."
            if label == 1
            else f"Drug {drug_name} is not a {category_name}."
        )

        docs.append(
            {
                "doc_id": len(docs),
                "drug_name": drug_name,
                "text": text,
                "metadata": {
                    "drug_name": drug_name,
                    "category_name": category_name,
                    "category_label": label,
                    "source": "semi_synthetic_template",
                },
            }
        )

    if not docs:
        raise ValueError("Synthetic corpus is empty. Check mapping coverage and unresolved policy.")

    token_vocab: dict[str, int] = {}
    for doc in docs:
        for token in _tokenize(doc["text"]):
            token_vocab.setdefault(token, len(token_vocab))

    matrix = torch.zeros((len(docs), len(token_vocab)), dtype=torch.float32)
    for idx, doc in enumerate(docs):
        for token in _tokenize(doc["text"]):
            token_id = token_vocab[token]
            matrix[idx, token_id] += 1.0

    row_norm = matrix.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    normalized_matrix = matrix / row_norm

    doc_labels = torch.tensor([int(doc["metadata"]["category_label"]) for doc in docs], dtype=torch.long)
    doc_ids = torch.tensor([int(doc["doc_id"]) for doc in docs], dtype=torch.long)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "documents.jsonl", docs)
    torch.save(
        {
            "doc_ids": doc_ids,
            "doc_labels": doc_labels,
            "doc_features": normalized_matrix,
            "token_vocab": token_vocab,
            "drug_names": [doc["drug_name"] for doc in docs],
        },
        output_dir / "features.pt",
    )

    summary = {
        "num_documents": len(docs),
        "num_unique_tokens": len(token_vocab),
        "num_positive_documents": int((doc_labels == 1).sum().item()),
        "num_negative_documents": int((doc_labels == 0).sum().item()),
        "num_unresolved_drugs": len(unresolved),
        "excluded_unresolved": exclude_unresolved,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "unresolved_drugs.json", sorted(unresolved))
    return summary
