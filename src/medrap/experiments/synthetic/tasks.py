"""Synthetic supervision recipes used for retrieval diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LabelCollisionRecipe:
    """Binary recipe where multiple drugs share the same target label."""

    positive_drug_ids: set[int]

    def target(self, drug_id: int) -> int:
        return int(drug_id in self.positive_drug_ids)


@dataclass(slots=True)
class ContinuousTargetRecipe:
    """Disambiguated regression recipe with unique per-drug targets."""

    values_by_drug_id: dict[int, float]

    def target(self, drug_id: int) -> float:
        return float(self.values_by_drug_id[drug_id])
