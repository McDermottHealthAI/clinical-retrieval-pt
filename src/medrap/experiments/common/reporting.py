"""Shared reporting helpers for experiment runners."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def collect_numeric_metric_rows(
    *,
    experiment_name: str,
    experiment_results: dict[str, Any],
    sources: tuple[str, ...] = ("test_summary", "callback_metrics"),
) -> list[dict[str, Any]]:
    """Flatten nested metric payloads into metric rows for CSV/JSON reporting."""
    rows: list[dict[str, Any]] = []
    for baseline, payload in experiment_results.items():
        for source in sources:
            metrics = payload.get(source, {})
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                if isinstance(value, bool | int | float):
                    rows.append(
                        {
                            "experiment": experiment_name,
                            "baseline": baseline,
                            "source": source,
                            "metric": metric_name,
                            "value": float(value),
                        }
                    )
    return rows


def write_metric_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flattened metric rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "baseline", "source", "metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_run_summary(
    *,
    output_dir: Path,
    report_path: Path,
    metrics_csv_path: Path,
    metrics_json_path: Path,
    baselines: list[str],
    experiment_name: str,
) -> None:
    """Print a concise summary for completed experiment runs."""
    print(f"[medrap] finished experiment={experiment_name} baselines={','.join(baselines)}")
    print(f"[medrap] report={report_path}")
    print(f"[medrap] metrics_csv={metrics_csv_path} (experiment,baseline,source,metric,value)")
    print(f"[medrap] metrics_json={metrics_json_path}")
    print(f"[medrap] output_dir={output_dir}")
