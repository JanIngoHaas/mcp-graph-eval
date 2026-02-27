"""Summary aggregation for deterministic triple-grounding scoring."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import statistics


def compute_group_stats(results: list[dict], metric_key: str) -> dict[str, Any]:
    values = [float(r.get(metric_key, 0.0)) for r in results]
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def summarize_results(results: list[dict], metadata: dict[str, Any]) -> dict[str, Any]:
    factored_rows = [row for row in results if bool(row.get("factored_in", True))]

    by_type_rows = defaultdict(list)
    for row in factored_rows:
        by_type_rows[str(row.get("qtype") or "unknown")].append(row)

    by_type = {}
    for qtype, rows in by_type_rows.items():
        by_type[qtype] = {
            "count": len(rows),
            "triple_precision": compute_group_stats(rows, "triple_precision"),
            "triple_recall": compute_group_stats(rows, "triple_recall"),
            "triple_f1": compute_group_stats(rows, "triple_f1"),
        }

    overall = {
        "triple_precision": compute_group_stats(factored_rows, "triple_precision"),
        "triple_recall": compute_group_stats(factored_rows, "triple_recall"),
        "triple_f1": compute_group_stats(factored_rows, "triple_f1"),
    }

    total = len(results)
    factored_total = len(factored_rows)

    return {
        "metadata": {
            **metadata,
            "total_questions": total,
            "factored_questions": factored_total,
            "excluded_questions": total - factored_total,
        },
        "summary": {
            "overall": overall,
            "by_type": by_type,
        },
    }
