"""Summary aggregation for deterministic triple-first scoring."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np


def compute_group_stats(results: list[dict], metric_key: str) -> dict[str, Any]:
    values = [float(r.get(metric_key, 0.0)) for r in results]
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def summarize_results(results: list[dict], metadata: dict[str, Any]) -> dict[str, Any]:
    factored_rows = [row for row in results if bool(row.get("factored_in", True))]

    by_type_rows = defaultdict(list)
    for row in factored_rows:
        by_type_rows[str(row.get("qtype") or "unknown")].append(row)

    by_type = {}
    for qtype, rows in by_type_rows.items():
        trace_reviewed = sum(1 for r in rows if r.get("trace_valid") is not None)
        trace_valid_yes = sum(1 for r in rows if r.get("trace_valid") is True)
        trace_valid_no = sum(1 for r in rows if r.get("trace_valid") is False)
        by_type[qtype] = {
            "count": len(rows),
            "triple_f1": compute_group_stats(rows, "triple_f1"),
            "combined_f1": compute_group_stats(rows, "combined_f1"),
            "trace_reviewed_count": trace_reviewed,
            "trace_reviewed_rate": (trace_reviewed / len(rows)) if rows else 0.0,
            "trace_valid_yes_count": trace_valid_yes,
            "trace_valid_no_count": trace_valid_no,
            "trace_template_exact_rate": (
                sum(1 for r in rows if r.get("trace_template_exact")) / len(rows)
            ) if rows else 0.0,
        }

    overall = {
        "triple_precision": compute_group_stats(factored_rows, "triple_precision"),
        "triple_recall": compute_group_stats(factored_rows, "triple_recall"),
        "triple_f1": compute_group_stats(factored_rows, "triple_f1"),
        "combined_f1": compute_group_stats(factored_rows, "combined_f1"),
        "trace_reviewed_count": sum(1 for r in factored_rows if r.get("trace_valid") is not None),
        "trace_valid_yes_count": sum(1 for r in factored_rows if r.get("trace_valid") is True),
        "trace_valid_no_count": sum(1 for r in factored_rows if r.get("trace_valid") is False),
        "trace_template_exact_rate": (
            sum(1 for r in factored_rows if r.get("trace_template_exact")) / len(factored_rows)
        ) if factored_rows else 0.0,
    }

    total = len(results)
    factored_total = len(factored_rows)
    overall["trace_reviewed_rate"] = (
        overall["trace_reviewed_count"] / factored_total
    ) if factored_total else 0.0

    return {
        "metadata": {
            **metadata,
            "total_questions": total,
            "factored_questions": factored_total,
            "excluded_questions": total - factored_total,
            "manual_review_pending": sum(
                1
                for r in factored_rows
                if (r.get("manual_review", {}) or {}).get("status") == "pending"
                or r.get("trace_valid") is None
            ),
        },
        "summary": {
            "overall": overall,
            "by_type": by_type,
        },
    }
