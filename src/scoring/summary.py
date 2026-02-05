"""Summary aggregation for scoring results."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
import numpy as np


def compute_group_stats(results: list[dict], metric_key: str) -> dict[str, Any]:
    """Compute mean, std, and values for a metric."""
    values = [r[metric_key] for r in results]
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'n': 0, 'values': []}
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'n': len(values),
        'values': values
    }


def _filter_factored_results(results: list[dict], include_anomalies: bool) -> list[dict]:
    if include_anomalies:
        return results
    return [r for r in results if r.get('factored_in', True)]


def summarize_results(
    results: list[dict],
    metadata: dict[str, Any],
    include_anomalies: bool,
    compute_pvalues: bool = False,
) -> dict[str, Any]:
    filtered = _filter_factored_results(results, include_anomalies)

    # Group by question type
    by_type = defaultdict(list)
    for r in filtered:
        by_type[r['qtype']].append(r)

    # Compute per-type stats
    type_stats = {}
    for qtype, type_results in by_type.items():
        type_stats[qtype] = {
            'count': len(type_results),
            'trace_f1': compute_group_stats(type_results, 'trace_f1'),
            'triple_f1': compute_group_stats(type_results, 'triple_f1'),
            'combined_f1': compute_group_stats(type_results, 'combined_f1'),
        }

    # Compute overall stats
    overall_stats = {
        'trace_f1': compute_group_stats(filtered, 'trace_f1'),
        'trace_precision': compute_group_stats(filtered, 'trace_precision'),
        'trace_recall': compute_group_stats(filtered, 'trace_recall'),
        'triple_f1': compute_group_stats(filtered, 'triple_f1'),
        'triple_precision': compute_group_stats(filtered, 'triple_precision'),
        'triple_recall': compute_group_stats(filtered, 'triple_recall'),
        'combined_f1': compute_group_stats(filtered, 'combined_f1'),
    }

    # Compute p-values between question types (optional)
    pvalues: dict[str, float] = {}
    if compute_pvalues and len(by_type) == 2:
        try:
            from scipy import stats
        except Exception:
            stats = None
        if stats is not None:
            qtypes = list(by_type.keys())
            t1, t2 = qtypes
            for metric in ['trace_f1', 'triple_f1', 'combined_f1']:
                vals1 = type_stats[t1][metric]['values']
                vals2 = type_stats[t2][metric]['values']
                if len(vals1) >= 2 and len(vals2) >= 2:
                    try:
                        _, pvalue = stats.ttest_ind(vals1, vals2, equal_var=False)
                        pvalues[f'{metric}_{t1}_vs_{t2}'] = float(pvalue)
                    except Exception:
                        pvalues[f'{metric}_{t1}_vs_{t2}'] = float('nan')
                else:
                    pvalues[f'{metric}_{t1}_vs_{t2}'] = float('nan')

    # Remove raw values from output (keep stats only)
    for qtype in type_stats:
        for metric in ['trace_f1', 'triple_f1', 'combined_f1']:
            del type_stats[qtype][metric]['values']
    for metric in overall_stats:
        del overall_stats[metric]['values']

    anomaly_tally = Counter()
    for r in results:
        for flag in r.get('anomaly_flags', []):
            anomaly_tally[flag] += 1

    excluded_count = sum(1 for r in results if not r.get('factored_in', True))
    manual_pending = sum(1 for r in results if r.get('manual_review', {}).get('status') == 'pending')

    return {
        'metadata': {
            **metadata,
            'total_questions': len(results),
            'factored_questions': len(filtered),
            'excluded_questions': excluded_count,
            'manual_review_pending': manual_pending,
        },
        'summary': {
            'overall': overall_stats,
            'by_type': type_stats,
            'pvalues': pvalues,
            'anomalies': dict(anomaly_tally),
        },
    }
