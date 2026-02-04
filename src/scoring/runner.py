"""
Runner script for evaluating KG agent results.

Usage:
    python -m src.scoring.runner <eval_results_file.json> [--alpha 0.5] [--output results.json]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy import stats

from .metrics import (
    compute_combined_score,
    CombinedScore,
    TraceScore,
    TripleScore,
)


def _extract_explain_success(received_trace: list) -> bool | None:
    """Return success flag from explain object if present."""
    for item in received_trace or []:
        if isinstance(item, dict) and "success" in item:
            return item.get("success")
    return None


def evaluate_single_item(item: dict, alpha: float) -> dict:
    """Evaluate a single question/answer pair."""
    question_id = item.get('id', 'unknown')
    question = item.get('question', '')
    
    expected = item.get('expected', {})
    received = item.get('received', {})
    
    expected_trace = expected.get('trace', [])
    received_trace = received.get('trace', [])
    expected_triples = expected.get('triples', [])
    received_triples = received.get('triples', [])
    
    # Use qtype from item; fall back to 'unknown' if missing
    qtype = item.get('qtype') or 'unknown'
    
    score = compute_combined_score(
        expected_trace=expected_trace,
        received_trace=received_trace,
        expected_triples=expected_triples,
        received_triples=received_triples,
        alpha=alpha
    )

    # Special handling for impossible questions:
    # - explain must set success=false
    # - citations are optional and ignored for correctness
    if qtype == "impossible":
        success_flag = _extract_explain_success(received_trace)
        if success_flag is not False:
            score = CombinedScore(
                trace_score=TraceScore(precision=0.0, recall=0.0, f1=0.0, step_details=[]),
                triple_score=TripleScore(
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    matched=0,
                    expected_count=0,
                    received_count=len(received_triples),
                ),
                combined_f1=0.0,
                alpha=alpha,
            )
        else:
            score = CombinedScore(
                trace_score=score.trace_score,
                triple_score=TripleScore(
                    precision=1.0,
                    recall=1.0,
                    f1=1.0,
                    matched=0,
                    expected_count=0,
                    received_count=len(received_triples),
                ),
                combined_f1=alpha * score.trace_score.f1 + (1 - alpha) * 1.0,
                alpha=alpha,
            )
    
    return {
        'id': question_id,
        'question': question,
        'qtype': qtype,
        'trace_f1': score.trace_score.f1,
        'trace_precision': score.trace_score.precision,
        'trace_recall': score.trace_score.recall,
        'triple_f1': score.triple_score.f1,
        'triple_precision': score.triple_score.precision,
        'triple_recall': score.triple_score.recall,
        'triples_matched': score.triple_score.matched,
        'triples_expected': score.triple_score.expected_count,
        'triples_received': score.triple_score.received_count,
        'combined_f1': score.combined_f1,
        'error': item.get('error'),
    }


def compute_group_stats(results: list[dict], metric_key: str) -> dict:
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


def compute_ttest_pvalue(group1_values: list, group2_values: list) -> float:
    """Compute two-sample t-test p-value."""
    if len(group1_values) < 2 or len(group2_values) < 2:
        return float('nan')
    try:
        _, pvalue = stats.ttest_ind(group1_values, group2_values, equal_var=False)
        return float(pvalue)
    except Exception:
        return float('nan')


def evaluate_results(data: dict, alpha: float) -> dict:
    """Evaluate all results in an eval file."""
    metadata = data.get('metadata', {})
    output = data.get('output', [])
    
    results = []
    for item in output:
        result = evaluate_single_item(item, alpha)
        results.append(result)
    
    # Group by question type
    by_type = defaultdict(list)
    for r in results:
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
        'trace_f1': compute_group_stats(results, 'trace_f1'),
        'trace_precision': compute_group_stats(results, 'trace_precision'),
        'trace_recall': compute_group_stats(results, 'trace_recall'),
        'triple_f1': compute_group_stats(results, 'triple_f1'),
        'triple_precision': compute_group_stats(results, 'triple_precision'),
        'triple_recall': compute_group_stats(results, 'triple_recall'),
        'combined_f1': compute_group_stats(results, 'combined_f1'),
    }
    
    # Compute p-values between question types (if we have two types)
    pvalues = {}
    qtypes = list(by_type.keys())
    if len(qtypes) == 2:
        t1, t2 = qtypes
        for metric in ['trace_f1', 'triple_f1', 'combined_f1']:
            vals1 = type_stats[t1][metric]['values']
            vals2 = type_stats[t2][metric]['values']
            pvalues[f'{metric}_{t1}_vs_{t2}'] = compute_ttest_pvalue(vals1, vals2)
    
    # Remove raw values from output (keep stats only)
    for qtype in type_stats:
        for metric in ['trace_f1', 'triple_f1', 'combined_f1']:
            del type_stats[qtype][metric]['values']
    for metric in overall_stats:
        del overall_stats[metric]['values']
    
    return {
        'metadata': {
            **metadata,
            'scoring_alpha': alpha,
            'total_questions': len(results),
        },
        'summary': {
            'overall': overall_stats,
            'by_type': type_stats,
            'pvalues': pvalues,
        },
        'per_question': results,
    }


def print_table(headers: list[str], rows: list[list], col_widths: list[int] = None):
    """Print a simple ASCII table."""
    if not col_widths:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                      for i in range(len(headers))]
    
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    
    print(sep)
    header_row = "|" + "|".join(str(h).center(col_widths[i]) for i, h in enumerate(headers)) + "|"
    print(header_row)
    print(sep)
    
    for row in rows:
        row_str = "|" + "|".join(str(cell).center(col_widths[i]) for i, cell in enumerate(row)) + "|"
        print(row_str)
    
    print(sep)


def print_full_report(evaluation: dict, input_file: Path):
    """Print a comprehensive CLI report."""
    metadata = evaluation['metadata']
    summary = evaluation['summary']
    per_question = evaluation['per_question']
    overall = summary['overall']
    by_type = summary['by_type']
    pvalues = summary.get('pvalues', {})
    
    # Header
    print()
    print("=" * 80)
    print("  KG AGENT EVALUATION REPORT")
    print("=" * 80)
    print()
    
    # Metadata
    print("METADATA")
    print("-" * 80)
    print(f"  Input File:     {input_file.name}")
    print(f"  Model:          {metadata.get('model', 'unknown')}")
    print(f"  Created:        {metadata.get('created_at', 'unknown')}")
    print(f"  Questions:      {metadata.get('total_questions', 0)}")
    print(f"  Alpha (weight): {metadata.get('scoring_alpha', 0.5)}")
    print()
    
    # Per-question breakdown
    print("PER-QUESTION BREAKDOWN")
    print("-" * 80)
    print()
    
    headers = ["ID", "Type", "Trace", "Triple", "Combined", "Question"]
    col_widths = [4, 14, 8, 8, 10, 32]
    
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    
    print(sep)
    header_row = "|" + "|".join(str(h).center(col_widths[i]) for i, h in enumerate(headers)) + "|"
    print(header_row)
    print(sep)
    
    for q in per_question:
        question = q['question']
        if len(question) > 30:
            question = question[:27] + "..."
        
        row = [
            str(q['id']),
            q['qtype'],
            f"{q['trace_f1']:.3f}",
            f"{q['triple_f1']:.3f}",
            f"{q['combined_f1']:.3f}",
            question
        ]
        row_str = "|" + "|".join(
            str(row[i]).center(col_widths[i]) if i < 5 else " " + str(row[i]).ljust(col_widths[i]-1) 
            for i in range(len(row))
        ) + "|"
        print(row_str)
    
    print(sep)
    print()
    
    # Results by question type
    print("RESULTS BY QUESTION TYPE")
    print("-" * 80)
    print()
    
    type_headers = ["Type", "N", "Trace F1", "Triple F1", "Combined F1"]
    type_rows = []
    for qtype, stats in by_type.items():
        type_rows.append([
            qtype,
            str(stats['count']),
            f"{stats['trace_f1']['mean']:.4f} (+/- {stats['trace_f1']['std']:.4f})",
            f"{stats['triple_f1']['mean']:.4f} (+/- {stats['triple_f1']['std']:.4f})",
            f"{stats['combined_f1']['mean']:.4f} (+/- {stats['combined_f1']['std']:.4f})",
        ])
    
    print_table(type_headers, type_rows, col_widths=[14, 4, 22, 22, 22])
    print()
    
    # P-values (if available)
    if pvalues:
        print("  P-VALUES (two-sample t-test, unequal variance)")
        for key, pval in pvalues.items():
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"    {key}: {pval:.6f} {sig}")
        print()
    
    # Overall aggregate scores
    print("OVERALL AGGREGATE SCORES")
    print("-" * 80)
    print()
    
    print("  Trace Evaluation (order-agnostic, required/optional matching)")
    print_table(
        headers=["Metric", "Mean", "Std Dev"],
        rows=[
            ["Precision", f"{overall['trace_precision']['mean']:.4f}", f"{overall['trace_precision']['std']:.4f}"],
            ["Recall", f"{overall['trace_recall']['mean']:.4f}", f"{overall['trace_recall']['std']:.4f}"],
            ["F1 Score", f"{overall['trace_f1']['mean']:.4f}", f"{overall['trace_f1']['std']:.4f}"],
        ],
        col_widths=[20, 12, 12]
    )
    print()
    
    print("  Triple/Citation Evaluation (exact match)")
    print_table(
        headers=["Metric", "Mean", "Std Dev"],
        rows=[
            ["Precision", f"{overall['triple_precision']['mean']:.4f}", f"{overall['triple_precision']['std']:.4f}"],
            ["Recall", f"{overall['triple_recall']['mean']:.4f}", f"{overall['triple_recall']['std']:.4f}"],
            ["F1 Score", f"{overall['triple_f1']['mean']:.4f}", f"{overall['triple_f1']['std']:.4f}"],
        ],
        col_widths=[20, 12, 12]
    )
    print()
    
    print(f"  Combined F1 Score: {overall['combined_f1']['mean']:.4f} (+/- {overall['combined_f1']['std']:.4f})")
    print()
    
    # Footer
    print("-" * 80)
    print(f"  Generated: {datetime.now().isoformat()}")
    print("-" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate KG agent results with F1 scoring'
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to eval results JSON file'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Weight for trace score (0-1). Default: 0.5 (equal weight)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output file for JSON results (if not specified, prints to CLI)'
    )
    
    args = parser.parse_args()
    
    # Load input
    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Evaluate
    evaluation = evaluate_results(data, args.alpha)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        print_full_report(evaluation, args.input_file)


if __name__ == '__main__':
    main()
