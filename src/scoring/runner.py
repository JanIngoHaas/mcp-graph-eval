"""
Runner script for evaluating KG agent results.

Usage:
    python -m src.scoring.runner <eval_results_file.json> [--alpha 0.5] [--output results.json]
    python -m src.scoring.runner <eval_results_file.json> --output results.toml --format toml
    python -m src.scoring.runner <eval_results_file.toml> --auto-only
    python -m src.scoring.runner <eval_results_file.toml> --triple-only
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from .metrics import (
    compute_combined_score,
    CombinedScore,
    TraceScore,
    TripleScore,
    extract_cited_execution_keys,
)
from .anomalies import load_anomaly_config, detect_anomalies, AnomalyConfig
from .io_utils import load_data, dump_data, infer_format
from .summary import summarize_results
from .ambiguity import (
    AmbiguityResolver,
    AmbiguityExpansion,
    build_effective_ambiguity_trace,
)


def _build_versioned_output_path(path: Path, flavor: str) -> Path:
    """Return a non-existing sibling path with timestamp suffix."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}.{flavor}-{stamp}{path.suffix}")
    seq = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{flavor}-{stamp}-{seq}{path.suffix}")
        seq += 1
    return candidate


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _extract_explain_found(received_trace: list) -> bool | None:
    """Return found flag from explain object if present (fallback: legacy success)."""
    for item in received_trace or []:
        if not isinstance(item, dict):
            continue
        if "found" in item:
            return _coerce_bool(item.get("found"))
        if "success" in item:
            return _coerce_bool(item.get("success"))
    return None


def evaluate_single_item(
    item: dict,
    alpha: float,
    anomaly_config: AnomalyConfig,
    ambiguity_resolver: AmbiguityResolver | None = None,
) -> dict:
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
    scoring_qtype = qtype

    ambiguity_expansion: AmbiguityExpansion | None = None
    if ambiguity_resolver is not None:
        ambiguity_expansion = ambiguity_resolver.maybe_expand(item)
        if ambiguity_expansion is not None:
            expected_triples = ambiguity_expansion.expected_triples
            if ambiguity_expansion.qtype_override:
                scoring_qtype = ambiguity_expansion.qtype_override
    ignored_execution_keys = extract_cited_execution_keys(item.get("runtime_trace", []))
    if ambiguity_expansion is not None:
        expected_trace = build_effective_ambiguity_trace(
            original_expected_trace=expected_trace,
            received_trace=received_trace,
            ignored_execution_keys=ignored_execution_keys,
            ambiguity_expansion=ambiguity_expansion,
        )
    score = compute_combined_score(
        expected_trace=expected_trace,
        received_trace=received_trace,
        expected_triples=expected_triples,
        received_triples=received_triples,
        alpha=alpha,
        ignored_execution_keys=ignored_execution_keys,
    )

    # Special handling for impossible questions:
    # - explain must set found=false
    # - citations are optional and ignored for correctness
    if scoring_qtype == "impossible":
        found_flag = _extract_explain_found(received_trace)
        if found_flag is not False:
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
    
    anomaly_flags = detect_anomalies(
        trace_f1=score.trace_score.f1,
        triple_f1=score.triple_score.f1,
        triple_precision=score.triple_score.precision,
        triple_recall=score.triple_score.recall,
        qtype=scoring_qtype,
        config=anomaly_config,
        expected_triples=expected_triples,
        received_triples=received_triples,
        explain_found=_extract_explain_found(received_trace),
    )
    anomaly_detected = bool(anomaly_flags)
    exclude_reason = ",".join(anomaly_flags) if anomaly_flags else None

    return {
        'id': question_id,
        'question': question,
        'qtype': qtype,
        'scoring_qtype': scoring_qtype,
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
        'auto_trace_precision': score.trace_score.precision,
        'auto_trace_recall': score.trace_score.recall,
        'auto_trace_f1': score.trace_score.f1,
        'auto_triple_precision': score.triple_score.precision,
        'auto_triple_recall': score.triple_score.recall,
        'auto_triple_f1': score.triple_score.f1,
        'auto_combined_f1': score.combined_f1,
        'anomaly_flags': anomaly_flags,
        'anomaly_detected': anomaly_detected,
        'manual_review': {
            'required': anomaly_detected,
            'status': 'pending' if anomaly_detected else 'none',
        },
        'factored_in': not anomaly_detected,
        'exclude_reason': exclude_reason if anomaly_detected else None,
        'scoring_source': 'excluded' if anomaly_detected else 'auto',
        'ambiguity_expanded': ambiguity_expansion is not None,
        'ambiguity_reason': ambiguity_expansion.reason if ambiguity_expansion else None,
        'ambiguity_label': ambiguity_expansion.label if ambiguity_expansion else None,
        'ambiguity_candidate_count': ambiguity_expansion.candidate_count if ambiguity_expansion else None,
        'error': item.get('error'),
    }


def evaluate_results(
    data: dict,
    alpha: float,
    anomaly_config: AnomalyConfig,
    include_anomalies: bool,
    scoring_mode: str = "combined",
    ambiguity_resolver: AmbiguityResolver | None = None,
) -> dict:
    """Evaluate all results in an eval file."""
    metadata = data.get('metadata', {})
    output = data.get('output', [])
    
    results = []
    for item in output:
        result = evaluate_single_item(item, alpha, anomaly_config, ambiguity_resolver=ambiguity_resolver)
        results.append(result)
    summary = summarize_results(
        results,
        {**metadata, 'scoring_alpha': alpha, 'scoring_mode': scoring_mode},
        include_anomalies,
        compute_pvalues=True,
    )
    return {
        **summary,
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
    print(f"  Factored:       {metadata.get('factored_questions', 0)}")
    print(f"  Excluded:       {metadata.get('excluded_questions', 0)}")
    print(f"  Manual Pending: {metadata.get('manual_review_pending', 0)}")
    print(f"  Scoring Mode:   {metadata.get('scoring_mode', 'combined')}")
    print(f"  Alpha (weight): {metadata.get('scoring_alpha', 0.5)}")
    print()
    
    # Per-question breakdown
    print("PER-QUESTION BREAKDOWN")
    print("-" * 80)
    print()
    
    headers = ["ID", "Type", "Trace", "Triple", "Combined", "Status", "Question"]
    col_widths = [4, 14, 8, 8, 10, 38, 32]
    
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    
    print(sep)
    header_row = "|" + "|".join(str(h).center(col_widths[i]) for i, h in enumerate(headers)) + "|"
    print(header_row)
    print(sep)
    
    for q in per_question:
        question = q['question']
        if len(question) > 30:
            question = question[:27] + "..."
        
        if not q.get('factored_in', True):
            reason = q.get('exclude_reason') or 'unknown'
            status = f"excluded (reasons: {reason})"
        else:
            status = q.get('scoring_source', 'auto')
        row = [
            str(q['id']),
            q['qtype'],
            f"{q['trace_f1']:.3f}",
            f"{q['triple_f1']:.3f}",
            f"{q['combined_f1']:.3f}",
            status,
            question
        ]
        row_str = "|" + "|".join(
            str(row[i]).center(col_widths[i]) if i < 6 else " " + str(row[i]).ljust(col_widths[i]-1)
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
    load_dotenv()
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
        '--triple-only',
        action='store_true',
        help='Compute combined score from triples only (trace weight forced to 0).'
    )
    parser.add_argument(
        '--anomaly-config',
        type=Path,
        default=None,
        help='Path to anomaly config (TOML or JSON)'
    )
    parser.add_argument(
        '--include-anomalies',
        action='store_true',
        help='Include anomaly-flagged questions in summary stats'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output file for results (default: <input>.scored.<ext>)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default=None,
        help='Output format: json or toml (defaults to inferred from output path)'
    )
    parser.add_argument(
        '--auto-only',
        action='store_true',
        help='Skip manual review UI; auto scoring only'
    )
    parser.add_argument(
        '--anomalies-only',
        action='store_true',
        help='Manual review UI shows anomalies only'
    )
    parser.add_argument(
        '--web-host',
        type=str,
        default='127.0.0.1',
        help='Host for manual review web UI'
    )
    parser.add_argument(
        '--web-port',
        type=int,
        default=5000,
        help='Port for manual review web UI'
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        print("Error: --alpha must be within [0, 1].", file=sys.stderr)
        sys.exit(1)

    effective_alpha = 0.0 if args.triple_only else args.alpha
    scoring_mode = "triple_only" if args.triple_only else "combined"
    if args.triple_only and args.alpha != 0.5:
        print("Note: --triple-only enabled; ignoring --alpha and forcing alpha=0.0.")
    
    # Load input
    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    data = load_data(args.input_file)
    
    anomaly_path = args.anomaly_config
    if anomaly_path is None:
        default_path = Path("config/anomaly.toml")
        if default_path.exists():
            anomaly_path = default_path
    anomaly_config = load_anomaly_config(anomaly_path)
    sparql_endpoint = (os.getenv("SPARQL_ENDPOINT") or "").strip() or None
    if not sparql_endpoint:
        print(
            "Error: ambiguity resolution is mandatory, but SPARQL_ENDPOINT is not set in .env.",
            file=sys.stderr,
        )
        sys.exit(2)
    ambiguity_resolver = AmbiguityResolver(sparql_endpoint=sparql_endpoint)
    print(f"Ambiguity expansion enabled with KG: {ambiguity_resolver.source_description}")

    output_path = args.output
    if output_path is None:
        suffix = args.input_file.suffix or ".toml"
        output_path = args.input_file.with_name(f"{args.input_file.stem}.scored{suffix}")

    # Never overwrite existing output files with automatic scoring.
    # Manual review sessions use the existing scored file directly, but make
    # a backup copy at session start.
    working_output = output_path
    if output_path.exists() and args.auto_only:
        working_output = _build_versioned_output_path(output_path, "fresh")
        print(f"Existing file preserved: {output_path}")
        print(f"Auto-scored output will be written to: {working_output}")
        evaluation = evaluate_results(
            data,
            effective_alpha,
            anomaly_config,
            args.include_anomalies,
            scoring_mode=scoring_mode,
            ambiguity_resolver=ambiguity_resolver,
        )
        out_format = infer_format(working_output, args.format)
        dump_data(evaluation, working_output, out_format)
        print(f"Results saved to: {working_output}")
    elif output_path.exists() and not args.auto_only:
        backup_path = _build_versioned_output_path(output_path, "session-backup")
        try:
            shutil.copy2(output_path, backup_path)
            print(f"Session backup created: {backup_path}")
        except OSError as exc:
            print(f"Warning: failed to create session backup: {exc}", file=sys.stderr)
        if args.triple_only:
            print(
                "Note: --triple-only applies to auto-scoring only. "
                "Existing output is reused for manual review without recomputation.",
                file=sys.stderr,
            )
        print(f"Using existing scored file for manual review: {output_path}")
        working_output = output_path
    else:
        evaluation = evaluate_results(
            data,
            effective_alpha,
            anomaly_config,
            args.include_anomalies,
            scoring_mode=scoring_mode,
            ambiguity_resolver=ambiguity_resolver,
        )
        out_format = infer_format(working_output, args.format)
        dump_data(evaluation, working_output, out_format)
        print(f"Results saved to: {working_output}")

    if not args.auto_only:
        try:
            from .manual_review_web import run_manual_review_web
        except ModuleNotFoundError as exc:
            if exc.name not in {"flask", "markdown"}:
                raise
            print(
                "Manual review web dependencies are missing (flask/markdown). "
                "Install dependencies and retry.",
                file=sys.stderr,
            )
            sys.exit(2)

        run_manual_review_web(
            scores_path=working_output,
            eval_path=args.input_file,
            output_path=working_output,
            format_hint=args.format,
            anomalies_only=args.anomalies_only,
            host=args.web_host,
            port=args.web_port,
        )


if __name__ == '__main__':
    main()
