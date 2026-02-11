"""
Deterministic scorer:
- automatic metric = triple F1 only
- trace is classified into template suggestions, then manually gated
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> bool:
        return False

from .io_utils import dump_data, infer_format, load_data
from .metrics import (
    compute_triple_f1,
    extract_tool_sequence,
    suggest_trace_template,
)
from .summary import summarize_results


def _build_versioned_output_path(path: Path, flavor: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}.{flavor}-{stamp}{path.suffix}")
    seq = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{flavor}-{stamp}-{seq}{path.suffix}")
        seq += 1
    return candidate


def _dump_with_fallback(data: dict[str, Any], path: Path, format_hint: str | None) -> Path:
    fmt = infer_format(path, format_hint)
    try:
        dump_data(data, path, fmt)
        return path
    except RuntimeError as exc:
        if fmt == "toml" and "tomlkit" in str(exc).lower():
            fallback = path.with_suffix(".json")
            print(
                f"Warning: TOML writer unavailable ({exc}). "
                f"Writing JSON fallback to {fallback} instead."
            )
            dump_data(data, fallback, "json")
            return fallback
        raise


def evaluate_single_item(item: dict[str, Any]) -> dict[str, Any]:
    expected = item.get("expected", {}) or {}
    received = item.get("received", {}) or {}

    expected_triples = expected.get("triples", []) or []
    received_triples = received.get("triples", []) or []
    received_trace = received.get("trace", []) or []
    qtype = str(item.get("qtype") or "unknown")

    triple = compute_triple_f1(expected_triples, received_triples)
    trace_suggestion = suggest_trace_template(qtype, received_trace)
    tool_sequence = extract_tool_sequence(received_trace)
    explanation_tool_called = _has_explain_tool_call(item.get("runtime_trace", []) or [])

    return {
        "id": item.get("id", "unknown"),
        "question": item.get("question", ""),
        "qtype": qtype,
        "triple_precision": triple.precision,
        "triple_recall": triple.recall,
        "triple_f1": triple.f1,
        "triples_matched": triple.matched,
        "triples_expected": triple.expected_count,
        "triples_received": triple.received_count,
        "combined_f1": triple.f1,
        "trace_valid": None,
        "trace_template_id": trace_suggestion.template_id,
        "trace_template_similarity": trace_suggestion.similarity,
        "trace_template_exact": trace_suggestion.exact_match,
        "trace_template_tools": trace_suggestion.template_tools,
        "trace_tool_sequence": tool_sequence,
        "explanation_tool_called": explanation_tool_called,
        "manual_review": {
            "status": "pending",
            "trace_valid": None,
            "comments": [],
        },
        "factored_in": True,
        "exclude_reason": None,
        "scoring_source": "auto",
        "error": item.get("error"),
    }


def _has_explain_tool_call(runtime_trace: list[Any]) -> bool:
    """True iff runtime trace contains an explicit explain tool call."""
    if not isinstance(runtime_trace, list) or not runtime_trace:
        return False
    for item in runtime_trace:
        if isinstance(item, dict):
            for tc in item.get("tool_calls") or []:
                if isinstance(tc, dict) and str(tc.get("name") or "").strip() == "explain":
                    return True
            if str(item.get("name") or "").strip() == "explain":
                return True
    return False


def evaluate_results(
    data: dict[str, Any],
    force_explanation_present: bool = False,
    exclude_impossible: bool = False,
) -> dict[str, Any]:
    metadata = data.get("metadata", {}) or {}
    output = data.get("output", []) or []
    per_question = [evaluate_single_item(item) for item in output]
    penalized_ids: list[Any] = []
    if force_explanation_present:
        for item, row in zip(output, per_question):
            runtime_trace = item.get("runtime_trace", []) or []
            explanation_present = _has_explain_tool_call(runtime_trace)
            row["explanation_present"] = explanation_present
            if not explanation_present:
                row["triple_f1"] = 0.0
                row["combined_f1"] = 0.0
                penalized_ids.append(row.get("id"))
    excluded_ids: list[Any] = []
    if exclude_impossible:
        for row in per_question:
            if str(row.get("qtype") or "").strip().lower() == "impossible":
                row["factored_in"] = False
                row["exclude_reason"] = "qtype_impossible"
                excluded_ids.append(row.get("id"))
    summary = summarize_results(
        per_question,
        {
            **metadata,
            "scoring_mode": "triple_only",
            "scoring_alpha": 0.0,
            "force_explanation_present": force_explanation_present,
            "explanation_missing_penalty_count": len(penalized_ids),
            "explanation_missing_penalty_ids": penalized_ids,
            "exclude_impossible": exclude_impossible,
            "excluded_impossible_count": len(excluded_ids),
            "excluded_impossible_ids": excluded_ids,
        },
    )
    return {
        **summary,
        "per_question": per_question,
    }


def print_report(evaluation: dict[str, Any], input_file: Path) -> None:
    metadata = evaluation.get("metadata", {}) or {}
    overall = (evaluation.get("summary", {}) or {}).get("overall", {}) or {}

    print()
    print("=" * 72)
    print("Deterministic Triple-Only Scoring")
    print("=" * 72)
    print(f"Input:             {input_file}")
    print(f"Model:             {metadata.get('model', 'unknown')}")
    print(f"Questions:         {metadata.get('total_questions', 0)}")
    print(f"Factored:          {metadata.get('factored_questions', 0)}")
    print(f"Excluded:          {metadata.get('excluded_questions', 0)}")
    print(f"Manual pending:    {metadata.get('manual_review_pending', 0)}")
    print(f"Triple P/R/F1:     {overall.get('triple_precision', {}).get('mean', 0.0):.4f} / "
          f"{overall.get('triple_recall', {}).get('mean', 0.0):.4f} / "
          f"{overall.get('triple_f1', {}).get('mean', 0.0):.4f}")
    print(f"Combined F1 mean:  {overall.get('combined_f1', {}).get('mean', 0.0):.4f}")
    print("=" * 72)
    print()


def _load_or_initialize_scores(
    eval_path: Path,
    output_path: Path,
    output_format_hint: str | None,
    auto_only: bool,
    force_explanation_present: bool,
    exclude_impossible: bool,
) -> tuple[dict[str, Any], Path]:
    eval_data = load_data(eval_path)

    if output_path.exists() and auto_only:
        fresh_path = _build_versioned_output_path(output_path, "fresh")
        print(f"Existing file preserved: {output_path}")
        print(f"Auto-scored output will be written to: {fresh_path}")
        evaluation = evaluate_results(
            eval_data,
            force_explanation_present=force_explanation_present,
            exclude_impossible=exclude_impossible,
        )
        written_path = _dump_with_fallback(evaluation, fresh_path, output_format_hint)
        return evaluation, written_path

    if output_path.exists() and not auto_only:
        backup_path = _build_versioned_output_path(output_path, "session-backup")
        try:
            shutil.copy2(output_path, backup_path)
            print(f"Session backup created: {backup_path}")
        except OSError as exc:
            print(f"Warning: failed to create session backup: {exc}", file=sys.stderr)
        existing = load_data(output_path)
        return existing, output_path

    evaluation = evaluate_results(
        eval_data,
        force_explanation_present=force_explanation_present,
        exclude_impossible=exclude_impossible,
    )
    written_path = _dump_with_fallback(evaluation, output_path, output_format_hint)
    return evaluation, written_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Deterministic triple-only scorer")
    parser.add_argument("input_file", type=Path, help="Path to eval results (.json/.toml)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file for scored results (default: <input>.scored.<ext>)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        help="Output format override: json or toml",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Run scoring only; do not launch manual review web UI",
    )
    parser.add_argument(
        "-x",
        "--exclude-impossible",
        action="store_true",
        help="Exclude qtype=impossible rows from aggregated metrics.",
    )
    parser.add_argument(
        "--review-all",
        action="store_true",
        help="In web UI, review all questions (default: pending only)",
    )
    parser.add_argument("--web-host", type=str, default="127.0.0.1", help="Host for manual review web UI")
    parser.add_argument("--web-port", type=int, default=5000, help="Port for manual review web UI")
    parser.add_argument(
        "--force_explanation_present",
        action="store_true",
        help="Set per-question F1 to 0 when received explanation trace is missing/empty.",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        suffix = args.input_file.suffix or ".toml"
        output_path = args.input_file.with_name(f"{args.input_file.stem}.scored{suffix}")

    scores, working_output = _load_or_initialize_scores(
        eval_path=args.input_file,
        output_path=output_path,
        output_format_hint=args.format,
        auto_only=args.auto_only,
        force_explanation_present=args.force_explanation_present,
        exclude_impossible=args.exclude_impossible,
    )
    print_report(scores, args.input_file)
    print(f"Results saved to: {working_output}")

    if args.auto_only:
        return

    try:
        from .manual_review_web import run_manual_review_web
    except ModuleNotFoundError as exc:
        if exc.name not in {"flask", "markdown"}:
            raise
        print(
            "Manual review web dependencies are missing (flask/markdown). Install deps and retry.",
            file=sys.stderr,
        )
        sys.exit(2)

    run_manual_review_web(
        scores_path=working_output,
        eval_path=args.input_file,
        output_path=working_output,
        format_hint=args.format,
        review_all=args.review_all,
        host=args.web_host,
        port=args.web_port,
    )


if __name__ == "__main__":
    main()
