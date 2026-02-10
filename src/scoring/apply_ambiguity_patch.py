"""
Re-apply ambiguity-aware auto scoring to an existing scored file.

This updates auto/anomaly/ambiguity fields from a fresh ambiguity-enabled pass,
while preserving manual reviewer decisions.

Usage:
  python -m src.scoring.apply_ambiguity_patch \
    --eval results/hypmol/eval_results_gemini-3-flash-preview_cloud.toml \
    --scored results/hypmol/eval_results_gemini-3-flash-preview_cloud.scored.toml
"""

from __future__ import annotations

import argparse
import os
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

from .ambiguity import AmbiguityResolver
from .anomalies import load_anomaly_config
from .io_utils import dump_data, infer_format, load_data
from .runner import evaluate_results
from .summary import summarize_results


AUTO_SCORE_FIELDS = [
    "auto_trace_precision",
    "auto_trace_recall",
    "auto_trace_f1",
    "auto_triple_precision",
    "auto_triple_recall",
    "auto_triple_f1",
    "auto_combined_f1",
]

FINAL_SCORE_FIELDS = [
    "trace_precision",
    "trace_recall",
    "trace_f1",
    "triple_precision",
    "triple_recall",
    "triple_f1",
    "combined_f1",
    "factored_in",
    "exclude_reason",
    "scoring_source",
]

REFRESH_FIELDS = [
    "qtype",
    "scoring_qtype",
    "triples_matched",
    "triples_expected",
    "triples_received",
    "anomaly_flags",
    "anomaly_detected",
    "ambiguity_expanded",
    "ambiguity_reason",
    "ambiguity_label",
    "ambiguity_candidate_count",
    "error",
]

LOCKED_SCORING_SOURCES = {"manual", "manual_perfect"}
LOCKED_MANUAL_STATUSES = {"scored", "accepted", "skipped"}


def _build_versioned_output_path(path: Path, flavor: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}.{flavor}-{stamp}{path.suffix}")
    seq = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{flavor}-{stamp}-{seq}{path.suffix}")
        seq += 1
    return candidate


def _id_key(value: object) -> str:
    return str(value)


def _infer_eval_path_from_scored(scored_path: Path) -> Path:
    stem = scored_path.stem
    if stem.endswith(".scored"):
        base = stem[: -len(".scored")]
        return scored_path.with_name(f"{base}{scored_path.suffix}")
    raise ValueError(
        "Could not infer eval path from scored file name. "
        "Pass --eval explicitly."
    )


def _is_final_locked(entry: dict[str, Any]) -> bool:
    source = str(entry.get("scoring_source") or "")
    status = str(entry.get("manual_review", {}).get("status") or "")
    return source in LOCKED_SCORING_SOURCES or status in LOCKED_MANUAL_STATUSES


def _merge_entry(existing: dict[str, Any], fresh: dict[str, Any]) -> tuple[bool, bool]:
    """
    Merge one per-question entry.

    Returns:
      (final_scores_locked, ambiguity_expanded)
    """
    final_locked = _is_final_locked(existing)

    for field in AUTO_SCORE_FIELDS:
        existing[field] = fresh.get(field)

    for field in REFRESH_FIELDS:
        existing[field] = fresh.get(field)

    existing_manual = existing.setdefault("manual_review", {})
    fresh_manual = fresh.get("manual_review", {})
    if not final_locked:
        existing_manual["required"] = bool(fresh_manual.get("required", fresh.get("anomaly_detected", False)))
        current_status = str(existing_manual.get("status") or "")
        if current_status not in LOCKED_MANUAL_STATUSES:
            existing_manual["status"] = str(fresh_manual.get("status", "none"))

    if not final_locked:
        for field in FINAL_SCORE_FIELDS:
            existing[field] = fresh.get(field)

    return final_locked, bool(fresh.get("ambiguity_expanded"))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Re-apply ambiguity-aware auto-scoring onto an existing scored file "
            "while preserving manual reviewer decisions."
        )
    )
    parser.add_argument(
        "--scored",
        type=Path,
        required=True,
        help="Path to existing scored file (.toml/.json).",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Path to raw eval file. Defaults to inferred <name>.toml from <name>.scored.toml.",
    )
    parser.add_argument(
        "--anomaly-config",
        type=Path,
        default=None,
        help="Path to anomaly config (TOML/JSON). Defaults to config/anomaly.toml when present.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic timestamped backup of the scored file before writing.",
    )
    args = parser.parse_args()

    scored_path = args.scored
    if not scored_path.exists():
        print(f"Error: scored file not found: {scored_path}", file=sys.stderr)
        sys.exit(1)

    eval_path = args.eval
    if eval_path is None:
        try:
            eval_path = _infer_eval_path_from_scored(scored_path)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
    if not eval_path.exists():
        print(f"Error: eval file not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    sparql_endpoint = (os.getenv("SPARQL_ENDPOINT") or "").strip() or None
    if not sparql_endpoint:
        print("Error: SPARQL_ENDPOINT is required.", file=sys.stderr)
        sys.exit(2)

    anomaly_path = args.anomaly_config
    if anomaly_path is None:
        default_path = Path("config/anomaly.toml")
        if default_path.exists():
            anomaly_path = default_path
    anomaly_config = load_anomaly_config(anomaly_path)
    ambiguity_resolver = AmbiguityResolver(sparql_endpoint=sparql_endpoint)

    scored_data = load_data(scored_path)
    eval_data = load_data(eval_path)

    metadata = scored_data.get("metadata", {})
    alpha = float(metadata.get("scoring_alpha", 0.5))
    scoring_mode = str(metadata.get("scoring_mode", "combined")).strip().lower()
    effective_alpha = 0.0 if scoring_mode == "triple_only" else alpha

    fresh = evaluate_results(
        eval_data,
        effective_alpha,
        anomaly_config,
        include_anomalies=False,
        scoring_mode=scoring_mode,
        ambiguity_resolver=ambiguity_resolver,
    )
    fresh_by_id = {_id_key(item.get("id")): item for item in fresh.get("per_question", [])}

    per_question = scored_data.get("per_question")
    if not isinstance(per_question, list):
        print("Error: scored file has no valid per_question list.", file=sys.stderr)
        sys.exit(3)

    merged = 0
    missing = 0
    locked = 0
    ambiguity_expanded = 0

    for existing in per_question:
        if not isinstance(existing, dict):
            continue
        key = _id_key(existing.get("id"))
        fresh_entry = fresh_by_id.get(key)
        if fresh_entry is None:
            missing += 1
            continue
        is_locked, was_ambiguity_expanded = _merge_entry(existing, fresh_entry)
        merged += 1
        if is_locked:
            locked += 1
        if was_ambiguity_expanded:
            ambiguity_expanded += 1

    summary = summarize_results(
        per_question,
        metadata,
        include_anomalies=False,
        compute_pvalues=False,
    )
    scored_data["metadata"] = summary["metadata"]
    scored_data["summary"] = summary["summary"]
    scored_data["per_question"] = per_question

    if not args.no_backup:
        backup_path = _build_versioned_output_path(scored_path, "ambiguity-backup")
        shutil.copy2(scored_path, backup_path)
        print(f"Backup created: {backup_path}")

    fmt = infer_format(scored_path, None)
    dump_data(scored_data, scored_path, fmt)

    print(f"Patched scored file: {scored_path}")
    print(f"Eval source: {eval_path}")
    print(f"Merged entries: {merged}")
    print(f"Locked entries preserved: {locked}")
    print(f"Ambiguity-expanded entries: {ambiguity_expanded}")
    if missing:
        print(f"Warning: entries missing in fresh evaluation: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
