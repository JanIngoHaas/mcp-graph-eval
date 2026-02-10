"""Minimal Flask UI for manual trace-valid gating."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, url_for

from .io_utils import dump_data, infer_format, load_data
from .metrics import _extract_expected_params, _extract_received_steps, stable_json, suggest_trace_template
from .summary import summarize_results

TRACE_IGNORED_TOOLS = {"cite", "explain", "finalize"}


def _id_key(value: object) -> str:
    return str(value)


def _short(text: str, limit: int = 180) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _normalize_expected_trace_paths(eval_entry: dict[str, Any]) -> list[list[dict[str, Any]]]:
    expected = eval_entry.get("expected", {}) or {}
    trace_paths = expected.get("trace_paths")
    if isinstance(trace_paths, list) and trace_paths and all(isinstance(path, list) for path in trace_paths):
        return [list(path) for path in trace_paths]
    trace = expected.get("trace", [])
    if isinstance(trace, list):
        return [list(trace)]
    return []


def _tool_seq_expected(path: list[dict[str, Any]]) -> list[str]:
    seq: list[str] = []
    for step in path:
        tool = str(step.get("tool") or "").strip()
        if not tool or tool in TRACE_IGNORED_TOOLS:
            continue
        seq.append(tool)
    return seq


def _tool_seq_received(received_trace: list[Any]) -> list[str]:
    seq: list[str] = []
    for step in _extract_received_steps(received_trace):
        tool = str(step.get("toolName") or "").strip()
        if not tool or tool in TRACE_IGNORED_TOOLS:
            continue
        seq.append(tool)
    return seq


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    m = len(a)
    n = len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def _similarity(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return (2.0 * _lcs_len(a, b)) / (len(a) + len(b))


def _select_best_expected_path(
    trace_paths: list[list[dict[str, Any]]],
    received_trace: list[Any],
) -> tuple[list[dict[str, Any]], int]:
    if not trace_paths:
        return [], 0
    if len(trace_paths) == 1:
        return trace_paths[0], 0

    received_seq = _tool_seq_received(received_trace)
    best_idx = 0
    best_score = -1.0
    for idx, path in enumerate(trace_paths):
        score = _similarity(_tool_seq_expected(path), received_seq)
        if score > best_score:
            best_score = score
            best_idx = idx
    return trace_paths[best_idx], best_idx


def _ensure_manual_struct(entry: dict[str, Any]) -> dict[str, Any]:
    if "triple_f1" not in entry:
        entry["triple_f1"] = float(entry.get("evidence_f1", 0.0))
    if "triple_precision" not in entry:
        entry["triple_precision"] = float(entry.get("evidence_precision", 0.0))
    if "triple_recall" not in entry:
        entry["triple_recall"] = float(entry.get("evidence_recall", 0.0))
    if "combined_f1" not in entry:
        entry["combined_f1"] = float(entry.get("triple_f1", 0.0))

    manual = entry.setdefault("manual_review", {})
    manual.setdefault("status", "pending" if entry.get("trace_valid") is None else "reviewed")
    manual.setdefault("trace_valid", entry.get("trace_valid"))
    manual.setdefault("comments", [])
    if not isinstance(manual.get("comments"), list):
        manual["comments"] = []
    return manual


def _set_trace_valid(entry: dict[str, Any], value: bool | None) -> None:
    entry["trace_valid"] = value
    if value is False:
        entry["combined_f1"] = 0.0
        entry["scoring_source"] = "manual_trace_gate"
    else:
        entry["combined_f1"] = float(entry.get("triple_f1", 0.0))
        entry["scoring_source"] = "manual_trace_gate" if value is True else "auto"

    manual = _ensure_manual_struct(entry)
    manual["trace_valid"] = value
    manual["status"] = "reviewed" if value is not None else "pending"
    manual["updated_at"] = datetime.now().isoformat()


@dataclass
class ReviewState:
    scores: dict[str, Any]
    eval_data: dict[str, Any]
    output_path: Path
    format_hint: str | None
    review_all: bool

    def __post_init__(self) -> None:
        self.eval_by_id = {_id_key(item.get("id")): item for item in self.eval_data.get("output", [])}
        self.per_question = self.scores.get("per_question", [])
        if self.review_all:
            self.review_items = self.per_question
        else:
            self.review_items = [item for item in self.per_question if item.get("trace_valid") is None]
            if not self.review_items:
                self.review_items = self.per_question

        for entry in self.per_question:
            _ensure_manual_struct(entry)

    def total(self) -> int:
        return len(self.review_items)

    def clamp_index(self, idx: int) -> int:
        if self.total() == 0:
            return 0
        return max(0, min(idx, self.total() - 1))

    def get_entry(self, idx: int) -> tuple[int, dict[str, Any], dict[str, Any]]:
        if self.total() == 0:
            raise ValueError("No questions available for review.")
        clamped = self.clamp_index(idx)
        entry = self.review_items[clamped]
        eval_entry = self.eval_by_id.get(_id_key(entry.get("id")))
        if not eval_entry:
            raise KeyError(f"Missing eval entry for question id={entry.get('id')}")
        return clamped, entry, eval_entry

    def save(self) -> None:
        self.scores["per_question"] = self.per_question
        summary = summarize_results(self.per_question, self.scores.get("metadata", {}))
        self.scores["metadata"] = summary["metadata"]
        self.scores["summary"] = summary["summary"]
        dump_data(self.scores, self.output_path, infer_format(self.output_path, self.format_hint))

    def question_context(self, idx: int) -> dict[str, Any]:
        index, entry, eval_entry = self.get_entry(idx)
        manual = _ensure_manual_struct(entry)

        expected = eval_entry.get("expected", {}) or {}
        received = eval_entry.get("received", {}) or {}
        received_trace = received.get("trace", []) or []
        if not entry.get("trace_template_id"):
            suggestion = suggest_trace_template(str(entry.get("qtype") or "unknown"), received_trace)
            entry["trace_template_id"] = suggestion.template_id
            entry["trace_template_similarity"] = suggestion.similarity
            entry["trace_template_exact"] = suggestion.exact_match
            entry["trace_template_tools"] = suggestion.template_tools
            entry["trace_tool_sequence"] = suggestion.sequence_tools
        trace_paths = _normalize_expected_trace_paths(eval_entry)
        expected_trace, selected_trace_path_idx = _select_best_expected_path(trace_paths, received_trace)
        received_steps = _extract_received_steps(received_trace)

        expected_rows = []
        for i, step in enumerate(expected_trace):
            expected_rows.append(
                {
                    "idx": i,
                    "tool": str(step.get("tool", "")),
                    "required": bool(step.get("required")),
                    "is_answer": bool(step.get("is_answer")),
                    "params_short": _short(stable_json(_extract_expected_params(step)), 160),
                    "params_full": stable_json(_extract_expected_params(step)),
                }
            )

        received_rows = []
        for i, step in enumerate(received_steps):
            params = step.get("toolParams") or {}
            received_rows.append(
                {
                    "idx": i,
                    "tool": str(step.get("toolName", "")),
                    "params_short": _short(stable_json(params), 160),
                    "params_full": stable_json(params),
                    "description": str(step.get("description", "")),
                    "description_short": _short(str(step.get("description", "")), 120),
                }
            )

        expected_triples = expected.get("triples", []) or []
        received_triples = received.get("triples", []) or []

        sidebar_items = []
        for row_idx, row in enumerate(self.review_items):
            status = (_ensure_manual_struct(row)).get("status", "pending")
            sidebar_items.append(
                {
                    "index": row_idx,
                    "id": row.get("id"),
                    "active": row_idx == index,
                    "status": status,
                    "combined_f1": float(row.get("combined_f1", 0.0)),
                }
            )

        reviewed_count = sum(1 for row in self.per_question if row.get("trace_valid") is not None)
        pending_count = len(self.per_question) - reviewed_count
        total = self.total()

        return {
            "index": index,
            "total": total,
            "prev_index": (index - 1) % total if total else 0,
            "next_index": (index + 1) % total if total else 0,
            "entry": entry,
            "question": eval_entry.get("question", ""),
            "qtype": entry.get("qtype", "unknown"),
            "status": manual.get("status", "pending"),
            "sidebar_items": sidebar_items,
            "reviewed_count": reviewed_count,
            "pending_count": pending_count,
            "trace_valid": entry.get("trace_valid"),
            "trace_template_id": entry.get("trace_template_id"),
            "trace_template_similarity": float(entry.get("trace_template_similarity", 0.0)),
            "trace_template_exact": bool(entry.get("trace_template_exact")),
            "trace_template_tools": entry.get("trace_template_tools", []) or [],
            "trace_tool_sequence": entry.get("trace_tool_sequence", []) or [],
            "expected_trace_rows": expected_rows,
            "model_trace_rows": received_rows,
            "expected_trace_path_idx": selected_trace_path_idx,
            "expected_trace_path_count": len(trace_paths),
            "expected_triples": expected_triples,
            "received_triples": received_triples,
            "triple_precision": float(entry.get("triple_precision", 0.0)),
            "triple_recall": float(entry.get("triple_recall", 0.0)),
            "triple_f1": float(entry.get("triple_f1", 0.0)),
            "combined_f1": float(entry.get("combined_f1", 0.0)),
            "comments": manual.get("comments", []),
        }


def create_app(state: ReviewState) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("templates")),
        static_folder=str(Path(__file__).with_name("static")),
    )
    app.secret_key = "manual-review-web"

    def _save_and_redirect(index: int, anchor: str | None = None) -> Any:
        state.save()
        if anchor:
            return redirect(url_for("question", index=index, _anchor=anchor))
        return redirect(url_for("question", index=index))

    @app.get("/")
    def home() -> Any:
        if state.total() == 0:
            return "No questions available for review."
        return redirect(url_for("question", index=0))

    @app.get("/question/<int:index>")
    def question(index: int) -> Any:
        if state.total() == 0:
            return "No questions available for review."
        context = state.question_context(index)
        return render_template("manual_review_web.html", **context)

    @app.post("/question/<int:index>/trace-decision")
    def trace_decision(index: int) -> Any:
        _, entry, _ = state.get_entry(index)
        action = (request.form.get("action") or "").strip().lower()

        if action == "valid":
            _set_trace_valid(entry, True)
            flash("Trace marked valid.", "info")
        elif action == "invalid":
            _set_trace_valid(entry, False)
            flash("Trace marked invalid (combined score set to 0).", "info")
        elif action == "clear":
            _set_trace_valid(entry, None)
            flash("Trace decision cleared.", "info")
        else:
            flash("Unknown action.", "error")

        return _save_and_redirect(index)

    @app.post("/question/<int:index>/comment")
    def add_comment(index: int) -> Any:
        _, entry, _ = state.get_entry(index)
        text = (request.form.get("comment") or "").strip()
        if not text:
            flash("Comment cannot be empty.", "error")
            return redirect(url_for("question", index=index, _anchor="comments"))

        manual = _ensure_manual_struct(entry)
        manual["comments"].append(text)
        manual["updated_at"] = datetime.now().isoformat()
        flash("Comment added.", "info")
        return _save_and_redirect(index, "comments")

    @app.post("/save")
    def save_scores() -> Any:
        try:
            index = int(request.form.get("index", "0"))
        except ValueError:
            index = 0
        state.save()
        flash(f"Saved results to {state.output_path}", "info")
        return redirect(url_for("question", index=state.clamp_index(index)))

    return app


def run_manual_review_web(
    scores_path: Path,
    eval_path: Path,
    output_path: Path,
    format_hint: str | None,
    review_all: bool,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> None:
    scores = load_data(scores_path)
    eval_data = load_data(eval_path)
    state = ReviewState(scores, eval_data, output_path, format_hint, review_all=review_all)
    app = create_app(state)

    print()
    print("Manual review web UI")
    print(f"- URL:  http://{host}:{port}")
    print(f"- Data: {output_path}")
    print()
    app.run(host=host, port=port, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual review UI (trace gate + triple-only score)")
    parser.add_argument("--scores", type=Path, required=True, help="Path to scored results")
    parser.add_argument("--eval", type=Path, required=True, help="Path to eval results")
    parser.add_argument("--output", type=Path, required=True, help="Path for updated scored output")
    parser.add_argument("--review-all", action="store_true", help="Review all rows, not only pending")
    parser.add_argument("--format", type=str, default=None, help="Output format override: json or toml")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    args = parser.parse_args()

    run_manual_review_web(
        scores_path=args.scores,
        eval_path=args.eval,
        output_path=args.output,
        format_hint=args.format,
        review_all=args.review_all,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
