"""Flask web UI for manual scoring review."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, url_for
import markdown

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from src.eval.langchain_adapter import MCP_SERVER_URL
from .io_utils import load_data, dump_data, infer_format
from .metrics import (
    _extract_received_steps,
    _extract_expected_params,
    match_trace,
    match_triple,
    extract_cited_execution_keys,
)
from .summary import summarize_results


RECEIVED_STEP_LABELS = ["ok", "wrong"]
RECEIVED_TRIPLE_LABELS = ["support", "reject", "ignore"]

ANOMALY_INTERPRETATIONS = {
    "triple_pos_trace_zero": "Triples > 0 but trace near zero; different path or trace mismatch.",
    "trace_pos_triple_zero": "Trace high but triples near zero; citation mismatch or retrieval mismatch.",
    "both_zero": "Both trace and triple near zero; failure or severe mismatch.",
    "triple_perfect_trace_not": "Triples near-perfect but trace not; model likely found a different path.",
    "literal_subject_mismatch": "Same literal but different subject URI; likely ambiguous entity.",
    "overcitation_low_precision_high_recall": "Recall is high but precision is very low; likely over-citation.",
    "overcitation_ratio": "Received citations far exceed expected citations; likely broad retrieval.",
    "impossible_found_not_false": "Impossible question did not declare found=false in explanation payload.",
}

DETAILS_TAG_RE = re.compile(r"<details(?![^>]*\bmarkdown=)([^>]*)>", flags=re.IGNORECASE)
TABLE_DELIM_RE = re.compile(r"^\s*\|?\s*:?-{3,}[-| :]*\|?\s*$")


def _id_key(value: object) -> str:
    return str(value)


def _short(text: str, limit: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _format_params(params: dict) -> str:
    if not params:
        return ""
    try:
        rendered = json.dumps(params, ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = str(params)
    return _short(rendered, 160)


def _format_params_long(params: dict) -> str:
    if not params:
        return ""
    try:
        return json.dumps(params, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(params)


def _trace_detail(step: dict, received: bool) -> str:
    if received:
        params = step.get("toolParams") or {}
    else:
        params = _extract_expected_params(step)
    return _format_params(params)


def _get_manual(entry: dict) -> dict:
    return entry.setdefault("manual_review", {})


def _get_labels(entry: dict, category: str, side: str) -> dict[str, str]:
    manual = _get_manual(entry)
    cat = manual.setdefault(category, {})
    key = f"{side}_labels"
    labels = cat.setdefault(key, {})
    if labels and any(not isinstance(k, str) for k in labels.keys()):
        labels = {str(k): v for k, v in labels.items()}
        cat[key] = labels
    return labels


def _get_comments(entry: dict) -> list[str]:
    manual = _get_manual(entry)
    comments = manual.setdefault("comments", [])
    normalized: list[str] = []
    for item in comments:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict) and "text" in item:
            normalized.append(str(item.get("text", "")))
    if normalized != comments:
        manual["comments"] = normalized
    return manual["comments"]


def _add_comment(entry: dict, text: str) -> None:
    comments = _get_comments(entry)
    comments.append(text)


def _seed_labels_from_auto(
    entry: dict,
    expected_steps: list[dict],
    received_steps: list[dict],
    expected_triples: list[dict],
    received_triples: list[dict],
) -> None:
    manual = _get_manual(entry)
    if manual.get("seeded_from_auto"):
        return

    trace_labels = _get_labels(entry, "trace", "received")
    triple_labels = _get_labels(entry, "triples", "received")

    if not trace_labels:
        trace_matches = match_trace(expected_steps, received_steps)
        for i in range(len(received_steps)):
            trace_labels[str(i)] = "ok" if i in trace_matches else "wrong"
    else:
        # Ensure all received steps always have an explicit label.
        trace_matches = match_trace(expected_steps, received_steps)
        for i in range(len(received_steps)):
            trace_labels.setdefault(str(i), "ok" if i in trace_matches else "wrong")

    if not triple_labels:
        triple_matches = match_triple(expected_triples, received_triples)
        for i in range(len(received_triples)):
            triple_labels[str(i)] = "support" if i in triple_matches else "reject"
    else:
        # Ensure all received triples always have an explicit label.
        triple_matches = match_triple(expected_triples, received_triples)
        for i in range(len(received_triples)):
            triple_labels.setdefault(str(i), "support" if i in triple_matches else "reject")

    manual["seeded_from_auto"] = True


def _get_expected_overrides(entry: dict) -> tuple[dict[str, bool], list[dict]]:
    manual = _get_manual(entry)
    trace = manual.setdefault("trace", {})
    overrides = trace.setdefault("expected_required_overrides", {})
    added = trace.setdefault("added_expected_steps", [])
    if overrides and any(not isinstance(k, str) for k in overrides.keys()):
        overrides = {str(k): bool(v) for k, v in overrides.items()}
        trace["expected_required_overrides"] = overrides
    return overrides, added


def _expected_step_from_received(step: dict) -> dict:
    tool = step.get("toolName") or ""
    params = step.get("toolParams") or {}
    expected: dict[str, Any] = {"tool": tool, "required": True, "_manual_added": True}

    if tool == "search":
        if "query" in params:
            expected["query"] = params["query"]
    elif tool == "inspect":
        if "uri" in params:
            expected["uri"] = params["uri"]
    elif tool == "fact":
        for key in ["subject", "predicate", "object"]:
            if key in params:
                expected[key] = params[key]
    elif tool == "query_builder":
        for key in ["type", "filters", "project"]:
            if key in params:
                expected[key] = params[key]
    elif tool == "query":
        if "query" in params:
            expected["query"] = params["query"]
    else:
        expected["params"] = params
    return expected


def _apply_expected_overrides(expected_steps: list[dict], overrides: dict[str, bool], added: list[dict]) -> list[dict]:
    updated = []
    for idx, step in enumerate(expected_steps):
        step_copy = dict(step)
        override = overrides.get(str(idx))
        if override is not None:
            step_copy["required"] = bool(override)
            if not override:
                step_copy["is_answer"] = False
        updated.append(step_copy)
    for step in added:
        step_copy = dict(step)
        if "required" not in step_copy:
            step_copy["required"] = True
        updated.append(step_copy)
    return updated


def _set_label(entry: dict, category: str, side: str, idx: int, label: str | None) -> None:
    manual = _get_manual(entry)
    cat = manual.setdefault(category, {})
    key = f"{side}_labels"
    labels = cat.setdefault(key, {})
    idx_key = str(idx)
    if label is None:
        labels.pop(idx_key, None)
    else:
        labels[idx_key] = label


def _compute_trace_metrics(
    received_steps: list[dict],
    expected_steps: list[dict],
    received_labels: dict[str, str],
    ignored_execution_keys: set[str] | None = None,
) -> tuple[float, float, float]:
    total_received = len(received_steps)
    ignored_execution_keys = ignored_execution_keys or set()
    required_indices = [
        i
        for i, step in enumerate(expected_steps)
        if step.get("required") and not step.get("is_answer")
    ]
    optional_indices = set(range(len(expected_steps))) - set(required_indices)

    matches = match_trace(expected_steps, received_steps)
    optional_only_received = {
        r_idx
        for r_idx, exp_list in matches.items()
        if exp_list and all(e_idx in optional_indices for e_idx in exp_list)
    }
    ignored_received = optional_only_received | {
        idx
        for idx, step in enumerate(received_steps)
        if step.get("executionKey") in ignored_execution_keys
    }
    ok_received_indices = {
        i for i in range(total_received)
        if received_labels.get(str(i)) == "ok" and i not in ignored_received
    }
    effective_received = total_received - len(ignored_received)
    precision = len(ok_received_indices) / effective_received if effective_received else 0.0

    if not required_indices:
        return precision, 0.0, 0.0

    matched_expected = {
        exp_idx
        for recv_idx, exp_list in matches.items()
        if recv_idx in ok_received_indices
        for exp_idx in exp_list
    }

    recall = len([i for i in required_indices if i in matched_expected]) / len(required_indices)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _compute_triple_metrics(
    received_triples: list[dict],
    expected_triples: list[dict],
    received_labels: dict[str, str],
) -> tuple[float, float, float]:
    total_received = len(received_triples)
    supported_indices = {
        i for i in range(total_received)
        if received_labels.get(str(i)) == "support"
    }
    ignored_indices = {
        i for i in range(total_received)
        if received_labels.get(str(i)) == "ignore"
    }
    effective_received = total_received - len(ignored_indices)
    precision = len(supported_indices) / effective_received if effective_received else 0.0

    matches = match_triple(expected_triples, received_triples)
    matched_expected = {
        expected_idx
        for recv_idx in supported_indices
        for expected_idx in matches.get(recv_idx, [])
    }
    recall = len(matched_expected) / len(expected_triples) if expected_triples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _apply_manual_scores(
    entry: dict,
    trace_precision: float,
    trace_recall: float,
    trace_f1: float,
    triple_precision: float,
    triple_recall: float,
    triple_f1: float,
    combined_f1: float,
) -> None:
    entry["manual_trace_precision"] = trace_precision
    entry["manual_trace_recall"] = trace_recall
    entry["manual_trace_f1"] = trace_f1
    entry["manual_triple_precision"] = triple_precision
    entry["manual_triple_recall"] = triple_recall
    entry["manual_triple_f1"] = triple_f1
    entry["manual_combined_f1"] = combined_f1

    entry["trace_precision"] = trace_precision
    entry["trace_recall"] = trace_recall
    entry["trace_f1"] = trace_f1
    entry["triple_precision"] = triple_precision
    entry["triple_recall"] = triple_recall
    entry["triple_f1"] = triple_f1
    entry["combined_f1"] = combined_f1

    entry["factored_in"] = True
    entry["exclude_reason"] = None
    entry["scoring_source"] = "manual"
    manual = _get_manual(entry)
    manual["reviewed"] = True
    manual["status"] = "scored"


def _apply_auto_scores(entry: dict) -> None:
    entry["trace_precision"] = entry.get("auto_trace_precision", entry.get("trace_precision", 0.0))
    entry["trace_recall"] = entry.get("auto_trace_recall", entry.get("trace_recall", 0.0))
    entry["trace_f1"] = entry.get("auto_trace_f1", entry.get("trace_f1", 0.0))
    entry["triple_precision"] = entry.get("auto_triple_precision", entry.get("triple_precision", 0.0))
    entry["triple_recall"] = entry.get("auto_triple_recall", entry.get("triple_recall", 0.0))
    entry["triple_f1"] = entry.get("auto_triple_f1", entry.get("triple_f1", 0.0))
    entry["combined_f1"] = entry.get("auto_combined_f1", entry.get("combined_f1", 0.0))
    entry["factored_in"] = True
    entry["exclude_reason"] = None
    entry["scoring_source"] = "auto"
    manual = _get_manual(entry)
    manual["reviewed"] = True
    manual["status"] = "accepted"


def _apply_skip(entry: dict) -> None:
    entry["factored_in"] = False
    entry["exclude_reason"] = "manual_skip"
    entry["scoring_source"] = "excluded"
    manual = _get_manual(entry)
    manual["reviewed"] = True
    manual["status"] = "skipped"


def _apply_perfect_scores(entry: dict) -> None:
    _apply_manual_scores(
        entry=entry,
        trace_precision=1.0,
        trace_recall=1.0,
        trace_f1=1.0,
        triple_precision=1.0,
        triple_recall=1.0,
        triple_f1=1.0,
        combined_f1=1.0,
    )
    entry["scoring_source"] = "manual_perfect"


async def _run_tool(tool_name: str, params: dict) -> Any:
    client = MultiServerMCPClient(
        {
            "kg-mcp": {
                "url": MCP_SERVER_URL,
                "transport": "http",
            }
        }
    )
    async with client.session("kg-mcp") as session:
        tools = await load_mcp_tools(session)
        tool_map = {t.name: t for t in tools}
        tool = tool_map.get(tool_name)
        if tool is None:
            raise RuntimeError(f"Tool not found: {tool_name}")
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(params)
        return tool.invoke(params)


def _run_coro(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _stringify_result(result: Any, limit: int = 8000) -> str:
    try:
        if isinstance(result, dict) and result.get("type") == "text" and "text" in result:
            text = str(result.get("text", ""))
        elif isinstance(result, list) and result and all(isinstance(x, dict) for x in result):
            texts = [str(item.get("text", "")) for item in result if item.get("type") == "text" and "text" in item]
            if texts:
                text = "\n\n".join(texts)
            else:
                text = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            text = json.dumps(result, indent=2, ensure_ascii=False)
    except Exception:
        text = str(result)
    if limit and len(text) > limit:
        return text[:limit] + "\n... (truncated)"
    return text


def _render_markdown(text: str) -> str:
    normalized = _normalize_markdown(text)
    try:
        return markdown.markdown(
            normalized,
            extensions=["extra", "tables", "sane_lists", "md_in_html"],
            output_format="html5",
        )
    except Exception:
        return markdown.markdown(
            normalized,
            extensions=["extra", "tables", "sane_lists"],
            output_format="html5",
        )


def _normalize_markdown(text: str) -> str:
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    # Ensure markdown can be parsed inside <details> blocks.
    value = DETAILS_TAG_RE.sub(r'<details markdown="1"\1>', value)

    # Help the table parser by ensuring a blank line before pipe-table headers.
    lines = value.split("\n")
    out: list[str] = []
    for idx, line in enumerate(lines):
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        is_table_header = line.lstrip().startswith("|")
        is_table_delim = bool(TABLE_DELIM_RE.match(next_line))
        if is_table_header and is_table_delim and out and out[-1].strip():
            out.append("")
        out.append(line)

    return "\n".join(out)


def _resolve_tool_params_expected(step: dict) -> tuple[str, dict]:
    tool = step.get("tool") or ""
    params = _extract_expected_params(step)
    return tool, params or {}


def _resolve_tool_params_received(step: dict) -> tuple[str, dict]:
    tool = step.get("toolName") or ""
    params = step.get("toolParams") or {}
    return tool, params


def _format_exception(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        parts = [str(sub) for sub in exc.exceptions]
        return "; ".join(parts) if parts else str(exc)
    return str(exc)


def _invert_matches(matches: dict[int, list[int]]) -> dict[int, list[int]]:
    rev: dict[int, list[int]] = {}
    for received_idx, expected_list in matches.items():
        for expected_idx in expected_list:
            rev.setdefault(expected_idx, []).append(received_idx)
    return rev


def _entry_status(entry: dict) -> str:
    manual_status = entry.get("manual_review", {}).get("status")
    if manual_status in {"pending", "scored", "accepted", "skipped"}:
        return manual_status
    source = entry.get("scoring_source")
    if source:
        return str(source)
    return "unknown"


@dataclass
class ReviewState:
    scores: dict
    eval_data: dict
    output_path: Path
    format_hint: str | None
    review_all: bool

    def __post_init__(self) -> None:
        self.eval_by_id = {_id_key(item.get("id")): item for item in self.eval_data.get("output", [])}
        self.per_question = self.scores.get("per_question", [])
        if self.review_all:
            self.review_items = self.per_question
        else:
            self.review_items = [entry for entry in self.per_question if entry.get("anomaly_detected")]

        for entry in self.per_question:
            entry.setdefault("auto_trace_precision", entry.get("trace_precision", 0.0))
            entry.setdefault("auto_trace_recall", entry.get("trace_recall", 0.0))
            entry.setdefault("auto_triple_precision", entry.get("triple_precision", 0.0))
            entry.setdefault("auto_triple_recall", entry.get("triple_recall", 0.0))

        self.alpha = float(self.scores.get("metadata", {}).get("scoring_alpha", 0.5))

    def total(self) -> int:
        return len(self.review_items)

    def ensure_has_items(self) -> None:
        if not self.review_items:
            raise RuntimeError("No questions selected for review.")

    def clamp_index(self, index: int) -> int:
        self.ensure_has_items()
        if index < 0:
            return 0
        if index >= self.total():
            return self.total() - 1
        return index

    def get_entry(self, index: int) -> tuple[int, dict, dict]:
        idx = self.clamp_index(index)
        entry = self.review_items[idx]
        eval_entry = self.eval_by_id.get(_id_key(entry.get("id")))
        if not eval_entry:
            raise KeyError(f"Missing eval data for question id={entry.get('id')}")
        return idx, entry, eval_entry

    def save(self) -> None:
        self.scores["per_question"] = self.per_question
        summary = summarize_results(
            self.per_question,
            self.scores.get("metadata", {}),
            include_anomalies=False,
            compute_pvalues=False,
        )
        self.scores["metadata"] = summary["metadata"]
        self.scores["summary"] = summary["summary"]
        fmt = infer_format(self.output_path, self.format_hint)
        dump_data(self.scores, self.output_path, fmt)

    def question_context(self, index: int) -> dict[str, Any]:
        idx, entry, eval_entry = self.get_entry(index)

        received_steps = _extract_received_steps(eval_entry.get("received", {}).get("trace", []))
        expected_steps_raw = eval_entry.get("expected", {}).get("trace", [])
        received_triples = eval_entry.get("received", {}).get("triples", [])
        expected_triples = eval_entry.get("expected", {}).get("triples", [])

        expected_overrides, expected_added = _get_expected_overrides(entry)
        expected_steps = _apply_expected_overrides(expected_steps_raw, expected_overrides, expected_added)

        ignored_execution_keys = extract_cited_execution_keys(eval_entry.get("runtime_trace", []))
        _seed_labels_from_auto(entry, expected_steps, received_steps, expected_triples, received_triples)
        received_step_labels = _get_labels(entry, "trace", "received")
        received_triple_labels = _get_labels(entry, "triples", "received")

        tp, tr, tf = _compute_trace_metrics(received_steps, expected_steps, received_step_labels, ignored_execution_keys)
        cp, cr, cf = _compute_triple_metrics(received_triples, expected_triples, received_triple_labels)
        manual_combined = self.alpha * tf + (1 - self.alpha) * cf

        trace_matches = match_trace(expected_steps, received_steps)
        trace_matches_rev = _invert_matches(trace_matches)
        triple_matches = match_triple(expected_triples, received_triples)
        triple_matches_rev = _invert_matches(triple_matches)

        model_trace_rows = []
        for step_idx, step in enumerate(received_steps):
            tool = step.get("toolName") or "unknown"
            explanation = (step.get("description") or "").strip()
            model_trace_rows.append(
                {
                    "idx": step_idx,
                    "tool": tool,
                    "is_answer": step.get("executionKey") in ignored_execution_keys,
                    "label": received_step_labels.get(str(step_idx), ""),
                    "params_short": _trace_detail(step, received=True),
                    "params_full": _format_params_long(step.get("toolParams") or {}),
                    "explanation": explanation,
                    "explanation_short": _short(explanation, 160) if explanation else "",
                    "explanation_missing": not explanation,
                    "matches": trace_matches.get(step_idx, []),
                    "matched": step_idx in trace_matches,
                }
            )

        expected_trace_rows = []
        for step_idx, step in enumerate(expected_steps):
            tool = step.get("tool") or "unknown"
            if step.get("_manual_added"):
                tool = f"{tool} (+)"
            expected_trace_rows.append(
                {
                    "idx": step_idx,
                    "tool": tool,
                    "required": bool(step.get("required")) and not bool(step.get("is_answer")),
                    "is_answer": bool(step.get("is_answer")),
                    "params_short": _trace_detail(step, received=False),
                    "params_full": _format_params_long(_extract_expected_params(step)),
                    "matches": trace_matches_rev.get(step_idx, []),
                    "matched": step_idx in trace_matches_rev,
                }
            )

        model_triple_rows = []
        for triple_idx, triple in enumerate(received_triples):
            model_triple_rows.append(
                {
                    "idx": triple_idx,
                    "label": received_triple_labels.get(str(triple_idx), ""),
                    "subject": triple.get("subject", ""),
                    "predicate": triple.get("predicate", ""),
                    "object": triple.get("object", ""),
                    "matches": triple_matches.get(triple_idx, []),
                    "matched": triple_idx in triple_matches,
                }
            )

        expected_triple_rows = []
        for triple_idx, triple in enumerate(expected_triples):
            expected_triple_rows.append(
                {
                    "idx": triple_idx,
                    "subject": triple.get("subject", ""),
                    "predicate": triple.get("predicate", ""),
                    "object": triple.get("object", ""),
                    "matches": triple_matches_rev.get(triple_idx, []),
                    "matched": triple_idx in triple_matches_rev,
                }
            )

        anomalies = entry.get("anomaly_flags") or []
        anomaly_notes = [(flag, ANOMALY_INTERPRETATIONS.get(flag, "")) for flag in anomalies]

        explain_trace = eval_entry.get("received", {}).get("trace", [])
        explain_summary = explain_trace[0] if explain_trace and isinstance(explain_trace[0], dict) else {}

        sidebar_items = []
        for row_idx, row in enumerate(self.review_items):
            sidebar_items.append(
                {
                    "index": row_idx,
                    "id": row.get("id"),
                    "status": _entry_status(row),
                    "anomaly": bool(row.get("anomaly_detected")),
                    "active": row_idx == idx,
                    "combined_f1": row.get("combined_f1", 0.0),
                }
            )

        total = self.total()
        prev_index = (idx - 1) % total
        next_index = (idx + 1) % total

        auto_trace = {
            "precision": float(entry.get("auto_trace_precision", entry.get("trace_precision", 0.0))),
            "recall": float(entry.get("auto_trace_recall", entry.get("trace_recall", 0.0))),
            "f1": float(entry.get("auto_trace_f1", entry.get("trace_f1", 0.0))),
        }
        auto_triple = {
            "precision": float(entry.get("auto_triple_precision", entry.get("triple_precision", 0.0))),
            "recall": float(entry.get("auto_triple_recall", entry.get("triple_recall", 0.0))),
            "f1": float(entry.get("auto_triple_f1", entry.get("triple_f1", 0.0))),
        }
        auto_combined = float(entry.get("auto_combined_f1", entry.get("combined_f1", 0.0)))

        manual_trace = {"precision": tp, "recall": tr, "f1": tf}
        manual_triple = {"precision": cp, "recall": cr, "f1": cf}
        triple_support_count = sum(1 for row in model_triple_rows if row["label"] == "support")
        triple_ignore_count = sum(1 for row in model_triple_rows if row["label"] == "ignore")
        triple_reject_count = max(0, len(model_triple_rows) - triple_support_count - triple_ignore_count)

        return {
            "index": idx,
            "total": total,
            "prev_index": prev_index,
            "next_index": next_index,
            "entry": entry,
            "eval_entry": eval_entry,
            "question": eval_entry.get("question", ""),
            "qtype": entry.get("qtype", "unknown"),
            "status": _entry_status(entry),
            "model_trace_rows": model_trace_rows,
            "expected_trace_rows": expected_trace_rows,
            "model_triple_rows": model_triple_rows,
            "expected_triple_rows": expected_triple_rows,
            "auto_trace": auto_trace,
            "auto_triple": auto_triple,
            "auto_combined": auto_combined,
            "manual_trace": manual_trace,
            "manual_triple": manual_triple,
            "manual_combined": manual_combined,
            "triple_support_count": triple_support_count,
            "triple_reject_count": triple_reject_count,
            "triple_ignore_count": triple_ignore_count,
            "anomaly_notes": anomaly_notes,
            "comments": _get_comments(entry),
            "sidebar_items": sidebar_items,
            "received_step_labels": RECEIVED_STEP_LABELS,
            "received_triple_labels": RECEIVED_TRIPLE_LABELS,
            "progress_done": idx,
            "progress_remaining": max(0, total - idx - 1),
            "runtime_trace_count": len(eval_entry.get("runtime_trace", [])),
            "explain_trace_count": len(explain_trace),
            "explain_summary": explain_summary,
        }


def create_app(state: ReviewState) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("templates")),
        static_folder=str(Path(__file__).with_name("static")),
    )
    app.secret_key = "manual-review-web"

    def _autosave_redirect(index: int, anchor: str | None = None) -> Any:
        try:
            state.save()
        except Exception as exc:
            flash(f"Autosave failed: {exc}", "error")
        if anchor:
            return redirect(url_for("question", index=index, _anchor=anchor))
        return redirect(url_for("question", index=index))

    @app.get("/")
    def home() -> Any:
        if state.total() == 0:
            return "No questions selected for review."
        return redirect(url_for("question", index=0))

    @app.get("/question/<int:index>")
    def question(index: int) -> Any:
        if state.total() == 0:
            return "No questions selected for review."
        clamped = state.clamp_index(index)
        if clamped != index:
            return redirect(url_for("question", index=clamped))

        try:
            context = state.question_context(index)
        except KeyError as exc:
            abort(404, str(exc))

        return render_template("manual_review_web.html", **context)

    @app.post("/question/<int:index>/label")
    def set_label(index: int) -> Any:
        _, entry, _ = state.get_entry(index)

        category = (request.form.get("category") or "").strip()
        side = (request.form.get("side") or "").strip()
        try:
            row_idx = int(request.form.get("idx", "-1"))
        except ValueError:
            flash("Invalid row index.", "error")
            return redirect(url_for("question", index=index))

        label = (request.form.get("label") or "").strip()

        valid = {
            ("trace", "received"): set(RECEIVED_STEP_LABELS),
            ("triples", "received"): set(RECEIVED_TRIPLE_LABELS),
        }
        allowed = valid.get((category, side))
        if allowed is None:
            flash("Invalid label target.", "error")
            return redirect(url_for("question", index=index))
        if (category, side) == ("trace", "received") and not label:
            flash("Trace label is required (ok/wrong).", "error")
            return redirect(url_for("question", index=index))
        if (category, side) == ("triples", "received") and not label:
            flash("Triple label is required (support/reject/ignore).", "error")
            return redirect(url_for("question", index=index))
        if label and label not in allowed:
            flash(f"Invalid label '{label}'.", "error")
            return redirect(url_for("question", index=index))

        _set_label(entry, category, side, row_idx, label or None)
        return _autosave_redirect(index, request.form.get("anchor") or "")

    @app.post("/question/<int:index>/expected-required")
    def set_expected_required(index: int) -> Any:
        _, entry, eval_entry = state.get_entry(index)

        try:
            row_idx = int(request.form.get("idx", "-1"))
        except ValueError:
            flash("Invalid expected-step index.", "error")
            return redirect(url_for("question", index=index))

        required = (request.form.get("required") or "false").strip().lower() == "true"

        expected_steps_raw = eval_entry.get("expected", {}).get("trace", [])
        expected_raw_len = len(expected_steps_raw)
        overrides, added = _get_expected_overrides(entry)

        if row_idx < expected_raw_len:
            overrides[str(row_idx)] = required
        else:
            added_idx = row_idx - expected_raw_len
            if 0 <= added_idx < len(added):
                added[added_idx]["required"] = required
                if not required:
                    added[added_idx]["is_answer"] = False
            else:
                flash("Expected-step index is out of range.", "error")

        return _autosave_redirect(index, "trace")

    @app.post("/question/<int:index>/expected-add")
    def add_expected(index: int) -> Any:
        _, entry, eval_entry = state.get_entry(index)
        try:
            model_idx = int(request.form.get("model_idx", "-1"))
        except ValueError:
            flash("Invalid model-step index.", "error")
            return redirect(url_for("question", index=index))

        received_steps = _extract_received_steps(eval_entry.get("received", {}).get("trace", []))
        if model_idx < 0 or model_idx >= len(received_steps):
            flash("Model-step index is out of range.", "error")
            return redirect(url_for("question", index=index, _anchor="trace"))

        _, added = _get_expected_overrides(entry)
        added.append(_expected_step_from_received(received_steps[model_idx]))
        flash("Copied model step into expected trace as required. You can toggle it to optional on the right.", "info")
        return _autosave_redirect(index, "trace")

    @app.post("/question/<int:index>/comment")
    def add_comment(index: int) -> Any:
        _, entry, _ = state.get_entry(index)
        text = (request.form.get("comment") or "").strip()
        changed = False
        if not text:
            flash("Comment cannot be empty.", "error")
        else:
            _add_comment(entry, text)
            changed = True
        if changed:
            return _autosave_redirect(index, "comments")
        return redirect(url_for("question", index=index, _anchor="comments"))

    @app.post("/question/<int:index>/decision")
    def set_decision(index: int) -> Any:
        _, entry, eval_entry = state.get_entry(index)
        action = (request.form.get("action") or "").strip()
        changed = False

        if action == "apply":
            apply_comment = (request.form.get("apply_comment") or "").strip()
            if not apply_comment:
                flash("Apply Manual requires a comment.", "error")
                return redirect(url_for("question", index=index))
            _add_comment(entry, f"[apply-manual] {apply_comment}")

            received_steps = _extract_received_steps(eval_entry.get("received", {}).get("trace", []))
            expected_steps_raw = eval_entry.get("expected", {}).get("trace", [])
            expected_overrides, expected_added = _get_expected_overrides(entry)
            expected_steps = _apply_expected_overrides(expected_steps_raw, expected_overrides, expected_added)
            received_triples = eval_entry.get("received", {}).get("triples", [])
            expected_triples = eval_entry.get("expected", {}).get("triples", [])
            rs = _get_labels(entry, "trace", "received")
            rt = _get_labels(entry, "triples", "received")
            ignored_execution_keys = extract_cited_execution_keys(eval_entry.get("runtime_trace", []))
            tp, tr, tf = _compute_trace_metrics(received_steps, expected_steps, rs, ignored_execution_keys)
            cp, cr, cf = _compute_triple_metrics(received_triples, expected_triples, rt)
            combined = state.alpha * tf + (1 - state.alpha) * cf
            _apply_manual_scores(entry, tp, tr, tf, cp, cr, cf, combined)
            flash("Applied manual scores.", "info")
            changed = True
        elif action == "accept":
            _apply_auto_scores(entry)
            flash("Accepted auto scores.", "info")
            changed = True
        elif action == "exclude":
            _apply_skip(entry)
            flash("Excluded from aggregate scoring.", "info")
            changed = True
        elif action == "perfect":
            perfect_comment = (request.form.get("perfect_comment") or "").strip()
            if not perfect_comment:
                flash("Perfect Answer requires a comment.", "error")
                return redirect(url_for("question", index=index))
            _add_comment(entry, f"[perfect-answer] {perfect_comment}")
            _apply_perfect_scores(entry)
            flash("Applied Perfect Answer override (all metrics set to 1.0).", "info")
            changed = True
        else:
            flash("Unknown decision action.", "error")

        if changed:
            return _autosave_redirect(index)
        return redirect(url_for("question", index=index))

    @app.post("/save")
    def save_scores() -> Any:
        try:
            index = int(request.form.get("index", "0"))
        except ValueError:
            index = 0

        state.save()
        flash(f"Saved results to {state.output_path}", "info")
        if state.total() == 0:
            return redirect(url_for("home"))
        return redirect(url_for("question", index=state.clamp_index(index)))

    @app.get("/api/question/<int:index>/runtime")
    def api_runtime(index: int) -> Any:
        _, _, eval_entry = state.get_entry(index)
        runtime_trace = eval_entry.get("runtime_trace", [])
        return jsonify({"count": len(runtime_trace), "payload": runtime_trace})

    @app.get("/api/question/<int:index>/explain")
    def api_explain(index: int) -> Any:
        _, _, eval_entry = state.get_entry(index)
        explain_trace = eval_entry.get("received", {}).get("trace", [])
        return jsonify({"count": len(explain_trace), "payload": explain_trace})

    @app.post("/api/question/<int:index>/reexec")
    def api_reexec(index: int) -> Any:
        _, entry, eval_entry = state.get_entry(index)
        payload = request.get_json(silent=True) or {}
        side = str(payload.get("side") or "").strip().lower()
        try:
            row_idx = int(payload.get("idx", -1))
        except Exception:
            return jsonify({"ok": False, "error": "Invalid row index."}), 400

        try:
            if side == "model":
                received_steps = _extract_received_steps(eval_entry.get("received", {}).get("trace", []))
                if row_idx < 0 or row_idx >= len(received_steps):
                    return jsonify({"ok": False, "error": "Model row index out of range."}), 400
                tool, params = _resolve_tool_params_received(received_steps[row_idx])
            elif side == "expected":
                expected_steps_raw = eval_entry.get("expected", {}).get("trace", [])
                expected_overrides, expected_added = _get_expected_overrides(entry)
                expected_steps = _apply_expected_overrides(expected_steps_raw, expected_overrides, expected_added)
                if row_idx < 0 or row_idx >= len(expected_steps):
                    return jsonify({"ok": False, "error": "Expected row index out of range."}), 400
                tool, params = _resolve_tool_params_expected(expected_steps[row_idx])
            else:
                return jsonify({"ok": False, "error": "Invalid side. Use 'model' or 'expected'."}), 400

            result = _run_coro(_run_tool(tool, params))
            result_text = _stringify_result(result)
            return jsonify(
                {
                    "ok": True,
                    "tool": tool,
                    "params": params,
                    "result": result_text,
                    "result_html": _render_markdown(result_text),
                }
            )
        except BaseException as exc:
            return jsonify({"ok": False, "error": _format_exception(exc)}), 500

    return app


def run_manual_review_web(
    scores_path: Path,
    eval_path: Path,
    output_path: Path,
    format_hint: str | None,
    anomalies_only: bool,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> None:
    scores = load_data(scores_path)
    eval_data = load_data(eval_path)
    state = ReviewState(scores, eval_data, output_path, format_hint, review_all=not anomalies_only)
    app = create_app(state)

    print()
    print("Manual review web UI")
    print(f"- URL:  http://{host}:{port}")
    print(f"- Data: {output_path}")
    print()

    app.run(host=host, port=port, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual scoring web UI (Flask)")
    parser.add_argument("--scores", type=Path, required=True, help="Path to scored results (TOML/JSON)")
    parser.add_argument("--eval", type=Path, required=True, help="Path to raw eval results (TOML/JSON)")
    parser.add_argument("--output", type=Path, required=True, help="Path to write updated scores (TOML/JSON)")
    parser.add_argument("--anomalies", action="store_true", help="Review anomalies only")
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        help="Output format: json or toml (defaults to inferred from output path)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Flask app")
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask app")
    args = parser.parse_args()

    run_manual_review_web(
        scores_path=args.scores,
        eval_path=args.eval,
        output_path=args.output,
        format_hint=args.format,
        anomalies_only=args.anomalies,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
