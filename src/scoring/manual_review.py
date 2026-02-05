"""
Textual TUI for manual scoring.

Features:
- Side-by-side model vs expected traces and triples
- Per-step and per-triple labels
- Live MCP re-execution of any step
- Auto score summary + anomaly interpretation
- Writes manual scores back into scored results file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll, HorizontalScroll
from rich.text import Text
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static, Label, Markdown, TextArea
from textual.message import Message

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from src.eval.langchain_adapter import MCP_SERVER_URL
from .io_utils import load_data, dump_data, infer_format
from .metrics import _extract_received_steps, _extract_expected_params, match_trace, match_triple, extract_cited_execution_keys
from .summary import summarize_results


RECEIVED_STEP_LABELS = ["ok", "wrong"]
RECEIVED_TRIPLE_LABELS = ["ok", "wrong", "extra"]

ANOMALY_INTERPRETATIONS = {
    "triple_pos_trace_zero": "Triples > 0 but trace near zero; different path or trace mismatch.",
    "trace_pos_triple_zero": "Trace high but triples near zero; citation mismatch or retrieval mismatch.",
    "both_zero": "Both trace and triple near zero; failure or severe mismatch.",
    "triple_perfect_trace_not": "Triples near-perfect but trace not; model likely found a different path.",
    "literal_subject_mismatch": "Same literal but different subject URI; likely ambiguous entity.",
}


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
        params = _format_params(step.get("toolParams") or {})
    else:
        params = _format_params(_extract_expected_params(step))
    return params



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
    # normalize legacy dict comments to strings if possible
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

    if not triple_labels:
        triple_matches = match_triple(expected_triples, received_triples)
        for i in range(len(received_triples)):
            triple_labels[str(i)] = "ok" if i in triple_matches else "wrong"

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
        # Fallback: store params to preserve context
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
    ok_received_indices = {i for i in range(total_received) if received_labels.get(str(i)) == "ok"}
    precision = len(ok_received_indices) / total_received if total_received else 0.0

    matches = match_triple(expected_triples, received_triples)
    covered = 0
    for i in ok_received_indices:
        if i in matches:
            covered += 1
    recall = covered / len(expected_triples) if expected_triples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _apply_manual_scores(entry: dict, trace_f1: float, triple_f1: float, combined_f1: float) -> None:
    entry["manual_trace_f1"] = trace_f1
    entry["manual_triple_f1"] = triple_f1
    entry["manual_combined_f1"] = combined_f1
    entry["trace_f1"] = trace_f1
    entry["triple_f1"] = triple_f1
    entry["combined_f1"] = combined_f1
    entry["factored_in"] = True
    entry["exclude_reason"] = None
    entry["scoring_source"] = "manual"
    manual = _get_manual(entry)
    manual["reviewed"] = True
    manual["status"] = "scored"


def _apply_auto_scores(entry: dict) -> None:
    entry["trace_f1"] = entry.get("auto_trace_f1", entry.get("trace_f1", 0.0))
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


async def _run_tool(tool_name: str, params: dict) -> Any:
    client = MultiServerMCPClient({
        "kg-mcp": {
            "url": MCP_SERVER_URL,
            "transport": "http",
        }
    })
    async with client.session("kg-mcp") as session:
        tools = await load_mcp_tools(session)
        tool_map = {t.name: t for t in tools}
        tool = tool_map.get(tool_name)
        if tool is None:
            raise RuntimeError(f"Tool not found: {tool_name}")
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(params)
        return tool.invoke(params)


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


def _resolve_tool_params_expected(step: dict) -> tuple[str, dict]:
    tool = step.get("tool") or ""
    params = _extract_expected_params(step)
    return tool, params or {}


def _resolve_tool_params_received(step: dict) -> tuple[str, dict]:
    tool = step.get("toolName") or ""
    params = step.get("toolParams") or {}
    return tool, params


class ToolOutput(Message):
    def __init__(self, text: str, markdown: bool | None = False) -> None:
        super().__init__()
        self.text = text
        self.markdown = markdown


class CommentModal(ModalScreen[str | None]):
    def compose(self) -> ComposeResult:
        with Container(id="comment_modal"):
            yield Label("Add Comment", id="comment_modal_title")
            yield TextArea(id="comment_modal_input", compact=False)
            with Horizontal(id="comment_modal_actions"):
                yield Button("Cancel", id="comment_modal_cancel")
                yield Button("Save", id="comment_modal_save")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "comment_modal_cancel":
            self.dismiss(None)
            return
        if event.button.id == "comment_modal_save":
            input_area = self.query_one("#comment_modal_input", TextArea)
            text = (input_area.text or "").strip()
            self.dismiss(text if text else None)


def _format_exception(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        parts = []
        for sub in exc.exceptions:
            parts.append(str(sub))
        return "; ".join(parts) if parts else str(exc)
    return str(exc)


class ReviewApp(App):
    CSS = """
    Screen {
        layout: vertical;
        background: #0f1117;
        color: #e6edf5;
    }
    #header {
        border: solid #2d5ea6;
        background: #12161d;
    }
    #comment_modal {
        border: solid #2d5ea6;
        background: #12161d;
        padding: 1 2;
        width: 70%;
        height: 60%;
    }
    #comment_modal_input {
        border: solid #2d5ea6;
        height: 1fr;
    }
    #comment_modal_actions {
        height: auto;
        margin-top: 1;
    }
    #legend {
        color: #9aa7b7;
    }
    #row1, #row2 {
        layout: horizontal;
    }
    #row1 {
        height: 1.6fr;
        min-height: 10;
    }
    #row2 {
        height: 1.6fr;
        min-height: 10;
    }
    #row1, #row2, #bottom {
        border: solid #1b2636;
    }

    #model_trace, #model_triples {
        background: #1f2a38;
    }
    #expected_trace, #expected_triples {
        background: #20262f;
    }
    #bottom {
        layout: horizontal;
        height: 1.4fr;
        min-height: 10;
    }
    #output_panel {
        border: solid #2d5ea6;
        overflow: scroll;
        scrollbar-gutter: stable;
    }
    #controls {
        border: solid #2d5ea6;
        overflow: scroll;
        scrollbar-gutter: stable;
    }
    #model_trace_title, #model_triples_title {
        color: #6aa4ff;
    }
    #expected_trace_title, #expected_triples_title {
        color: #9aa7b7;
    }
    Button {
        background: #1a2230;
        color: #e6edf5;
        border: solid #2d5ea6;
    }
    Button:hover {
        background: #22314a;
    }
    .panel {
        border: solid #2d5ea6;
        overflow: scroll scroll;
    }
    .table-panel {
        layout: vertical;
        width: 1fr;
        height: 1fr;
        border: solid #2d5ea6;
        background: #12161d;
        
    }
    .label-bar {
        height: auto;
        overflow: scroll hidden;
        scrollbar-gutter: stable;
        border-top: solid #2d5ea6;
        background: #141923;
        
    }
    #model_trace, #expected_trace, #model_triples, #expected_triples {
        height: 1fr;
        min-height: 8;
    }
    .label-bar Button {
        margin: 0 1 0 0;
    }
    .control-row Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(self, scores: dict, eval_data: dict, output_path: Path, format_hint: str | None, review_all: bool) -> None:
        super().__init__()
        self.scores = scores
        self.eval_data = eval_data
        self.output_path = output_path
        self.format_hint = format_hint
        self.review_all = review_all

        self.eval_by_id = {_id_key(item.get("id")): item for item in eval_data.get("output", [])}
        self.per_question = scores.get("per_question", [])
        if review_all:
            self.review_items = self.per_question
        else:
            self.review_items = [entry for entry in self.per_question if entry.get("anomaly_detected")]

        self.alpha = float(scores.get("metadata", {}).get("scoring_alpha", 0.5))
        self.index = 0
        self.active_table = "model_trace"
        self.selected_indices: dict[str, int | None] = {
            "model_trace": None,
            "expected_trace": None,
            "model_triples": None,
            "expected_triples": None,
        }
        self.highlight_rows: dict[str, set[int]] = {
            "model_trace": set(),
            "expected_trace": set(),
            "model_triples": set(),
            "expected_triples": set(),
        }
        self.trace_matches: dict[int, list[int]] = {}
        self.trace_matches_rev: dict[int, list[int]] = {}
        self.triple_matches: dict[int, list[int]] = {}
        self.triple_matches_rev: dict[int, list[int]] = {}
        self.current_entry: dict | None = None
        self.current_eval_entry: dict | None = None
        self.current_expected_raw_len: int = 0
        self.current_received_steps: list[dict] = []
        self.current_expected_steps: list[dict] = []
        self.current_received_triples: list[dict] = []
        self.current_expected_triples: list[dict] = []
        self.current_received_step_labels: dict[str, str] = {}
        self.current_received_triple_labels: dict[str, str] = {}
        self.current_ignored_execution_keys: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Horizontal(id="row1"):
            with Vertical(id="model_trace_panel", classes="table-panel"):
                yield Label("Model Trace", id="model_trace_title")
                yield DataTable(id="model_trace", zebra_stripes=True)
                with HorizontalScroll(classes="label-bar"):
                    for label in RECEIVED_STEP_LABELS + ["clear"]:
                        yield Button(label, id=f"label_trace_received_{label}", compact=True)
                    yield Button("Re-exec Model Step", id="reexec_model", compact=True)
            with Vertical(id="expected_trace_panel", classes="table-panel"):
                yield Label("Expected Trace", id="expected_trace_title")
                yield DataTable(id="expected_trace")
                with HorizontalScroll(classes="label-bar"):
                    yield Button("Mark Required", id="expected_require", compact=True)
                    yield Button("Mark Not Required", id="expected_optional", compact=True)
                    yield Button("Add Required Step", id="expected_add_required", compact=True)
                    yield Button("Re-exec Expected Step", id="reexec_expected", compact=True)
            with Container(id="controls", classes="panel"):
                yield Static(
                    "Manual mapping: P = ok/received, R = ok & matched required/required.",
                    id="legend",
                )
                with Horizontal(classes="control-row"):
                    yield Button("Prev", id="prev")
                    yield Button("Next", id="next")
                    yield Static("", id="progress_status")
                    yield Button("Apply Manual Scores", id="apply")
                    yield Button("Accept Auto Scores", id="accept")
                with Horizontal(classes="control-row"):
                    yield Button("Exclude", id="exclude")
                    yield Button("Add Comment", id="add_comment")
                    yield Button("Save", id="save")
                    yield Button("Quit", id="quit")
        with Horizontal(id="row2"):
            with Vertical(id="model_triples_panel", classes="table-panel"):
                yield Label("Model Triples", id="model_triples_title")
                yield DataTable(id="model_triples")
                with HorizontalScroll(classes="label-bar"):
                    for label in RECEIVED_TRIPLE_LABELS + ["clear"]:
                        yield Button(label, id=f"label_triples_received_{label}", compact=True)
            with Vertical(id="expected_triples_panel", classes="table-panel"):
                yield Label("Expected Triples", id="expected_triples_title")
                yield DataTable(id="expected_triples")
                # expected triples are shown for reference only
        with Horizontal(id="bottom"):
            with VerticalScroll(id="output_panel"):
                yield Markdown(id="output_md")
                yield TextArea(id="output_text", read_only=True)

    def on_mount(self) -> None:
        if not self.review_items:
            self.exit(message="No questions selected for review.")
            return
        for table_id in ("model_trace", "expected_trace", "model_triples", "expected_triples"):
            table = self.query_one(f"#{table_id}", DataTable)
            table.cursor_type = "row"
            table.show_cursor = True
        panel = self.query_one("#output_panel", VerticalScroll)
        panel.show_vertical_scrollbar = True
        panel.show_horizontal_scrollbar = True
        # controls = self.query_one("#controls", VerticalScroll)
        # controls.show_vertical_scrollbar = True
        # controls.show_horizontal_scrollbar = True
        md = self.query_one("#output_md", Markdown)
        txt = self.query_one("#output_text", TextArea)
        txt.show_vertical_scrollbar = True
        txt.show_horizontal_scrollbar = True
        md.display = False
        txt.display = True
        self.refresh_view()

    def refresh_view(self) -> None:
        entry = self.review_items[self.index]
        eval_entry = self.eval_by_id.get(_id_key(entry.get("id")))
        if not eval_entry:
            return

        question = eval_entry.get("question", "")
        self._set_output(question, markdown=False)

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

        tp, tr, tf = _compute_trace_metrics(
            received_steps,
            expected_steps,
            received_step_labels,
            ignored_execution_keys,
        )
        cp, cr, cf = _compute_triple_metrics(received_triples, expected_triples, received_triple_labels)
        combined = self.alpha * tf + (1 - self.alpha) * cf

        header = self.query_one("#header", Static)
        header.update(self._render_header(entry, eval_entry, tp, tr, tf, cp, cr, cf, combined))
        progress_status = self.query_one("#progress_status", Static)
        progress_status.update(self._render_progress_status())

        self.current_entry = entry
        self.current_eval_entry = eval_entry
        self.current_expected_raw_len = len(expected_steps_raw)
        self.current_received_steps = received_steps
        self.current_expected_steps = expected_steps
        self.current_received_triples = received_triples
        self.current_expected_triples = expected_triples
        self.current_received_step_labels = received_step_labels
        self.current_received_triple_labels = received_triple_labels
        self.current_ignored_execution_keys = ignored_execution_keys

        self.trace_matches = match_trace(expected_steps, received_steps)
        self.trace_matches_rev = self._invert_matches(self.trace_matches)
        self.triple_matches = match_triple(expected_triples, received_triples)
        self.triple_matches_rev = self._invert_matches(self.triple_matches)
        for key in self.highlight_rows:
            self.highlight_rows[key] = set()

        self._render_tables()

    def _open_comment_modal(self) -> None:
        def _on_close(text: str | None) -> None:
            if not text:
                return
            entry = self.review_items[self.index]
            _add_comment(entry, text)
            self._set_output("Comment added.")
            self.refresh_view()

        self.push_screen(CommentModal(), _on_close)

    def _render_progress_status(self) -> str:
        total = len(self.review_items)
        done = self.index
        remaining = max(0, total - self.index - 1)
        return f"Done: {done}/{total} • Ahead: {remaining}"

    def _render_tables(self) -> None:
        if self.current_eval_entry is None:
            return
        self._fill_trace_table(
            "model_trace",
            self.current_received_steps,
            self.current_received_step_labels,
            received=True,
            expected_steps=self.current_expected_steps,
            received_steps=self.current_received_steps,
            highlight_rows=self.highlight_rows["model_trace"],
        )
        self._fill_trace_table(
            "expected_trace",
            self.current_expected_steps,
            {},
            received=False,
            expected_steps=self.current_expected_steps,
            received_steps=self.current_received_steps,
            highlight_rows=self.highlight_rows["expected_trace"],
        )
        self._fill_triples_table(
            "model_triples",
            self.current_received_triples,
            self.current_received_triple_labels,
            received=True,
            expected_triples=self.current_expected_triples,
            received_triples=self.current_received_triples,
            highlight_rows=self.highlight_rows["model_triples"],
        )
        self._fill_triples_table(
            "expected_triples",
            self.current_expected_triples,
            {},
            received=False,
            expected_triples=self.current_expected_triples,
            received_triples=self.current_received_triples,
            highlight_rows=self.highlight_rows["expected_triples"],
        )

    def _render_header(self, entry: dict, eval_entry: dict, tp: float, tr: float, tf: float, cp: float, cr: float, cf: float, combined: float) -> str:
        auto_trace_p = entry.get("trace_precision", 0.0)
        auto_trace_r = entry.get("trace_recall", 0.0)
        auto_trace_f = entry.get("auto_trace_f1", entry.get("trace_f1", 0.0))
        auto_triple_p = entry.get("triple_precision", 0.0)
        auto_triple_r = entry.get("triple_recall", 0.0)
        auto_triple_f = entry.get("auto_triple_f1", entry.get("triple_f1", 0.0))
        auto_combined = entry.get("auto_combined_f1", entry.get("combined_f1", 0.0))
        anomalies = entry.get("anomaly_flags") or []

        lines = []
        lines.append(f"ID {entry.get('id')}  Type: {entry.get('qtype')}")
        lines.append(_short(eval_entry.get("question", ""), 140))
        lines.append("")
        lines.append(f"Auto Trace    P {auto_trace_p:.3f}  R {auto_trace_r:.3f}  F1 {auto_trace_f:.3f}")
        lines.append(f"Auto Triple   P {auto_triple_p:.3f}  R {auto_triple_r:.3f}  F1 {auto_triple_f:.3f}")
        lines.append(f"Auto Combined {auto_combined:.3f}")
        lines.append("")
        lines.append(f"Manual Trace  P {tp:.3f}  R {tr:.3f}  F1 {tf:.3f}")
        lines.append(f"Manual Triple P {cp:.3f}  R {cr:.3f}  F1 {cf:.3f}")
        lines.append(f"Manual Combined {combined:.3f}")
        if anomalies:
            lines.append("")
            lines.append("Anomalies: " + ", ".join(anomalies))
            for flag in anomalies:
                note = ANOMALY_INTERPRETATIONS.get(flag)
                if note:
                    lines.append(f"- {flag}: {note}")
        return "\n".join(lines)

    def _fill_trace_table(
        self,
        table_id: str,
        steps: list[dict],
        labels: dict[str, str],
        received: bool,
        expected_steps: list[dict] | None = None,
        received_steps: list[dict] | None = None,
        highlight_rows: set[int] | None = None,
    ) -> None:
        table = self.query_one(f"#{table_id}", DataTable)
        table.clear(columns=True)
        table.add_column("Idx", width=4)
        table.add_column("Tool", width=18)
        if received:
            table.add_column("Ans", width=8)
            table.add_column("Label", width=12)
        else:
            table.add_column("Req", width=6)
        table.add_column("Call Params", width=60)

        matched_expected = set()
        matched_received = set()
        if expected_steps is not None and received_steps is not None:
            matches = match_trace(expected_steps, received_steps)
            matched_received = set(matches.keys())
            matched_expected = {e_idx for e_list in matches.values() for e_idx in e_list}

        for idx, step in enumerate(steps):
            tool = step.get("toolName") if received else step.get("tool")
            tool = tool or "unknown"
            tool_cell: str | Text = tool
            if not received:
                if step.get("is_answer"):
                    tool = f"{tool} (ans)"
                elif step.get("required"):
                    tool = f"{tool} (req)"
                if step.get("_manual_added"):
                    tool = f"{tool} (+)"
            desc = _trace_detail(step, received)
            label = labels.get(str(idx), "")
            if received:
                if step.get("executionKey") in self.current_ignored_execution_keys:
                    ans_flag = Text("answer", style="bold yellow")
                else:
                    ans_flag = Text("normal", style="dim #7a8aa0")
                if idx in matched_received:
                    style = "green"
                elif highlight_rows and idx in highlight_rows:
                    style = "bold #6aa4ff"
                else:
                    style = None
                self._add_row(
                    table,
                    [str(idx), tool_cell, ans_flag, label, _short(desc, 120)],
                    key=str(idx),
                    style=style,
                )
            else:
                if idx in matched_expected:
                    style = "green"
                elif highlight_rows and idx in highlight_rows:
                    style = "bold #6aa4ff"
                else:
                    style = None
                if step.get("is_answer"):
                    req_flag = Text("ans", style="bold yellow")
                elif step.get("required"):
                    req_flag = Text("req", style="bold")
                else:
                    req_flag = Text("opt", style="dim #7a8aa0")
                self._add_row(table, [str(idx), tool, req_flag, _short(desc, 120)], key=str(idx), style=style)

    def _fill_triples_table(
        self,
        table_id: str,
        triples: list[dict],
        labels: dict[str, str],
        received: bool,
        expected_triples: list[dict] | None = None,
        received_triples: list[dict] | None = None,
        highlight_rows: set[int] | None = None,
    ) -> None:
        table = self.query_one(f"#{table_id}", DataTable)
        table.clear(columns=True)
        table.add_column("Idx", width=4)
        if received:
            table.add_column("Label", width=12)
        table.add_column("Subject", width=32)
        table.add_column("Predicate", width=32)
        table.add_column("Object", width=60)

        matches = {}
        matched_expected = set()
        matched_received = set()
        if expected_triples is not None and received_triples is not None:
            matches = match_triple(expected_triples, received_triples)
            matched_received = set(matches.keys())
            matched_expected = {e_idx for e_list in matches.values() for e_idx in e_list}

        for idx, t in enumerate(triples):
            label = labels.get(str(idx), "") if labels else ""
            matched = idx in matched_received if received else idx in matched_expected
            if matched:
                style = "green"
            elif highlight_rows and idx in highlight_rows:
                style = "bold #6aa4ff"
            else:
                style = None
            if received:
                self._add_row(
                    table,
                    [
                        str(idx),
                        label,
                        _short(t.get("subject", ""), 32),
                        _short(t.get("predicate", ""), 32),
                        _short(t.get("object", ""), 80),
                    ],
                    key=str(idx),
                    style=style,
                )
            else:
                self._add_row(
                    table,
                    [
                        str(idx),
                        _short(t.get("subject", ""), 32),
                        _short(t.get("predicate", ""), 32),
                        _short(t.get("object", ""), 80),
                    ],
                    key=str(idx),
                    style=style,
                )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        return

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table_id = event.data_table.id
        if table_id:
            self.active_table = table_id
            self.selected_indices[table_id] = event.cursor_row
            self._update_highlights(table_id, event.cursor_row)
            self._show_row_detail(table_id, event.cursor_row)

    def _show_row_detail(self, table_id: str | None, row_index: int | None) -> None:
        if not table_id or row_index is None:
            return
        entry = self.review_items[self.index]
        eval_entry = self.eval_by_id.get(_id_key(entry.get("id")))
        if not eval_entry:
            return
        idx = row_index
        if table_id in {"model_trace", "expected_trace"}:
            steps = _extract_received_steps(eval_entry.get("received", {}).get("trace", [])) if table_id == "model_trace" else eval_entry.get("expected", {}).get("trace", [])
            if idx >= len(steps):
                return
            step = steps[idx]
            if table_id == "model_trace":
                params = step.get("toolParams") or {}
            else:
                if "params" in step:
                    params = step.get("params")
                else:
                    params = _extract_expected_params(step)
            detail = _format_params_long(params) or "(no params)"
            if table_id == "model_trace":
                matched = self.trace_matches.get(idx, [])
                if matched:
                    detail += "\n\nMatched expected:\n"
                    detail += "\n".join(self._format_trace_match_details(matched, self.current_expected_steps))
            else:
                matched = self.trace_matches_rev.get(idx, [])
                if matched:
                    detail += "\n\nMatched received:\n"
                    detail += "\n".join(self._format_trace_match_details(matched, self.current_received_steps))
            self._set_output(detail)
        elif table_id in {"model_triples", "expected_triples"}:
            triples = eval_entry.get("received", {}).get("triples", []) if table_id == "model_triples" else eval_entry.get("expected", {}).get("triples", [])
            if idx >= len(triples):
                return
            t = triples[idx]
            detail = json.dumps(
                {
                    "subject": t.get("subject", ""),
                    "predicate": t.get("predicate", ""),
                    "object": t.get("object", ""),
                },
                indent=2,
                ensure_ascii=False,
            )
            if table_id == "model_triples":
                matched = self.triple_matches.get(idx, [])
                if matched:
                    detail += "\n\nMatched expected:\n"
                    detail += "\n".join(self._format_triple_match_details(matched, eval_entry.get("expected", {}).get("triples", [])))
            else:
                matched = self.triple_matches_rev.get(idx, [])
                if matched:
                    detail += "\n\nMatched received:\n"
                    detail += "\n".join(self._format_triple_match_details(matched, eval_entry.get("received", {}).get("triples", [])))
            self._set_output(detail)

    @staticmethod
    def _add_row(table: DataTable, cells: list[str], key: str, style: str | None) -> None:
        if style:
            styled_cells = [cell if isinstance(cell, Text) else Text(str(cell), style=style) for cell in cells]
            table.add_row(*styled_cells, key=key)
        else:
            table.add_row(*cells, key=key)

    @staticmethod
    def _invert_matches(matches: dict[int, list[int]]) -> dict[int, list[int]]:
        rev: dict[int, list[int]] = {}
        for r_idx, exp_list in matches.items():
            for e_idx in exp_list:
                rev.setdefault(e_idx, []).append(r_idx)
        return rev

    def _update_highlights(self, table_id: str, row_index: int) -> None:
        for key in self.highlight_rows:
            self.highlight_rows[key] = set()
        if table_id == "model_trace":
            self.highlight_rows["expected_trace"] = set(self.trace_matches.get(row_index, []))
        elif table_id == "expected_trace":
            self.highlight_rows["model_trace"] = set(self.trace_matches_rev.get(row_index, []))
        elif table_id == "model_triples":
            self.highlight_rows["expected_triples"] = set(self.triple_matches.get(row_index, []))
        elif table_id == "expected_triples":
            self.highlight_rows["model_triples"] = set(self.triple_matches_rev.get(row_index, []))
        self._render_tables()

    @staticmethod
    def _format_trace_match_details(indices: list[int], steps: list[dict]) -> list[str]:
        lines = []
        for idx in indices:
            if idx >= len(steps):
                continue
            step = steps[idx]
            tool = step.get("toolName") or step.get("tool") or "unknown"
            if "toolParams" in step:
                params = step.get("toolParams")
            elif "params" in step:
                params = step.get("params")
            else:
                params = _extract_expected_params(step)
            params_text = _format_params_long(params) or "(no params)"
            lines.append(f"- [{idx}] {tool}: {params_text}")
        return lines

    @staticmethod
    def _format_triple_match_details(indices: list[int], triples: list[dict]) -> list[str]:
        lines = []
        for idx in indices:
            if idx >= len(triples):
                continue
            t = triples[idx]
            lines.append(
                f"- [{idx}] subject={t.get('subject','')}, predicate={t.get('predicate','')}, object={t.get('object','')}"
            )
        return lines

    async def _reexec(self, side: str) -> None:
        try:
            entry = self.review_items[self.index]
            eval_entry = self.eval_by_id.get(_id_key(entry.get("id")))
            if not eval_entry:
                return
            received_steps = _extract_received_steps(eval_entry.get("received", {}).get("trace", []))
            expected_steps = self.current_expected_steps

            if side == "model":
                idx = self.selected_indices.get("model_trace")
                if idx is None:
                    table = self.query_one("#model_trace", DataTable)
                    if table.cursor_row is not None:
                        idx = table.cursor_row
                if idx is None or idx >= len(received_steps):
                    self.post_message(ToolOutput("Select a model step first."))
                    return
                tool, params = _resolve_tool_params_received(received_steps[idx])
            else:
                idx = self.selected_indices.get("expected_trace")
                if idx is None:
                    table = self.query_one("#expected_trace", DataTable)
                    if table.cursor_row is not None:
                        idx = table.cursor_row
                if idx is None or idx >= len(expected_steps):
                    self.post_message(ToolOutput("Select an expected step first."))
                    return
                tool, params = _resolve_tool_params_expected(expected_steps[idx])

            result = await _run_tool(tool, params)
            self.post_message(ToolOutput(_stringify_result(result), markdown=None))
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            self.post_message(ToolOutput(f"Tool error: {_format_exception(e)}"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        entry = self.review_items[self.index]

        if button_id == "prev":
            self.index = (self.index - 1) % len(self.review_items)
            self.refresh_view()
            return
        if button_id == "next":
            self.index = (self.index + 1) % len(self.review_items)
            self.refresh_view()
            return
        if button_id == "apply":
            entry = self.review_items[self.index]
            eval_entry = self.eval_by_id.get(_id_key(entry.get("id")))
            if eval_entry:
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
                combined = self.alpha * tf + (1 - self.alpha) * cf
                _apply_manual_scores(entry, tf, cf, combined)
                self.refresh_view()
            return
        if button_id == "accept":
            _apply_auto_scores(entry)
            self.refresh_view()
            return
        if button_id == "exclude":
            _apply_skip(entry)
            self.refresh_view()
            return
        if button_id == "save":
            self._save()
            return
        if button_id == "quit":
            self._save()
            self.exit()
            return
        if button_id == "reexec_model":
            self.run_worker(self._reexec("model"), exclusive=True, exit_on_error=False, group="reexec")
            return
        if button_id == "reexec_expected":
            self.run_worker(self._reexec("expected"), exclusive=True, exit_on_error=False, group="reexec")
            return
        if button_id == "expected_require":
            idx = self.selected_indices.get("expected_trace")
            if idx is None:
                return
            overrides, added = _get_expected_overrides(entry)
            if idx < self.current_expected_raw_len:
                overrides[str(idx)] = True
            else:
                added_idx = idx - self.current_expected_raw_len
                if 0 <= added_idx < len(added):
                    added[added_idx]["required"] = True
            self.refresh_view()
            return
        if button_id == "expected_optional":
            idx = self.selected_indices.get("expected_trace")
            if idx is None:
                return
            overrides, added = _get_expected_overrides(entry)
            if idx < self.current_expected_raw_len:
                overrides[str(idx)] = False
            else:
                added_idx = idx - self.current_expected_raw_len
                if 0 <= added_idx < len(added):
                    added[added_idx]["required"] = False
                    added[added_idx]["is_answer"] = False
            self.refresh_view()
            return
        if button_id == "expected_add_required":
            idx = self.selected_indices.get("model_trace")
            if idx is None or idx >= len(self.current_received_steps):
                self._set_output("Select a model trace row to import as required.")
                return
            overrides, added = _get_expected_overrides(entry)
            step = _expected_step_from_received(self.current_received_steps[idx])
            added.append(step)
            self.refresh_view()
            return
        if button_id == "add_comment":
            self._open_comment_modal()
            return

        if button_id.startswith("label_"):
            _, category, side, label = button_id.split("_", 3)
            table_id = f"{'model' if side == 'received' else 'expected'}_{'trace' if category == 'trace' else 'triples'}"
            idx = self.selected_indices.get(table_id)
            if idx is None:
                return
            if label == "clear":
                _set_label(entry, category, side, idx, None)
            else:
                _set_label(entry, category, side, idx, label)
            self.refresh_view()

    def _save(self) -> None:
        self.scores["per_question"] = self.per_question
        summary = summarize_results(self.per_question, self.scores.get("metadata", {}), include_anomalies=False, compute_pvalues=False)
        self.scores["metadata"] = summary["metadata"]
        self.scores["summary"] = summary["summary"]
        fmt = infer_format(self.output_path, self.format_hint)
        dump_data(self.scores, self.output_path, fmt)
        self._set_output(f"Saved results to {self.output_path}")

    def on_tool_output(self, message: ToolOutput) -> None:
        self._set_output(message.text, markdown=message.markdown)

    def _set_output(self, text: str, markdown: bool | None = False) -> None:
        md = self.query_one("#output_md", Markdown)
        txt = self.query_one("#output_text", TextArea)
        if markdown is None:
            use_markdown = self._looks_like_markdown(text)
        else:
            use_markdown = bool(markdown)
        if use_markdown:
            md.display = True
            txt.display = False
            md.update(self._normalize_markdown(text))
        else:
            md.display = False
            txt.display = True
            txt.text = text

    @staticmethod
    def _looks_like_markdown(text: str) -> bool:
        preview = text.replace("\\\\", "\\").replace("\\n", "\n").replace("\\r", "\r")
        stripped = preview.lstrip()
        if not stripped:
            return False
        if stripped.startswith("#") or stripped.startswith("\\#"):
            return True
        if re.search(r"(^|\n)\\?#+\s", preview):
            return True
        if "\\#" in text:
            return True
        if "```" in text:
            return True
        if "\n| " in preview and "---" in preview:
            return True
        if "\n- " in preview or "\n* " in preview:
            return True
        if "](" in text:
            return True
        if "\\|" in text or "| " in preview:
            return True
        return False

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        # Convert escaped newlines if present
        if "\\\\" in text:
            text = text.replace("\\\\", "\\")
        if "\\n" in text or "\\r" in text:
            text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\r")
        # Unescape common markdown escapes
        text = re.sub(r"\\([#*_`>|-])", r"\1", text)
        text = text.replace("\\|", "|")
        return text



def main() -> None:
    parser = argparse.ArgumentParser(description="Manual scoring TUI (Textual)")
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
    args = parser.parse_args()

    run_manual_review(
        scores_path=args.scores,
        eval_path=args.eval,
        output_path=args.output,
        format_hint=args.format,
        anomalies_only=args.anomalies,
    )


def run_manual_review(
    scores_path: Path,
    eval_path: Path,
    output_path: Path,
    format_hint: str | None,
    anomalies_only: bool,
) -> None:
    scores = load_data(scores_path)
    eval_data = load_data(eval_path)
    app = ReviewApp(scores, eval_data, output_path, format_hint, review_all=not anomalies_only)
    app.run()


if __name__ == "__main__":
    main()
