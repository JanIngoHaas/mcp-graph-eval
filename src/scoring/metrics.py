"""Scoring helpers for triple metrics and trace-template classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


TRACE_IGNORED_TOOLS = {"cite", "explain", "finalize"}
RDF_TYPE_PREDICATE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


TRACE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "direct": {
        "direct_no_tools": [],
        "direct_fact_basic": ["search", "inspect", "fact"],
        "direct_fact_with_property_inspect": ["search", "inspect", "inspect", "fact"],
        "direct_fact_no_anchor_inspect": ["search", "fact"],
        "direct_fact_double": ["search", "inspect", "fact", "fact"],
        "direct_fact_double_with_property_inspects": ["search", "inspect", "inspect", "fact", "inspect", "fact"],
        "direct_query_builder": ["search", "inspect", "query_builder"],
        "direct_query_builder_with_discovery": ["search", "inspect", "inspect", "query_builder"],
        "direct_query_builder_minimal": ["search", "query_builder"],
        "direct_query_builder_with_search_refinement": ["search", "inspect", "search", "query_builder"],
    },
    "hop": {
        "hop_no_tools": [],
        "hop_fact_basic": ["search", "inspect", "fact", "inspect", "fact"],
        "hop_fact_with_property_inspect": ["search", "inspect", "fact", "inspect", "inspect", "fact"],
        "hop_fact_two_answers": ["search", "inspect", "fact", "inspect", "fact", "fact"],
        "hop_fact_two_answers_with_property_inspects": [
            "search", "inspect", "fact", "inspect", "inspect", "fact", "inspect", "fact",
        ],
        "hop_fact_three_answers": ["search", "inspect", "fact", "inspect", "fact", "fact", "fact"],
        "hop_fact_bridge_then_query_builder": ["search", "inspect", "fact", "query_builder"],
        "hop_query_builder_direct": ["search", "inspect", "query_builder"],
        "hop_query_builder_with_discovery": ["search", "inspect", "inspect", "query_builder"],
    },
    "impossible": {
        "impossible_no_tools": [],
        "impossible_fact_check": ["search", "inspect", "fact"],
        "impossible_fact_with_property_inspect": ["search", "inspect", "inspect", "fact"],
        "impossible_fact_multi_check": ["search", "inspect", "fact", "fact"],
        "impossible_query_builder_check": ["search", "inspect", "query_builder"],
        "impossible_query_builder_with_discovery": ["search", "inspect", "inspect", "query_builder"],
        "impossible_query_builder_with_search_refinement": ["search", "inspect", "search", "query_builder"],
        "impossible_query_builder_then_fact_check": ["search", "inspect", "query_builder", "fact"],
    },
    "query_builder": {
        "qb_no_tools": [],
        "qb_basic": ["search", "inspect", "query_builder"],
        "qb_search_then_query_builder": ["search", "query_builder"],
        "qb_with_search_refinement": ["search", "inspect", "search", "query_builder"],
        "qb_with_discovery": ["search", "inspect", "inspect", "query_builder"],
        "qb_with_deep_discovery": ["search", "inspect", "inspect", "inspect", "query_builder"],
        "qb_with_deeper_discovery": ["search", "inspect", "inspect", "inspect", "inspect", "query_builder"],
        "qb_with_discovery_then_search_refinement": ["search", "inspect", "inspect", "search", "query_builder"],
        "qb_with_dual_search_refinement": ["search", "inspect", "search", "search", "query_builder"],
        "qb_inspect_only": ["inspect", "inspect", "query_builder"],
        "qb_inspect_then_search": ["inspect", "inspect", "search", "query_builder"],
        "qb_post_answer_validation": ["search", "inspect", "query_builder", "inspect"],
        "qb_post_answer_validation_verbose": ["search", "inspect", "query_builder", "inspect", "inspect"],
    },
}


@dataclass
class TripleScore:
    precision: float
    recall: float
    f1: float
    matched: int
    expected_count: int
    received_count: int


@dataclass
class TraceTemplateSuggestion:
    template_id: str
    template_tools: list[str]
    sequence_tools: list[str]
    similarity: float
    exact_match: bool


def _extract_received_steps(received_trace: list) -> list[dict]:
    """Extract step objects from received explanation trace."""
    steps: list[dict] = []
    for item in received_trace or []:
        if isinstance(item, dict) and isinstance(item.get("steps"), list):
            steps.extend(step for step in item.get("steps", []) if isinstance(step, dict))
        elif isinstance(item, dict) and "toolName" in item:
            steps.append(item)
    return steps


def extract_tool_sequence(received_trace: list) -> list[str]:
    """Extract normalized tool-name sequence from model trace."""
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


def _similarity(seq: list[str], template: list[str]) -> float:
    if not seq and not template:
        return 1.0
    if not seq or not template:
        return 0.0
    lcs = _lcs_len(seq, template)
    return (2.0 * lcs) / (len(seq) + len(template))


def suggest_trace_template(qtype: str, received_trace: list) -> TraceTemplateSuggestion:
    """Suggest closest trace template for a question type."""
    seq = extract_tool_sequence(received_trace)
    templates = TRACE_TEMPLATES.get(qtype) or TRACE_TEMPLATES.get("direct") or {}
    if not templates:
        return TraceTemplateSuggestion(
            template_id="none",
            template_tools=[],
            sequence_tools=seq,
            similarity=0.0,
            exact_match=False,
        )

    best_id = ""
    best_tools: list[str] = []
    best_score = -1.0
    for template_id, template_tools in templates.items():
        score = _similarity(seq, template_tools)
        if score > best_score:
            best_id = template_id
            best_tools = list(template_tools)
            best_score = score

    return TraceTemplateSuggestion(
        template_id=best_id,
        template_tools=best_tools,
        sequence_tools=seq,
        similarity=max(best_score, 0.0),
        exact_match=(seq == best_tools),
    )


def _extract_expected_params(step: dict) -> dict:
    """Extract parameters from expected step format for optional UI use."""
    tool = step.get("tool", "")
    params: dict[str, Any] = {}

    if tool == "search":
        if "query" in step:
            params["query"] = step["query"]
    elif tool == "inspect":
        if "uri" in step:
            params["uri"] = step["uri"]
    elif tool == "fact":
        for key in ["subject", "predicate", "object"]:
            if key in step:
                params[key] = step[key]
    elif tool == "query_builder":
        for key in ["type", "filters", "project"]:
            if key in step:
                params[key] = step[key]
    return params


def triple_to_tuple(triple: dict) -> tuple:
    return (
        str(triple.get("subject", "")),
        str(triple.get("predicate", "")),
        str(triple.get("object", "")),
    )


def _filter_scorable_triples(triples: list[dict]) -> list[dict]:
    """Exclude non-scoring triples from metric calculations."""
    return [t for t in triples if str(t.get("predicate", "")) != RDF_TYPE_PREDICATE]


def match_triple(expected_triples: list[dict], received_triples: list[dict]) -> dict[int, list[int]]:
    expected_filtered = _filter_scorable_triples(expected_triples)
    received_filtered = _filter_scorable_triples(received_triples)
    expected_tuples = [triple_to_tuple(t) for t in expected_filtered]
    received_tuples = [triple_to_tuple(t) for t in received_filtered]
    index_by_tuple: dict[tuple, list[int]] = {}
    for idx, tup in enumerate(expected_tuples):
        index_by_tuple.setdefault(tup, []).append(idx)

    matches: dict[int, list[int]] = {}
    for r_idx, tup in enumerate(received_tuples):
        exp_indices = index_by_tuple.get(tup)
        if exp_indices:
            matches[r_idx] = list(exp_indices)
    return matches


def compute_triple_f1(expected_triples: list[dict], received_triples: list[dict]) -> TripleScore:
    expected_triples = _filter_scorable_triples(expected_triples)
    received_triples = _filter_scorable_triples(received_triples)
    if not expected_triples and not received_triples:
        return TripleScore(precision=1.0, recall=1.0, f1=1.0, matched=0, expected_count=0, received_count=0)
    if not expected_triples:
        return TripleScore(precision=0.0, recall=1.0, f1=0.0, matched=0, expected_count=0, received_count=len(received_triples))
    if not received_triples:
        return TripleScore(precision=1.0, recall=0.0, f1=0.0, matched=0, expected_count=len(expected_triples), received_count=0)

    matches = match_triple(expected_triples, received_triples)
    matched_received = set(matches.keys())
    matched_expected = {e_idx for e_list in matches.values() for e_idx in e_list}

    precision = len(matched_received) / len(received_triples) if received_triples else 0.0
    recall = len(matched_expected) / len(expected_triples) if expected_triples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return TripleScore(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=len(matched_received),
        expected_count=len(expected_triples),
        received_count=len(received_triples),
    )


def stable_json(value: Any) -> str:
    """Best-effort stable JSON string for displaying/debugging payloads."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(value)
