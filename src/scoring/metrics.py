"""
F1 Score evaluation metrics for KG agent traces and citations.

This module provides:
- Trace comparison (tool + argument similarity)
- Triple/citation comparison (set-based F1)
- Combined scoring
"""

from dataclasses import dataclass
from typing import Any
import json
import re

from Levenshtein import distance as lev_distance, ratio as lev_ratio

RDFS_LABEL_URI = "http://www.w3.org/2000/01/rdf-schema#label"
TRACE_MATCH_THRESHOLD = 0.95
_CITATION_TO_EXECUTION_RE = re.compile(
    r"Citation Key:\s*([\w-]+).*?(?:Execution Key|Explanation Key):\s*([\w-]+)",
    re.DOTALL,
)

@dataclass
class StepScore:
    """Score for a single trace step comparison."""
    tool_match: bool
    arg_similarity: float
    combined: float


@dataclass 
class TraceScore:
    """Aggregated score for trace comparison."""
    precision: float
    recall: float
    f1: float
    step_details: list[dict]


@dataclass
class TripleScore:
    """Score for triple/citation comparison."""
    precision: float
    recall: float
    f1: float
    matched: int
    expected_count: int
    received_count: int


@dataclass
class CombinedScore:
    """Combined evaluation score."""
    trace_score: TraceScore
    triple_score: TripleScore
    combined_f1: float
    alpha: float  # weight used for combination


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity between two strings.
    Returns value in [0, 1] where 1 = identical.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    return lev_ratio(s1, s2)


def _normalize_text(value: str) -> str:
    """Normalize free-form text for comparison."""
    return " ".join(value.strip().split()).lower()


def _normalize_uri(value: str) -> str:
    """Normalize a URI-ish string for comparison."""
    cleaned = value.strip()
    lowered = cleaned.lower()
    # TODO: handle prefixes better...
    if lowered in {"rdfs:label", "label"}:
        return RDFS_LABEL_URI
    return cleaned


def _normalize_path(value: str) -> str:
    """Normalize a property path."""
    if not value:
        return ""
    parts = [p.strip() for p in value.split("->") if p.strip()]
    return "->".join(_normalize_uri(p) for p in parts)


def _canonicalize_query_builder(params: dict) -> dict:
    out: dict = {}

    if "type" in params:
        out["type"] = _normalize_uri(str(params["type"]))

    if "project" in params:
        proj = [_normalize_uri(str(item)) for item in (params.get("project") or [])]
        out["project"] = sorted(proj)

    if "filters" in params:
        filters = []
        for f in params.get("filters") or []:
            if not isinstance(f, dict):
                continue
            path = _normalize_path(str(f.get("path", "")))
            op = str(f.get("operator", "")).lower()
            val = str(f.get("value", "")).strip()
            filters.append({"path": path, "operator": op, "value": val})
        filters.sort(key=lambda x: (x["path"], x["operator"], x["value"]))
        out["filters"] = filters

    return out


def _canonicalize_tool_params(tool: str, params: dict) -> dict:
    """Canonicalize tool params for comparison."""
    if not isinstance(params, dict):
        return {}

    if tool == "search":
        if "query" in params:
            return {"query": _normalize_text(str(params["query"]))}
        return {}

    if tool == "inspect":
        if "uri" in params:
            return {"uri": _normalize_uri(str(params["uri"]))}
        return {}

    if tool == "fact":
        out = {}
        for key in ["subject", "predicate", "object"]:
            if key in params:
                out[key] = _normalize_uri(str(params[key]))
        return out

    if tool == "query_builder":
        return _canonicalize_query_builder(params)

    if tool == "query":
        if "query" in params:
            return {"query": " ".join(str(params["query"]).split())}
        return {}

    return params


def stringify_value(value: Any) -> str:
    """Convert any value to a comparable string."""
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        # Sort keys for consistent ordering
        return json.dumps(value, sort_keys=True)
    elif isinstance(value, list):
        return json.dumps(value, sort_keys=True)
    else:
        return str(value)


def compute_arg_similarity(tool: str, exp_params: dict, recv_params: dict) -> float:
    """
    Compute argument similarity using hierarchical key-value matching.
    
    For each key present in either dict:
    - If in both: compute string similarity of values
    - If in only one: 0 contribution
    
    Returns average similarity across all keys.
    """
    if not exp_params and not recv_params:
        return 1.0
    if not exp_params or not recv_params:
        return 0.0
    
    all_keys = set(exp_params.keys()) | set(recv_params.keys())
    
    # Exclude metadata keys that don't affect semantics (partly -- limit and offset does, but ignored here)
    exclude_keys = {'limit', 'offset', 'executionKey', 'expandProperties', 'required', 'is_answer'}
    relevant_keys = all_keys - exclude_keys
    
    if not relevant_keys:
        return 1.0

    # Specialize on the tool type
    if tool == "search":
        # We classify as successful if prefix match (either direction) - otherwise, lev
        q1 = str(exp_params.get("query", "")).strip()
        q2 = str(recv_params.get("query", "")).strip()
        if not q1 or not q2:
            return 0.0
        if q1.startswith(q2) or q2.startswith(q1):
            return 1.0
        return normalized_levenshtein(q1, q2) 
    
    if tool == "inspect": 
        # We classify as successful if literally same URI
        uri1 = str(exp_params.get("uri", "")).strip()
        uri2 = str(recv_params.get("uri", "")).strip()
        if not uri1 or not uri2:
            return 0.0
        if uri1 == uri2:
            return 1.0
        return normalized_levenshtein(uri1, uri2)
    
    if tool == "fact":
        # We classify as successful if literals match
        subject1 = str(exp_params.get("subject", "")).strip()
        predicate1 = str(exp_params.get("predicate", "")).strip()
        object1 = str(exp_params.get("object", "")).strip()
        subject2 = str(recv_params.get("subject", "")).strip()
        predicate2 = str(recv_params.get("predicate", "")).strip()
        object2 = str(recv_params.get("object", "")).strip()
        if not subject1 or not subject2 or not predicate1 or not predicate2 or not object1 or not object2:
            return 0.0
        if subject1 == subject2 and predicate1 == predicate2:
            if object1 == "_" or (object1 == object2):
                return 1.0            
        return normalized_levenshtein(subject1, subject2)

    if tool == "query_builder":

        total_score = 0.0

        # We classify query builder as successful if 
        # 1. Same type (Uris must match exactly)
        # 2. Same filters (args must match exactly)
        # 3. Same project (URIs must match exactly)

        # 1
        if "type" in exp_params and "type" in recv_params:
            if exp_params["type"].strip() == recv_params["type"].strip():
                total_score += 1

        # 2
        if "filters" in exp_params and "filters" in recv_params:
            val1 = stringify_value(exp_params["filters"])
            val2 = stringify_value(recv_params["filters"])
            total_score += normalized_levenshtein(val1, val2)

        # 3
        if "project" in exp_params and "project" in recv_params:
            val1 = stringify_value(exp_params["project"])
            val2 = stringify_value(recv_params["project"])
            total_score += normalized_levenshtein(val1, val2)

        return total_score / 3

    total_score = 0.0
    for key in relevant_keys:
        if key in exp_params and key in recv_params:
            val1 = stringify_value(exp_params[key])
            val2 = stringify_value(recv_params[key])
            total_score += normalized_levenshtein(val1, val2)
        # If key only in one, contributes 0
    
    return total_score / len(relevant_keys)


def compute_step_similarity(expected_step: dict, received_step: dict) -> StepScore:
    """
    Compare a single expected step with a received step.
    
    Returns StepScore with:
    - tool_match: whether tools are the same
    - arg_similarity: similarity of arguments (0-1)
    - combined: tool_match * arg_similarity
    """
    # Extract tool names (expected uses 'tool', received uses 'toolName')
    expected_tool = expected_step.get('tool', '')
    received_tool = received_step.get('toolName', '')
    
    tool_match = expected_tool == received_tool
    
    if not tool_match:
        return StepScore(tool_match=False, arg_similarity=0.0, combined=0.0)
    
    # Extract parameters
    # Expected format varies by tool type
    expected_params = _extract_expected_params(expected_step)
    received_params = received_step.get('toolParams', {})

    expected_params = _canonicalize_tool_params(expected_tool, expected_params)
    received_params = _canonicalize_tool_params(received_tool, received_params)

    arg_sim = compute_arg_similarity(expected_tool, expected_params, received_params)
    
    return StepScore(
        tool_match=True,
        arg_similarity=arg_sim,
        combined=arg_sim  # Since tool matched, combined = arg_sim
    )


def extract_cited_execution_keys(runtime_trace: list[dict]) -> set[str]:
    """
    Extract execution/explanation keys that immediately follow a citation key.
    """
    if not runtime_trace:
        return set()

    text = json.dumps(runtime_trace, default=str)
    return {execution_key for _, execution_key in _CITATION_TO_EXECUTION_RE.findall(text)}


def match_trace(expected_trace: list[dict], received_trace: list[dict]) -> dict[int, list[int]]:
    """
    Match received trace steps to expected steps.

    Returns a mapping: received_idx -> [expected_idx, ...] for all matches.
    A match is a tool+arg similarity >= TRACE_MATCH_THRESHOLD.
    """
    expected_steps = list(expected_trace)
    received_steps = _extract_received_steps(received_trace)
    matches: dict[int, list[int]] = {}

    for r_idx, recv_step in enumerate(received_steps):
        matched = []
        for e_idx, exp_step in enumerate(expected_steps):
            score = compute_step_similarity(exp_step, recv_step)
            if score.combined >= TRACE_MATCH_THRESHOLD:
                matched.append(e_idx)
        if matched:
            matches[r_idx] = matched
    return matches


def _extract_expected_params(step: dict) -> dict:
    """Extract parameters from expected step format."""
    tool = step.get('tool', '')
    params = {}
    
    if tool == 'search':
        if 'query' in step:
            params['query'] = step['query']
    elif tool == 'inspect':
        if 'uri' in step:
            params['uri'] = step['uri']
    elif tool == 'fact':
        for key in ['subject', 'predicate', 'object']:
            if key in step:
                params[key] = step[key]
    elif tool == 'query_builder':
        for key in ['type', 'filters', 'project']:
            if key in step:
                params[key] = step[key]
    elif tool == 'query':
        if 'query' in step:
            params['query'] = step['query']
    
    return params


def compute_trace_f1(
    expected_trace: list[dict],
    received_trace: list[dict],
    ignored_execution_keys: set[str] | None = None,
) -> TraceScore:
    """
    Compute F1 score for trace comparison using order-agnostic matching.

    - Expected steps can be tagged with `required`.
    - If any step is tagged, only required steps are scored.
    - Extra received steps are ignored unless they are plausible matches.
    """
    expected_steps = list(expected_trace)
    received_steps = _extract_received_steps(received_trace)
    ignored_execution_keys = ignored_execution_keys or set()

    if not expected_steps or not received_steps:
        return TraceScore(precision=0.0, recall=0.0, f1=0.0, step_details=[])

    required_indices = {
        idx
        for idx, step in enumerate(expected_steps)
        if step.get("required") and not step.get("is_answer")
    }
    optional_indices = set(range(len(expected_steps))) - required_indices

    matches = match_trace(expected_trace, received_trace)
    matched_expected = {e_idx for e_list in matches.values() for e_idx in e_list}

    optional_only_received = {
        r_idx
        for r_idx, exp_list in matches.items()
        if exp_list and all(e_idx in optional_indices for e_idx in exp_list)
    }
    ignored_received = {
        idx
        for idx, step in enumerate(received_steps)
        if step.get("executionKey") in ignored_execution_keys
    } | optional_only_received
    effective_received = len(received_steps) - len(ignored_received)
    precision = len(
        [
            r_idx
            for r_idx, exp_list in matches.items()
            if r_idx not in ignored_received and any(e_idx in required_indices for e_idx in exp_list)
        ]
    ) / effective_received if effective_received else 0.0
    matched_required = {
        e_idx
        for r_idx, exp_list in matches.items()
        if r_idx not in ignored_received
        for e_idx in exp_list
        if e_idx in required_indices
    }
    recall = len(matched_required) / len(required_indices) if required_indices else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    matched_pairs = []
    for r_idx, exp_list in matches.items():
        for e_idx in exp_list:
            matched_pairs.append({
                "received_idx": r_idx,
                "expected_idx": e_idx,
                "received_tool": received_steps[r_idx].get("toolName", ""),
                "expected_tool": expected_steps[e_idx].get("tool", ""),
                "score": 1.0,
            })

    return TraceScore(
        precision=precision,
        recall=recall,
        f1=f1,
        step_details=matched_pairs,
    )


def _extract_received_steps(received_trace: list) -> list[dict]:
    """
    Extract step objects from received trace.
    
    Received trace may contain explain objects with nested steps.
    """
    steps = []
    for item in received_trace:
        if 'steps' in item:
            # This is an explain object, extract its steps
            steps.extend(item.get('steps', []))
        elif 'toolName' in item:
            # This is already a step
            steps.append(item)
    return steps


def triple_to_tuple(triple: dict) -> tuple:
    """Convert a triple dict to a hashable tuple."""
    return (
        triple.get('subject', ''),
        triple.get('predicate', ''),
        triple.get('object', '')
    )


def match_triple(expected_triples: list[dict], received_triples: list[dict]) -> dict[int, list[int]]:
    """
    Match received triples to expected triples by exact tuple match.

    Returns a mapping: received_idx -> [expected_idx, ...] for all matches.
    """
    expected_tuples = [triple_to_tuple(t) for t in expected_triples]
    received_tuples = [triple_to_tuple(t) for t in received_triples]
    matches: dict[int, list[int]] = {}

    index_by_tuple: dict[tuple, list[int]] = {}
    for idx, tup in enumerate(expected_tuples):
        index_by_tuple.setdefault(tup, []).append(idx)

    for r_idx, tup in enumerate(received_tuples):
        exp_indices = index_by_tuple.get(tup)
        if exp_indices:
            matches[r_idx] = list(exp_indices)
    return matches


def compute_triple_f1(expected_triples: list[dict], received_triples: list[dict]) -> TripleScore:
    """
    Compute F1 score for triple/citation comparison using exact matches.

    Uses exact matching on (subject, predicate, object) tuples.
    """
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


def compute_combined_score(
    expected_trace: list[dict],
    received_trace: list[dict],
    expected_triples: list[dict],
    received_triples: list[dict],
    alpha: float = 0.5,
    ignored_execution_keys: set[str] | None = None,
) -> CombinedScore:
    """
    Compute combined F1 score from trace and triple scores.
    
    Args:
        alpha: Weight for trace score (1-alpha for triples)
               0.5 = equal weight
               Lower = prioritize correct answers over methodology
    """
    trace_score = compute_trace_f1(expected_trace, received_trace, ignored_execution_keys)
    triple_score = compute_triple_f1(expected_triples, received_triples)
    
    combined_f1 = alpha * trace_score.f1 + (1 - alpha) * triple_score.f1
    
    return CombinedScore(
        trace_score=trace_score,
        triple_score=triple_score,
        combined_f1=combined_f1,
        alpha=alpha
    )
