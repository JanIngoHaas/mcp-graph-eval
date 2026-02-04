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

from Levenshtein import distance as lev_distance
from sklearn.metrics import precision_score, recall_score, f1_score

RDFS_LABEL_URI = "http://www.w3.org/2000/01/rdf-schema#label"
TRACE_MATCH_THRESHOLD = 0.8


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
    
    max_len = max(len(s1), len(s2))
    distance = lev_distance(s1, s2)
    return 1.0 - (distance / max_len)


def _normalize_text(value: str) -> str:
    """Normalize free-form text for comparison."""
    return " ".join(value.strip().split()).lower()


def _normalize_uri(value: str) -> str:
    """Normalize a URI-ish string for comparison."""
    return value.strip()


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
        proj = []
        for item in params.get("project") or []:
            if isinstance(item, str) and item.lower() in {"label", "rdfs:label"}:
                proj.append(RDFS_LABEL_URI)
            else:
                proj.append(_normalize_uri(str(item)))
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


def compute_arg_similarity(params1: dict, params2: dict) -> float:
    """
    Compute argument similarity using hierarchical key-value matching.
    
    For each key present in either dict:
    - If in both: compute string similarity of values
    - If in only one: 0 contribution
    
    Returns average similarity across all keys.
    """
    if not params1 and not params2:
        return 1.0
    if not params1 or not params2:
        return 0.0
    
    all_keys = set(params1.keys()) | set(params2.keys())
    
    # Exclude metadata keys that don't affect semantics
    exclude_keys = {'limit', 'offset', 'executionKey', 'expandProperties', 'required', 'is_answer'}
    relevant_keys = all_keys - exclude_keys
    
    if not relevant_keys:
        return 1.0
    
    total_score = 0.0
    for key in relevant_keys:
        if key in params1 and key in params2:
            val1 = stringify_value(params1[key])
            val2 = stringify_value(params2[key])
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
    
    arg_sim = compute_arg_similarity(expected_params, received_params)
    
    return StepScore(
        tool_match=True,
        arg_similarity=arg_sim,
        combined=arg_sim  # Since tool matched, combined = arg_sim
    )


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


def compute_trace_f1(expected_trace: list[dict], received_trace: list[dict]) -> TraceScore:
    """
    Compute F1 score for trace comparison using order-agnostic matching.

    - Expected steps can be tagged with `required`.
    - If any step is tagged, only required steps are scored.
    - Extra received steps are ignored unless they are plausible matches.
    """
    # Expected steps include required + optional; recall is computed on required only.
    expected_steps = list(expected_trace)
    required_indices = {idx for idx, step in enumerate(expected_steps) if step.get("required")}

    # Extract steps from received trace (may be nested in explain objects)
    received_steps = _extract_received_steps(received_trace)

    # Handle edge cases
    if not expected_steps:
        return TraceScore(precision=0.0, recall=0.0, f1=0.0, step_details=[])
    if not received_steps:
        return TraceScore(precision=0.0, recall=0.0, f1=0.0, step_details=[])

    n = len(expected_steps)
    n_required = len(required_indices)
    m = len(received_steps)

    # Build similarity matrix
    sim_matrix = []
    for i, recv_step in enumerate(received_steps):
        row = []
        for j, exp_step in enumerate(expected_steps):
            score = compute_step_similarity(exp_step, recv_step)
            row.append(score.combined)
        sim_matrix.append(row)

    # Greedy maximum matching on similarity scores (order-agnostic)
    pairs = []
    for i in range(m):
        for j in range(n):
            sim = sim_matrix[i][j]
            if sim >= TRACE_MATCH_THRESHOLD:
                pairs.append((sim, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_score = 0.0
    matched_required_score = 0.0
    used_recv = set()
    used_exp = set()
    matched_pairs = []

    for sim, i, j in pairs:
        if i in used_recv or j in used_exp:
            continue
        used_recv.add(i)
        used_exp.add(j)
        matched_score += sim
        if j in required_indices:
            matched_required_score += sim
        matched_pairs.append({
            "received_idx": i,
            "expected_idx": j,
            "received_tool": received_steps[i].get("toolName", ""),
            "expected_tool": expected_steps[j].get("tool", ""),
            "score": sim,
        })

    # Precision penalizes extra steps not matching expected (required or optional).
    precision = matched_score / m if m > 0 else 0.0
    recall = matched_required_score / n_required if n_required > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

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


def compute_triple_f1(expected_triples: list[dict], received_triples: list[dict]) -> TripleScore:
    """
    Compute F1 score for triple/citation comparison using sklearn.
    
    Uses exact matching on (subject, predicate, object) tuples.
    Converts to binary classification: each unique triple is a class,
    y_true = 1 if in expected, y_pred = 1 if in received.
    """
    # Handle edge cases
    if not expected_triples and not received_triples:
        return TripleScore(precision=1.0, recall=1.0, f1=1.0, matched=0, expected_count=0, received_count=0)
    if not expected_triples:
        return TripleScore(precision=0.0, recall=1.0, f1=0.0, matched=0, expected_count=0, received_count=len(received_triples))
    if not received_triples:
        return TripleScore(precision=1.0, recall=0.0, f1=0.0, matched=0, expected_count=len(expected_triples), received_count=0)
    
    # Convert to sets of tuples
    expected_set = {triple_to_tuple(t) for t in expected_triples}
    received_set = {triple_to_tuple(t) for t in received_triples}
    
    # Find all unique triples (universe)
    all_triples = list(expected_set | received_set)
    
    # Create binary vectors
    y_true = [1 if t in expected_set else 0 for t in all_triples]
    y_pred = [1 if t in received_set else 0 for t in all_triples]
    
    # Use sklearn for precision/recall/f1
    # zero_division=1.0 handles case where there are no positive predictions
    precision = precision_score(y_true, y_pred, zero_division=1.0)
    recall = recall_score(y_true, y_pred, zero_division=1.0)
    f1 = f1_score(y_true, y_pred, zero_division=1.0)
    
    matched_count = len(expected_set & received_set)
    
    return TripleScore(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=matched_count,
        expected_count=len(expected_set),
        received_count=len(received_set)
    )


def compute_combined_score(
    expected_trace: list[dict],
    received_trace: list[dict],
    expected_triples: list[dict],
    received_triples: list[dict],
    alpha: float = 0.5
) -> CombinedScore:
    """
    Compute combined F1 score from trace and triple scores.
    
    Args:
        alpha: Weight for trace score (1-alpha for triples)
               0.5 = equal weight
               Lower = prioritize correct answers over methodology
    """
    trace_score = compute_trace_f1(expected_trace, received_trace)
    triple_score = compute_triple_f1(expected_triples, received_triples)
    
    combined_f1 = alpha * trace_score.f1 + (1 - alpha) * triple_score.f1
    
    return CombinedScore(
        trace_score=trace_score,
        triple_score=triple_score,
        combined_f1=combined_f1,
        alpha=alpha
    )
