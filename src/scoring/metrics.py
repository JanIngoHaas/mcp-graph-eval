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
    exclude_keys = {'limit', 'offset', 'executionKey', 'expandProperties'}
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
    Compute F1 score for trace comparison using LCS (Longest Common Subsequence).
    
    This is ORDER-SENSITIVE: steps must appear in the correct order to count.
    Extra steps in between are allowed but don't contribute to the match.
    
    Precision = LCS_score / len(received)  -- what fraction of received steps were useful
    Recall    = LCS_score / len(expected)  -- what fraction of expected steps were covered
    """
    # Handle edge cases
    if not expected_trace and not received_trace:
        return TraceScore(precision=1.0, recall=1.0, f1=1.0, step_details=[])
    if not expected_trace:
        return TraceScore(precision=0.0, recall=1.0, f1=0.0, step_details=[])
    if not received_trace:
        return TraceScore(precision=1.0, recall=0.0, f1=0.0, step_details=[])
    
    # Extract steps from received trace (may be nested in explain objects)
    received_steps = _extract_received_steps(received_trace)
    
    if not received_steps:
        return TraceScore(precision=1.0, recall=0.0, f1=0.0, step_details=[])
    
    n = len(expected_trace)
    m = len(received_steps)
    
    # Build similarity matrix
    sim_matrix = []
    for i, recv_step in enumerate(received_steps):
        row = []
        for j, exp_step in enumerate(expected_trace):
            score = compute_step_similarity(exp_step, recv_step)
            row.append(score.combined)
        sim_matrix.append(row)
    
    # Compute LCS with similarity scores using DP
    # dp[i][j] = best cumulative score using received[:i] and expected[:j]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    backtrack = [[None] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = sim_matrix[i-1][j-1]
            
            # Option 1: Match received[i-1] with expected[j-1]
            match_score = dp[i-1][j-1] + sim
            
            # Option 2: Skip received[i-1]
            skip_recv = dp[i-1][j]
            
            # Option 3: Skip expected[j-1]
            skip_exp = dp[i][j-1]
            
            # Take the best option (only match if similarity > 0)
            if sim > 0 and match_score >= skip_recv and match_score >= skip_exp:
                dp[i][j] = match_score
                backtrack[i][j] = 'match'
            elif skip_recv >= skip_exp:
                dp[i][j] = skip_recv
                backtrack[i][j] = 'skip_recv'
            else:
                dp[i][j] = skip_exp
                backtrack[i][j] = 'skip_exp'
    
    lcs_score = dp[m][n]
    
    # Backtrack to find matched pairs
    step_details = []
    i, j = m, n
    matched_pairs = []
    
    while i > 0 and j > 0:
        if backtrack[i][j] == 'match':
            matched_pairs.append({
                'received_idx': i - 1,
                'expected_idx': j - 1,
                'received_tool': received_steps[i-1].get('toolName', ''),
                'expected_tool': expected_trace[j-1].get('tool', ''),
                'score': sim_matrix[i-1][j-1]
            })
            i -= 1
            j -= 1
        elif backtrack[i][j] == 'skip_recv':
            i -= 1
        else:
            j -= 1
    
    matched_pairs.reverse()
    step_details = matched_pairs
    
    # Compute precision and recall
    # Precision: total matched score / number of received steps
    precision = lcs_score / m if m > 0 else 0.0
    
    # Recall: total matched score / number of expected steps
    recall = lcs_score / n if n > 0 else 0.0
    
    # Use sklearn-style F1 calculation: 2 * (p * r) / (p + r)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return TraceScore(
        precision=precision,
        recall=recall,
        f1=f1,
        step_details=step_details
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
