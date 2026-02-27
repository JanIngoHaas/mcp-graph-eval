"""Scoring helpers for triple metrics."""

from __future__ import annotations

from dataclasses import dataclass
RDF_TYPE_PREDICATE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


@dataclass
class TripleScore:
    precision: float
    recall: float
    f1: float
    matched: int
    expected_count: int
    received_count: int


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
