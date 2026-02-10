from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from copy import deepcopy

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS
from rdflib.plugins.stores.sparqlstore import SPARQLStore


@dataclass
class AmbiguityExpansion:
    expected_triples: list[dict]
    expected_trace: list[dict] | None = None
    qtype_override: str | None = None
    reason: str = ""
    label: str = ""
    candidate_count: int = 0


def _normalize_label(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def _extract_received_steps(received_trace: list) -> list[dict]:
    steps: list[dict] = []
    for item in received_trace or []:
        if not isinstance(item, dict):
            continue
        nested = item.get("steps")
        if isinstance(nested, list):
            for step in nested:
                if isinstance(step, dict) and step.get("toolName"):
                    steps.append(step)
        elif item.get("toolName"):
            steps.append(item)
    return steps


def _expected_from_received(step: dict, required: bool, is_answer: bool) -> dict:
    tool = str(step.get("toolName") or "")
    params = step.get("toolParams") if isinstance(step.get("toolParams"), dict) else {}
    expected: dict = {
        "tool": tool,
        "required": required,
        "is_answer": is_answer,
    }
    if tool == "search" and "query" in params:
        expected["query"] = params["query"]
    elif tool == "inspect" and "uri" in params:
        expected["uri"] = params["uri"]
    elif tool == "fact":
        for key in ("subject", "predicate", "object"):
            if key in params:
                expected[key] = params[key]
    elif tool == "query_builder":
        for key in ("type", "filters", "project"):
            if key in params:
                expected[key] = params[key]
    elif tool == "query" and "query" in params:
        expected["query"] = params["query"]
    return expected


def build_effective_ambiguity_trace(
    original_expected_trace: list[dict],
    received_trace: list[dict],
    ignored_execution_keys: set[str] | None,
    ambiguity_expansion: AmbiguityExpansion | None,
) -> list[dict]:
    """
    Build effective expected trace for ambiguity cases.

    Strategy:
    - Start from ambiguity-provided canonical trace (fallback to original).
    - If a query_builder answer step exists in received trace, replace the
      required answer query_builder step with that concrete step.
    - Add all received steps as optional expectations to avoid penalizing
      exploratory path variance in ambiguous questions.
    """
    if ambiguity_expansion is None or not ambiguity_expansion.expected_trace:
        return list(original_expected_trace)

    expected_steps = [deepcopy(step) for step in ambiguity_expansion.expected_trace]
    received_steps = _extract_received_steps(received_trace)
    ignored_execution_keys = ignored_execution_keys or set()

    required_qb_idx = None
    for idx, step in enumerate(expected_steps):
        if (
            step.get("tool") == "query_builder"
            and step.get("required")
            and step.get("is_answer")
        ):
            required_qb_idx = idx
            break

    qb_steps = [step for step in received_steps if step.get("toolName") == "query_builder"]
    preferred_qb = None
    for step in qb_steps:
        if step.get("executionKey") in ignored_execution_keys:
            preferred_qb = step
            break
    if preferred_qb is None and qb_steps:
        preferred_qb = qb_steps[-1]

    if required_qb_idx is not None and preferred_qb is not None:
        expected_steps[required_qb_idx] = _expected_from_received(
            preferred_qb,
            required=True,
            is_answer=True,
        )

    for step in received_steps:
        expected_steps.append(
            _expected_from_received(
                step,
                required=False,
                is_answer=False,
            )
        )

    return expected_steps


class AmbiguityResolver:
    """
    Resolves label-ambiguity at scoring-time by expanding expected triples.

    The resolver only activates for direct/hop/impossible questions where
    the expected trace indicates an anchor label that maps to multiple entities
    of the same anchor type.
    """

    def __init__(
        self,
        kg_ttl_path: Path | None = None,
        sparql_endpoint: str | None = None,
        include_type_triples: bool = True,
    ):
        if kg_ttl_path is None and not sparql_endpoint:
            raise ValueError("Either kg_ttl_path or sparql_endpoint must be provided")

        self.kg_ttl_path = Path(kg_ttl_path) if kg_ttl_path is not None else None
        self.sparql_endpoint = sparql_endpoint
        self.include_type_triples = include_type_triples
        if self.kg_ttl_path is not None:
            self.graph = Graph()
            self.graph.parse(self.kg_ttl_path, format="turtle")
            self.source_description = str(self.kg_ttl_path)
        else:
            store = SPARQLStore(self.sparql_endpoint)
            self.graph = Graph(store, identifier=None)
            self.source_description = self.sparql_endpoint or "unknown-endpoint"
        self._subjects_cache: dict[tuple[str, str], list[URIRef]] = {}
        self._primary_type_cache: dict[str, URIRef | None] = {}

    def maybe_expand(self, item: dict) -> AmbiguityExpansion | None:
        qtype = item.get("qtype") or "unknown"
        if qtype not in {"direct", "hop", "impossible"}:
            return None

        expected = item.get("expected", {})
        trace = list(expected.get("trace") or [])
        if not trace:
            return None

        label = self._extract_anchor_label(trace)
        anchor_uri_str = self._extract_anchor_uri(trace)
        if not label or not anchor_uri_str:
            return None

        anchor_uri = URIRef(anchor_uri_str)
        anchor_type = self._primary_type(anchor_uri)
        if anchor_type is None:
            return None

        candidates = self._subjects_by_label_type(label, anchor_type)
        if len(candidates) <= 1:
            return None

        if qtype == "direct":
            return self._expand_direct(trace, label, candidates, anchor_type)
        if qtype == "hop":
            return self._expand_hop(trace, label, candidates, anchor_type)
        return self._expand_impossible(trace, label, candidates, anchor_type)

    def _extract_anchor_label(self, trace: list[dict]) -> str | None:
        for step in trace:
            if step.get("tool") == "search" and step.get("required") and step.get("query"):
                return str(step.get("query"))
        for step in trace:
            if step.get("tool") == "search" and step.get("query"):
                return str(step.get("query"))
        return None

    def _extract_anchor_uri(self, trace: list[dict]) -> str | None:
        for step in trace:
            if step.get("tool") == "inspect" and step.get("required") and step.get("uri"):
                return str(step.get("uri"))
        return None

    def _primary_type(self, subject: URIRef) -> URIRef | None:
        key = str(subject)
        if key in self._primary_type_cache:
            return self._primary_type_cache[key]

        subject_types = [t for t in self.graph.objects(subject, RDF.type) if isinstance(t, URIRef)]
        first = subject_types[0] if subject_types else None
        self._primary_type_cache[key] = first
        return first

    def _subjects_by_label_type(self, label: str, type_uri: URIRef) -> list[URIRef]:
        cache_key = (label, str(type_uri))
        if cache_key in self._subjects_cache:
            return self._subjects_cache[cache_key]

        lit = Literal(label)
        exact: set[URIRef] = set()
        for subject in self.graph.subjects(RDFS.label, lit):
            if not isinstance(subject, URIRef):
                continue
            if any(obj == type_uri for obj in self.graph.objects(subject, RDF.type)):
                exact.add(subject)
        if exact:
            out = sorted(exact, key=str)
            self._subjects_cache[cache_key] = out
            return out

        # Fallback: normalized label comparison.
        norm = _normalize_label(label)
        fuzzy: set[URIRef] = set()
        for subject in self.graph.subjects(RDF.type, type_uri):
            if not isinstance(subject, URIRef):
                continue
            for lbl in self.graph.objects(subject, RDFS.label):
                if _normalize_label(str(lbl)) == norm:
                    fuzzy.add(subject)
                    break
        out = sorted(fuzzy, key=str)
        self._subjects_cache[cache_key] = out
        return out

    def _answer_predicates(self, trace: list[dict]) -> list[URIRef]:
        predicates: list[URIRef] = []
        seen = set()
        for step in trace:
            if step.get("tool") != "fact":
                continue
            if not step.get("is_answer"):
                continue
            pred = step.get("predicate")
            if not pred:
                continue
            pred_uri = URIRef(str(pred))
            if pred_uri in seen:
                continue
            seen.add(pred_uri)
            predicates.append(pred_uri)
        return predicates

    def _query_builder_trace(
        self,
        label: str,
        entity_type: URIRef,
        project_paths: list[str],
    ) -> list[dict]:
        return [
            {
                "tool": "search",
                "query": label,
                "required": False,
                "is_answer": False,
            },
            {
                "tool": "query_builder",
                "type": str(entity_type),
                "filters": [{"path": "rdfs:label", "operator": "contains", "value": label}],
                "project": project_paths,
                "required": True,
                "is_answer": True,
            },
        ]

    def _triples_for_predicates(
        self,
        subjects: Iterable[URIRef],
        predicates: Iterable[URIRef],
    ) -> list[dict]:
        out: set[tuple[str, str, str]] = set()

        for subject in subjects:
            has_answer_triple = False
            for predicate in predicates:
                for obj in self.graph.objects(subject, predicate):
                    out.add((str(subject), str(predicate), str(obj)))
                    has_answer_triple = True

            if self.include_type_triples and has_answer_triple:
                for t in self.graph.objects(subject, RDF.type):
                    out.add((str(subject), str(RDF.type), str(t)))

        sorted_out = sorted(out)
        return [{"subject": s, "predicate": p, "object": o} for s, p, o in sorted_out]

    def _expand_direct(
        self,
        trace: list[dict],
        label: str,
        candidates: list[URIRef],
        anchor_type: URIRef,
    ) -> AmbiguityExpansion | None:
        predicates = self._answer_predicates(trace)
        if not predicates:
            return None

        triples = self._triples_for_predicates(candidates, predicates)
        if not triples:
            return None

        project_paths = [str(RDFS.label)] + [str(p) for p in predicates]
        expected_trace = self._query_builder_trace(label, anchor_type, project_paths)

        return AmbiguityExpansion(
            expected_triples=triples,
            expected_trace=expected_trace,
            reason="direct_label_ambiguous",
            label=label,
            candidate_count=len(candidates),
        )

    def _expand_hop(
        self,
        trace: list[dict],
        label: str,
        candidates: list[URIRef],
        anchor_type: URIRef,
    ) -> AmbiguityExpansion | None:
        bridge_predicate: URIRef | None = None
        for step in trace:
            if step.get("tool") != "fact":
                continue
            if step.get("is_answer"):
                continue
            pred = step.get("predicate")
            obj = step.get("object")
            if pred and isinstance(obj, str) and obj.startswith("http"):
                bridge_predicate = URIRef(str(pred))
                break
        if bridge_predicate is None:
            return None

        answer_predicates = self._answer_predicates(trace)
        if not answer_predicates:
            return None

        target_entities: set[URIRef] = set()
        for anchor in candidates:
            for target in self.graph.objects(anchor, bridge_predicate):
                if isinstance(target, URIRef):
                    target_entities.add(target)
        if not target_entities:
            return None

        triples = self._triples_for_predicates(sorted(target_entities, key=str), answer_predicates)
        if not triples:
            return None

        project_paths = [f"{str(bridge_predicate)} -> {str(p)}" for p in answer_predicates]
        expected_trace = self._query_builder_trace(label, anchor_type, project_paths)

        return AmbiguityExpansion(
            expected_triples=triples,
            expected_trace=expected_trace,
            reason="hop_label_ambiguous",
            label=label,
            candidate_count=len(candidates),
        )

    def _expand_impossible(
        self,
        trace: list[dict],
        label: str,
        candidates: list[URIRef],
        anchor_type: URIRef,
    ) -> AmbiguityExpansion | None:
        impossible_predicate: URIRef | None = None
        for step in trace:
            if step.get("tool") != "fact":
                continue
            pred = step.get("predicate")
            obj = step.get("object")
            if pred and obj == "_":
                impossible_predicate = URIRef(str(pred))
                break
        if impossible_predicate is None:
            return None

        matching_subjects: list[URIRef] = []
        for subject in candidates:
            if any(True for _ in self.graph.objects(subject, impossible_predicate)):
                matching_subjects.append(subject)
        if not matching_subjects:
            # Still globally impossible under ambiguity.
            return None

        triples = self._triples_for_predicates(matching_subjects, [impossible_predicate])
        if not triples:
            return None

        project_paths = [str(RDFS.label), str(impossible_predicate)]
        expected_trace = self._query_builder_trace(label, anchor_type, project_paths)

        return AmbiguityExpansion(
            expected_triples=triples,
            expected_trace=expected_trace,
            qtype_override="direct",
            reason="impossible_label_ambiguous_not_globally_impossible",
            label=label,
            candidate_count=len(candidates),
        )
