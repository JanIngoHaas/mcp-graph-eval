import math
from typing import Dict, Iterable, Tuple

from src.vm.core import Rule, MANY, APPLY, Retry, Repeat
import src.grammar.actions as ops

# -- GRAMMAR RULES --

_DEFAULT_QTYPE_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("query_builder", 0.3),
    ("direct", 0.25),
    ("hop", 0.25),
    ("impossible", 0.2),
)

_QB_PROJECTION_MIN = 1
_QB_PROJECTION_MAX = 2


# Inlined reset keys (lists + scalar state)
RESET_KEYS = [
    "nl",
    "trace",
    "s_entities",
    "s_facts",
    "answer_triples",
    "qtype",
    "qb",
    "hop_bridge_predicate_uri",
    "hop_target_count",
    "hop_scope",
]

def compute_qtype_targets(total_amount: int, weights: Iterable[Tuple[str, float]]) -> Dict[str, int]:
    if total_amount <= 0:
        raise ValueError("total_amount must be positive")

    items = list(weights)
    if not items:
        raise ValueError("weights must be non-empty")

    targets: Dict[str, int] = {}
    for qtype, weight in items:
        targets[qtype] = int(math.ceil(total_amount * weight))

    return targets

import re as _re

def collect_sample(data: dict):
    if data.get("samples") is None:
        data["samples"] = []
    # Join NL fragments and normalise whitespace / punctuation
    raw = " ".join(data.get("nl") or [])
    question = _re.sub(r'\s+', ' ', raw).strip()
    question = question.replace("??", "?").replace("?.", "?").replace("..", ".")
    # Capitalise the first letter
    if question:
        question = question[0].upper() + question[1:]
    data["samples"].append({
        "question": question,
        "trace": data.get("trace") or [],
        "qtype": data.get("qtype"),
        "hop_bridge_predicate_uri": data.get("hop_bridge_predicate_uri"),
        "hop_target_count": data.get("hop_target_count"),
        "hop_scope": data.get("hop_scope"),
    })

def sample(rule_node):
    return Rule(
        rule_node,
        APPLY(
            collect_sample,
            access=[
                "samples",
                "nl",
                "trace",
                "qtype",
                "hop_bridge_predicate_uri",
                "hop_target_count",
                "hop_scope",
            ],
        ),
    )

def root(total_amount: int = 150, qtype_weights: Dict[str, float] | None = None):
    weights = qtype_weights or dict(_DEFAULT_QTYPE_WEIGHTS)
    targets = compute_qtype_targets(total_amount, weights.items())
    return Rule(
        Repeat(sample(Rule_query_builder()), count=targets["query_builder"], reset_keys=RESET_KEYS),
        Repeat(sample(Rule_direct()), count=targets["direct"], reset_keys=RESET_KEYS),
        Repeat(sample(Rule_forward_hop()), count=targets["hop"], reset_keys=RESET_KEYS),
        Repeat(sample(Rule_impossible()), count=targets["impossible"], reset_keys=RESET_KEYS),
    )

def Rule_impossible():
    """Generates a question about a property the entity does NOT have."""
    return Retry(Rule(
        APPLY(ops.sel_random_entity, access=["s_entities"]),
        APPLY(ops.gen_search, access=["s_entities", "trace", "nl"]),
        APPLY(ops.gen_inspect, access=["s_entities", "trace", "nl"]),
        APPLY(ops.gen_impossible_fact, access=["s_entities", "trace", "s_facts", "global_seen_impossible"]),
        Rule_fact_finale(),
        APPLY(ops.add_type_to_question("impossible"), access=["qtype"]),
    ), n=25)

def Rule_query_builder(preamble=None):
    """Generates a complex structured query. 'preamble' establishes the anchor entity."""
    if preamble is None:
        preamble = APPLY(ops.sel_stratified_entity, access=["s_entities"])

    return Retry(Rule(
        preamble,
        APPLY(ops.qb_init_from_anchor, access=["s_entities", "qb", "nl"]),
        MANY(
            APPLY(ops.qb_filter_generator(), access=["qb"]),
            min_count=1, max_count=2
        ),
        MANY(
            APPLY(ops.qb_projection_generator(), access=["qb"]),
            min_count=_QB_PROJECTION_MIN,
            max_count=_QB_PROJECTION_MAX
        ),
        APPLY(ops.qb_finalize_question, access=["qb", "trace", "nl", "global_seen_qb"]),
        APPLY(ops.add_type_to_question("query_builder"), access=["qtype"]),
    ), n=25)

def Rule_direct(max_facts=2):
    """Simple direct fact lookup about an entity."""
    return Retry(Rule(
        Rule_search(),
        Rule_inspect_anchor(),
        Rule_sample_facts(max_facts=max_facts),
        Rule_fact_finale(),
        APPLY(ops.add_type_to_question("direct"), access=["qtype"]),
    ), n=25)

def Rule_forward_hop():
    """Starts at an entity, hops to a related entity, and asks about it."""
    return Retry(Rule(
        Rule_search_hoppable(),
        Rule_inspect_anchor(),
        Rule_hop(),
        Rule_inspect_anchor(),
        Rule_sample_facts(),
        Rule_fact_finale(),
        APPLY(ops.add_type_to_question("hop"), access=["qtype"]),
    ), n=25)

def Rule_sample_facts(max_facts: int = 2):
    """Samples properties and values for the current focal entity."""
    return APPLY(ops.gen_random_facts(max_facts=max_facts), access=["s_entities", "trace", "s_facts", "answer_triples", "global_seen_direct"])

def Rule_search():
    """Initial discovery step."""
    return Rule(
        APPLY(ops.sel_random_entity, access=["s_entities"]),
        APPLY(ops.gen_search, access=["s_entities", "trace", "nl"]),
    )

def Rule_search_hoppable():
    """Initial discovery step for hop questions (requires an outgoing object property)."""
    return Rule(
        APPLY(ops.sel_hoppable_entity, access=["s_entities"]),
        APPLY(ops.gen_search, access=["s_entities", "trace", "nl"]),
    )

def Rule_inspect_anchor():
    """Focuses the investigation on the current entity."""
    return APPLY(ops.gen_inspect, access=["s_entities", "trace", "nl"])

def Rule_hop():
    """Transitions from the current entity to a related one."""
    return APPLY(
        ops.sel_hop_target,
        access=["s_entities", "trace", "nl", "hop_bridge_predicate_uri", "hop_target_count", "hop_scope", "global_seen_hop"],
    )

def Rule_fact_finale():
    """Generates the final question based on gathered facts."""
    return APPLY(ops.make_question, access=["s_entities", "s_facts", "nl"])
