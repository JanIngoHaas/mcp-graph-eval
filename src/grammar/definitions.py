from src.vm.core import Rule, Choice, MANY, APPLY, Push, Pop, Literal, Retry
import src.grammar.actions as ops

# -- GRAMMAR RULES --

def root():
    return Choice(
        (Rule_direct(), 0.25),
        (Rule_forward_hop(), 0.25),
        (Rule_impossible(), 0.15),
        (Rule_query_builder(), 0.45)
    )

def Rule_impossible():
    """Generates a question about a property the entity does NOT have."""
    return Retry(Rule(
        APPLY(ops.sel_random_entity, access=["s_entities"]),
        APPLY(ops.gen_search, access=["s_entities", "trace", "nl"]),
        APPLY(ops.gen_inspect, access=["s_entities", "trace", "nl"]),
        APPLY(ops.gen_impossible_fact, access=["s_entities", "trace", "s_facts"]),
        Rule_fact_finale(),
        APPLY(ops.add_type_to_question("impossible"), access=["qtype"]),
    ), n=5)

def Rule_query_builder(preamble=None):
    """Generates a complex structured query. 'preamble' establishes the anchor entity."""
    if preamble is None:
        preamble = APPLY(ops.sel_random_entity, access=["s_entities"])

    return Retry(Rule(
        preamble,
        APPLY(ops.qb_init_from_anchor, access=["s_entities", "qb", "nl"]),
        MANY(
            APPLY(ops.qb_filter_generator(), access=["qb"]),
            min_count=1, max_count=2
        ),
        MANY(
            APPLY(ops.qb_projection_generator(), access=["qb"]),
            probability=0.25
        ),
        APPLY(ops.qb_finalize_question, access=["qb", "trace", "nl"]),
        APPLY(ops.add_type_to_question("query_builder"), access=["qtype"]),
    ), n=5)

def Rule_direct(max_facts=2):
    """Simple direct fact lookup about an entity."""
    return Retry(Rule(
        Rule_search(),
        Rule_inspect_anchor(),
        Rule_sample_facts(max_facts=max_facts),
        Rule_fact_finale(),
        APPLY(ops.add_type_to_question("direct"), access=["qtype"]),
    ), n=5)

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
    ), n=5)

def Rule_sample_facts(max_facts: int = 2):
    """Samples properties and values for the current focal entity."""
    return APPLY(ops.gen_random_facts(max_facts=max_facts), access=["s_entities", "trace", "s_facts", "answer_triples"])

def Rule_search():
    """Initial discovery step."""
    return Rule(
        APPLY(ops.sel_random_entity, access=["s_entities"]),
        APPLY(ops.gen_search, access=["s_entities", "trace", "nl"])
    )

def Rule_search_hoppable():
    """Initial discovery step for hop questions (requires an outgoing object property)."""
    return Rule(
        APPLY(ops.sel_hoppable_entity, access=["s_entities"]),
        APPLY(ops.gen_search, access=["s_entities", "trace", "nl"])
    )

def Rule_inspect_anchor():
    """Focuses the investigation on the current entity."""
    return APPLY(ops.gen_inspect, access=["s_entities", "trace", "nl"])

def Rule_hop():
    """Transitions from the current entity to a related one."""
    return APPLY(ops.sel_hop_target, access=["s_entities", "trace", "nl"])

def Rule_fact_finale():
    """Generates the final question based on gathered facts."""
    return APPLY(ops.make_question, access=["s_entities", "s_facts", "nl"])
