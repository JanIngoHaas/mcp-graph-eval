from src.vm.core import Rule, Choice, MANY, APPLY, Push, Pop, Literal, Retry
import src.grammar.actions as ops

# -- GRAMMAR RULES --

def root():
    return Choice(
        (Rule_direct(), 0.3),
        (Rule_forward_hop(), 0.35),
        (Rule_backward_hop_sequence(), 0.35)
    )

def Rule_direct():
    """Simple direct fact lookup about an entity."""
    return Retry(Rule(
        Rule_search(),
        Rule_inspect_anchor(),
        Rule_sample_facts(),
        Rule_fact_finale()
    ), n=5)

def Rule_forward_hop():
    """Starts at an entity, hops to a related entity, and asks about it."""
    return Retry(Rule(
        Rule_search(),
        Rule_inspect_anchor(),
        Rule_hop(),
        Rule_inspect_anchor(),
        Rule_sample_facts(),
        Rule_fact_finale()
    ), n=5)

def Rule_backward_hop_sequence():
    """Starts at an entity, hops backwards to an incoming link, and asks about it."""
    return Retry(Rule(
        Rule_search(),
        # No initial inspect needed for backward hops! 
        # (Identifying the target is enough to search for its incoming links)
        Rule_backward_hop_action(),
        Rule_inspect_anchor(),
        Rule_sample_facts(),
        Rule_fact_finale()
    ), n=5)

def Rule_sample_facts(max_facts: int = 2):
    """Samples properties and values for the current focal entity."""
    return APPLY(ops.gen_random_facts(max_facts=max_facts), access=["s_entities", "trace", "s_facts"])

def Rule_search():
    """Initial discovery step."""
    return Rule(
        APPLY(ops.sel_random_entity, access=["s_entities"]),
        APPLY(ops.gen_search, access=["s_entities", "trace", "nl"])
    )

def Rule_inspect_anchor():
    """Focuses the investigation on the current entity."""
    return APPLY(ops.gen_inspect, access=["s_entities", "trace", "nl"])

def Rule_hop():
    """Transitions from the current entity to a related one."""
    return APPLY(ops.sel_hop_target, access=["s_entities", "nl"])

def Rule_backward_hop_action():
    """Transitions from the current entity to one that points to it."""
    return APPLY(ops.sel_backward_hop_target, access=["s_entities", "nl"])

def Rule_fact_finale():
    """Generates the final question based on gathered facts."""
    return APPLY(ops.make_question, access=["s_entities", "s_facts", "nl"])