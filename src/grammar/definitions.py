from src.vm.core import Rule, Choice, MANY, APPLY, Push, Pop, Literal
import src.grammar.actions as ops

# -- GRAMMAR RULES --

def root():
    return Choice(
        (Rule_one_hop(), 1.0),
        (Rule_two_hop(), 0.0)
    )

def Rule_one_hop():
    return Rule(
        Rule_search(),
        Rule_inspect_anchor(),
        MANY(
            Rule_fact_step(),
            probability=0.4,
        ),
        Rule_finale()
    )

def Rule_two_hop():
    return Rule(
        Rule_search(),
        Rule_inspect_anchor(),
        Rule_hop(),
        Rule_inspect_anchor(),
        MANY(
            Rule_fact_step(),
            probability=0.4,
        ),
        Rule_finale()
    )

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

def Rule_fact_step():
    """A single step of finding and recording a property-value fact."""
    return Rule(
        # a. Discovery
        APPLY(ops.sel_valid_property_of_entity, access=["s_entities", "current_property"]),
        APPLY(ops.sample_fact_value, access=["s_entities", "current_property", "current_value"]),
        
        # b. Recording
        APPLY(ops.gen_fact_trace, access=["s_entities", "current_property", "current_value", "trace"]),
        
        # c. State management (Update seen properties on the top entity object)
        APPLY(lambda data: data["s_entities"][-1].seen_properties.add(data["current_property"].uri), 
              access=["s_entities", "current_property"]),
        
        # d. Accumulation for Question
        APPLY(ops.combine_prop_val, access=["current_property", "current_value", "current_fact"]),
        Push("s_facts", read="current_fact")
    )

def Rule_finale():
    """Generates the final question based on gathered facts."""
    return APPLY(ops.make_question, access=["s_entities", "s_facts", "nl"])