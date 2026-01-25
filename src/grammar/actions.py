from dataclasses import dataclass, field
import random
from typing import Optional, List, Any
from src.grammar.ontology import OntologySampler, EntityNode, PropertyNode
from src.grammar.language_utils import (
    classify_property, get_hop_phrases, get_backward_hop_phrases,
    format_property_as_noun_phrase,
    get_search_phrases, compose_question
)
from src.vm.core import RetrySignal

# Singleton sampler for the knowledge graph
_sampler: Optional[OntologySampler] = None

def get_sampler() -> OntologySampler:
    global _sampler
    if _sampler is None:
        _sampler = OntologySampler()
    return _sampler

# --- State Objects ---

@dataclass
class WorkingEntity:
    """Wraps an EntityNode with session-specific state, like seen properties."""
    node: EntityNode
    seen_properties: set = field(default_factory=set)
    
    @property
    def label(self): return self.node.label
    @property
    def uri(self): return self.node.uri
    @property
    def type_uri(self): return self.node.type_uri

def _peek(data: dict) -> Optional[WorkingEntity]:
    stack = data.get("s_entities")
    return stack[-1] if stack else None

# --- Oracle Functions ---

def sel_random_entity(data: dict):
    """Samples a random entity and pushes it onto the focal stack."""
    s = get_sampler()
    cls = s.get_random_class()
    node = s.get_random_entity(cls.uri)
    data["s_entities"].append(WorkingEntity(node))

def gen_random_facts(min_facts: int = 1, max_facts: int = 3):
    """Factory that returns an action sampling [min_facts, max_facts] for the focal entity."""
    def _gen_facts_logic(data: dict) -> Any:
        ent = _peek(data)
        if not ent: raise RetrySignal("No focal entity to sample facts from")

        s = get_sampler()
        all_props = s.get_entity_data_properties(ent.uri)
        
        # Filter for properties we haven't seen on THIS instance
        candidates = [p for p in all_props if p.uri not in ent.seen_properties]
        
        if not candidates:
            raise RetrySignal(f"Entity {ent.label} has no unused properties")
            
        # Decide how many to sample
        upper_bound = min(max_facts, len(candidates))
        if upper_bound < min_facts:
            # If we can't even satisfy the minimum required facts, we fail/retry
            raise RetrySignal(f"Entity {ent.label} has only {len(candidates)} properties, but {min_facts} are required")

        num_to_sample = random.randint(min_facts, upper_bound)
        sampled = random.sample(candidates, num_to_sample)
        
        for prop in sampled:
            # Sample value
            val = random.choice(prop.values) if prop.values else s.sample_random_literal_value(prop.uri)
            
            # Record trace
            data["trace"].append({
                "tool": "fact",
                "subject": ent.uri,
                "predicate": prop.uri,
                "object": val
            })
            
            # Accumulate for question generation
            data["s_facts"].append((prop, val))
            
            # Mark as seen
            ent.seen_properties.add(prop.uri)
        return
    return _gen_facts_logic

def sel_hop_target(data: dict) -> Any:
    """Transitions the focus by pushing a related entity onto the stack."""
    ent = _peek(data)
    if not ent: raise RetrySignal("No focal entity to hop from")

    s = get_sampler()
    # Get object properties actually used by THIS specific entity
    obj_props = s.get_entity_object_properties(ent.uri)
    random.shuffle(obj_props)
    
    if obj_props:
        prop = obj_props[0]
        target_uri = random.choice(prop.values)
        
        # Proper entity resolution instead of a partial node
        target_node = s.get_entity_node(target_uri)
        
        # Only provide target label if there is ambiguity (multiple values for the same property)
        target_label_for_nl = target_node.label if len(prop.values) > 1 else None
        
        phrases = get_hop_phrases(prop.label, target_label_for_nl)
        data["nl"].append(random.choice(phrases))
        # Push new focus
        data["s_entities"].append(WorkingEntity(target_node))
        return

    raise RetrySignal(f"Entity {ent.label} has no outgoing links (object properties) to hop to")

def sel_backward_hop_target(data: dict) -> Any:
    """Transitions the focus by pushing an entity that points TO the current focal entity."""
    ent = _peek(data)
    if not ent: raise RetrySignal("No focal entity to backward-hop from")

    s = get_sampler()
    # Get subjects pointing to this entity
    in_props = s.get_incoming_properties(ent.uri)
    random.shuffle(in_props)
    
    if in_props:
        prop = in_props[0]
        # values here are subjects
        target_uri = random.choice(prop.values)
        
        target_node = s.get_entity_node(target_uri)
        
        phrases = get_backward_hop_phrases(prop.label, target_node.label)
        data["nl"].append(random.choice(phrases))
        # Push new focus
        data["s_entities"].append(WorkingEntity(target_node))
        return

    raise RetrySignal(f"Entity {ent.label} has no incoming links to backward-hop from")


# --- Action Functions (for APPLY) ---

def gen_search(data: dict):
    ent = _peek(data)
    if not ent: return
    
    data["trace"].append({"tool": "search", "query": ent.label})
    
    phrases = get_search_phrases(ent.label)
    data["nl"].append(random.choice(phrases))

def gen_inspect(data: dict):
    """Records an inspect tool call in the trace. No NL added (internal agent step)."""
    ent = _peek(data)
    if not ent: return
    data["trace"].append({"tool": "inspect", "uri": ent.uri})


def make_question(data: dict):
    s_facts = data["s_facts"]
    nl = data["nl"]
    ent = _peek(data)
    
    # Consolidate facts (lifo)
    facts = [item for item in reversed(s_facts) if isinstance(item, tuple)]
    
    if facts:
        is_plural = len(facts) > 1
        
        # Determine if we are "deep" in a hop to add connective particles
        stack_depth = len(data.get("s_entities", []))
        prefix = ""
        if stack_depth > 1:
            type_label = get_sampler().get_label(ent.type_uri)
            # Avoid 'thing' for a more natural persona
            if type_label.lower() in ["thing", "entity"]:
                type_label = random.choice(["entry", "record", "item"])
                
            prefix = random.choice([
                f"And for that {type_label}, ",
                f"Specifically for that {type_label}, ",
                f"Following that {type_label}, ",
                f"Regarding that {type_label}, ",
                f"While we are looking at that {type_label}, ",
                "For that entry, "
            ])
            article = "its"
        else:
            article = "its"

        # Consolidate phrases
        prop_labels = [p.label for p, val in facts]
        
        question = compose_question(prop_labels, prefix, article, ent.label)
        nl.append(question)
    else:
        nl.append(f"Could you provide more context for '{ent.label}'?")
