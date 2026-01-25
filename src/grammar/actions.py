from dataclasses import dataclass, field
import random
from typing import Optional, List, Any
from src.grammar.ontology import OntologySampler, EntityNode, PropertyNode
from src.grammar.utils import (
    classify_property, get_hop_phrases, format_property_as_noun_phrase,
    get_search_phrases, get_inspect_phrases
)
from src.vm.core import BREAK

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
    
    if data["s_entities"] is None: data["s_entities"] = []
    data["s_entities"].append(WorkingEntity(node))

def combine_prop_val(data: dict):
    """Combines property and value into a tuple for the stack."""
    data["current_fact"] = (data["current_property"], data["current_value"])

def sel_valid_property_of_entity(data: dict) -> Any:
    """Finds a property not already seen for the current focal entity."""
    ent = _peek(data)
    if not ent: return BREAK

    s = get_sampler()
    all_props = s.get_entity_properties(ent.uri)
    # Filter for literals or things we haven't seen on THIS instance
    candidates = [p for p in all_props if p.uri not in ent.seen_properties]
    
    if not candidates:
        return BREAK
        
    data["current_property"] = random.choice(candidates)

def sel_hop_target(data: dict) -> Any:
    """Transitions the focus by pushing a related entity onto the stack."""
    ent = _peek(data)
    if not ent: return BREAK

    s = get_sampler()
    # Get properties actually used by THIS specific entity
    all_props = s.get_entity_properties(ent.uri)
    obj_props = [p for p in all_props if p.range_type == "Class" and p.values]
    random.shuffle(obj_props)
    
    if obj_props:
        prop = obj_props[0]
        target_uri = random.choice(prop.values)
        
        # Proper entity resolution instead of a partial node
        target_node = s.get_entity_node(target_uri)
        
        phrases = get_hop_phrases(prop.label)
        data["nl"].append(random.choice(phrases))
        # Push new focus
        data["s_entities"].append(WorkingEntity(target_node))
        return

    return BREAK

def sample_fact_value(data: dict):
    """Samples a value for the current property relative to the focal entity."""
    prop = data["current_property"]
    
    # If discovery already gave us values, use them!
    if prop.values:
        data["current_value"] = random.choice(prop.values)
        return

    # Fallback (should be extremely rare now)
    ent = _peek(data)
    s = get_sampler()
    data["current_value"] = s.sample_random_literal_value(prop.uri)

# --- Action Functions (for APPLY) ---

def gen_search(data: dict):
    ent = _peek(data)
    if not ent: return
    
    if data["trace"] is None: data["trace"] = []
    if data["nl"] is None: data["nl"] = []
    
    data["trace"].append({"tool": "search", "query": ent.label})
    phrases = get_search_phrases(ent.label)
    data["nl"].append(random.choice(phrases))

def gen_inspect(data: dict):
    ent = _peek(data)
    if not ent: return

    if data["trace"] is None: data["trace"] = []
    if data["nl"] is None: data["nl"] = []
    
    data["trace"].append({"tool": "inspect", "uri": ent.uri})
    
    # Transition early if needed
    if len(data["nl"]) <= 1 and random.random() > 0.5:
        phrases = get_inspect_phrases()
        data["nl"].append(random.choice(phrases))

def gen_fact_trace(data: dict):
    ent = _peek(data)
    prop = data["current_property"]
    val = data["current_value"]
    
    if data["trace"] is None: data["trace"] = []
    data["trace"].append({
        "tool": "fact",
        "subject": ent.uri,
        "predicate": prop.uri,
        "object": val
    })

def make_question(data: dict):
    s_facts = data["s_facts"] or []
    nl = data["nl"] or []
    ent = _peek(data)
    
    # Consolidate facts (lifo)
    facts = [item for item in reversed(s_facts) if isinstance(item, tuple)]
    
    if facts:
        is_plural = len(facts) > 1
        
        # Determine if we are "deep" in a hop to add connective particles
        stack_depth = len(data.get("s_entities", []))
        prefix = ""
        if stack_depth > 1:
            type_label = get_sampler()._get_label(ent.type_uri)
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

        # Heuristic for researcher-like phrasing
        formatted_phrases = []
        for prop, val in facts:
            formatted_phrases.append(format_property_as_noun_phrase(prop.label))

        if not is_plural:
            label_str = formatted_phrases[0]
            verb = "is"
        else:
            label_str = f"{', '.join(formatted_phrases[:-1])} and {formatted_phrases[-1]}"
            verb = "are"

        # Researcher-style inquiries
        questions = [
            f"{prefix}what {verb} {article} {label_str}?",
            f"{prefix}could you please identify {article} {label_str}?",
            f"{prefix}I'm looking for {article} {label_str}.",
            f"{prefix}I'd like to verify {article} {label_str}.",
            f"What info do we have on {article} {label_str}?"
        ]
        nl.append(random.choice(questions))
    else:
        nl.append(f"Could you provide more context for '{ent.label}'?")
