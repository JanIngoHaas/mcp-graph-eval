from dataclasses import dataclass, field
import random
from typing import Optional, List, Any, Set
from rdflib import URIRef, Literal
from src.grammar.ontology import OntologySampler, TypeNode, EntityNode, PropertyNode, PropertyRange
from src.grammar.language_utils import (
    classify_property, get_hop_phrases,
    format_property_as_noun_phrase, humanize_label,
    get_search_phrases, compose_question, compose_qb_question
)
from src.vm.core import RetrySignal

# Singleton sampler for the knowledge graph
_sampler: Optional[OntologySampler] = None

def get_sampler() -> OntologySampler:
    global _sampler
    if _sampler is None:
        _sampler = OntologySampler()
    return _sampler

# --- Configuration ---
PROB_QB_DEEP_FILTER = 0.3
DEFAULT_QUERY_LIMIT = 10

# --- State Objects ---

@dataclass
class WorkingEntity:
    """Wraps an EntityNode with session-specific state, like seen properties."""
    node: EntityNode
    seen_properties: Set[URIRef] = field(default_factory=set)
    
    @property
    def label(self): return self.node.label
    @property
    def uri(self): return self.node.uri
    @property
    def type_uri(self): return self.node.type_uri

def resolve_operator_and_value(term: Any) -> tuple[str, str]:
    """Determines appropriate SPARQL operator and potentially transforms value based on type."""
    # Default: string conversion
    val_str = str(term.toPython()) if isinstance(term, Literal) else str(term)
    
    if isinstance(term, Literal):
        py_val = term.toPython()
        # 1. Numeric Logic
        if isinstance(py_val, (int, float)):
            # We can't use '!=', '<', '>' because we need to keep the matching.
            # For instance '>' would be semantically different from what we can only express
            return random.choice(["=", ">=", "<=",]), val_str
        
        # 2. String Logic
        if isinstance(py_val, str):
            # Shorten long strings for 'contains' to look like a search snippet
            if len(val_str) > 20:
                words = val_str.split()
                if len(words) > 4:
                    # Pick a window of 2-4 words
                    window_size = random.randint(2, 4)
                    start = random.randint(0, len(words) - window_size)
                    val_str = " ".join(words[start : start + window_size])
            return "contains", val_str
             
    # 3. Object Logic (URIRef)
    return random.choice(["=", "!="]), val_str

@dataclass
class QueryBuilderState:
    """State for constructing a complex query."""
    root_type: URIRef
    filters: List[dict] = field(default_factory=list)
    projects: List[URIRef] = field(default_factory=list)
    anchor_entity: Optional[WorkingEntity] = None


def _peek(data: dict) -> Optional[WorkingEntity]:
    stack = data.get("s_entities")
    return stack[-1] if stack else None

# --- Oracle Functions ---

def sel_random_entity(data: dict):
    """Samples a random entity and pushes it onto the focal stack."""
    s = get_sampler()
    type_node = s.get_random_type()
    node = s.get_random_entity(type_node.uri)
    data["s_entities"].append(WorkingEntity(node))

def gen_random_facts(min_facts: int = 1, max_facts: int = 3):
    """Factory that returns an action sampling [min_facts, max_facts] for the focal entity."""
    def _gen_facts_logic(data: dict) -> Any:
        ent: EntityNode = _peek(data)
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
                "subject": str(ent.uri),
                "predicate": str(prop.uri),
                "object": str(val)
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
        target_node = s.resolve_entity(target_uri)
        
        # Record the connection fact first
        data["trace"].append({
            "tool": "fact",
            "subject": str(ent.uri),
            "predicate": str(prop.uri),
            "object": str(target_node.uri)
        })
        
        phrases = get_hop_phrases(prop.label, target_node.label)
        data["nl"].append(random.choice(phrases))
        # Push new focus
        data["s_entities"].append(WorkingEntity(target_node))
        return

    raise RetrySignal(f"Entity {ent.label} has no outgoing links (object properties) to hop to")


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


# --- Query Builder Actions ---

def qb_init_from_anchor(data: dict):
    """Initializes a Query Builder session based on the current focal entity."""
    ent = _peek(data)
    if not ent: raise RetrySignal("No anchor entity for QB")
    
    qb = QueryBuilderState(
        root_type=ent.type_uri,
        anchor_entity=ent
    )
    data["qb"] = qb

def qb_filter_generator(prob_deep: Optional[float] = None):
    """Factory that returns an action adding a filter to the query builder."""
    p_deep = prob_deep if prob_deep is not None else PROB_QB_DEEP_FILTER
    
    def _qb_add_filter_logic(data: dict) -> Any:
        qb: QueryBuilderState = data.get("qb")
        if not qb or not qb.anchor_entity: return
        
        ent = qb.anchor_entity
        s = get_sampler()
        
        # Decide on filter depth (direct vs 1-hop)
        is_deep = random.random() < p_deep
        existing_paths = {f["path_uri"] for f in qb.filters}

        if is_deep:
            # 1-Hop Filter: ent -> p1 -> ent2 -> p2 -> val
            obj_props = s.get_entity_object_properties(ent.uri)
            if not obj_props: 
                raise RetrySignal(f"Anchor {ent.label} has no object properties for deep filter")
            
            # Find a path that isn't already used
            random.shuffle(obj_props)
            p1 = None
            p2 = None
            val_term = None
            
            for candidate_p1 in obj_props:
                if not candidate_p1.values: continue
                target_uri = random.choice(candidate_p1.values)
                target_props = s.get_entity_data_properties(target_uri)
                
                # Check for unique second hop
                for candidate_p2 in target_props:
                    path_uri = f"<{candidate_p1.uri}>.<{candidate_p2.uri}>"
                    if path_uri not in existing_paths:
                        p1, p2 = candidate_p1, candidate_p2
                        val_term = random.choice(p2.values)
                        break
                if p1: break
            
            if not p1:
                raise RetrySignal("No unique deep paths found from anchor")
            
            op, val_str = resolve_operator_and_value(val_term)
            qb.filters.append({
                "path_uri": f"<{p1.uri}>.<{p2.uri}>",
                "path_display": f"{p1.label}.{p2.label}",
                "operator": op,
                "value": val_str,
                "type": "deep"
            })
            
        else:
            # Direct Filter: ent -> p1 -> val
            all_props = s.get_entity_properties(ent.uri)
            if not all_props: 
                raise RetrySignal(f"Anchor {ent.label} has no properties for filter")
            
            candidates = [p for p in all_props if f"<{p.uri}>" not in existing_paths]
            if not candidates:
                 raise RetrySignal(f"No unique properties left for filter on {ent.label}")

            p1 = random.choice(candidates)
            if not p1.values: 
                raise RetrySignal(f"Property {p1.label} on {ent.label} has no values")
            
            val_term = random.choice(p1.values)
            op, val_str = resolve_operator_and_value(val_term)
            
            qb.filters.append({
                "path_uri": f"<{p1.uri}>",
                "path_display": p1.label,
                "operator": op,
                "value": val_str,
                "type": "direct"
            })
    return _qb_add_filter_logic

def qb_projection_generator():
    """Factory that returns an action adding a projection field to the query."""
    def _qb_add_projection_logic(data: dict) -> Any:
        qb: QueryBuilderState = data.get("qb")
        if not qb: return
        
        s = get_sampler()
        # Randomly select a data property of the root type to project
        all_props = s.get_entity_data_properties(qb.anchor_entity.uri)
        if not all_props: 
            raise RetrySignal(f"Anchor {qb.anchor_entity.label} has no data properties to project")
        
        p = random.choice(all_props)
        
        # Avoid projecting the same thing twice
        if p.uri in qb.projects:
            return
            
        qb.projects.append(p.uri)
    return _qb_add_projection_logic

def qb_finalize_question(data: dict):
    """Constructs the query_builder tool call and the NL question."""
    qb: QueryBuilderState = data.get("qb")
    if not qb or not qb.filters or not qb.projects:
        # Fallback if generation failed
        raise RetrySignal("Logic failed to generate valid QB state")
        
    # 1. Build JSON Tool Call
    # Map our internal state to the tool schema
    tool_filters = []
    for f in qb.filters:
        tool_filters.append({
            "path": f["path_uri"],
            "operator": f["operator"],
            "value": f["value"]
        })
    
    # Add label projection if not present (users usually want labels)
    from rdflib.namespace import RDFS
    if RDFS.label not in qb.projects:
        qb.projects.insert(0, RDFS.label)
        
    tool_call = {
        "tool": "query_builder",
        "type": str(qb.root_type),
        "filters": tool_filters,
        "project": [f"<{p}>" for p in qb.projects],
        "limit": DEFAULT_QUERY_LIMIT
    }
    
    data["trace"].append(tool_call)
    
    # 2. Build Natural Language Question
    root_type_node = TypeNode(qb.root_type, get_sampler().get_label(qb.root_type))
    proj_labels = [get_sampler().get_label(p) for p in qb.projects]
    
    question = compose_qb_question(
        root_plural=root_type_node.plural,
        filters=qb.filters,
        proj_labels=proj_labels,
        existing_nl=data["nl"]
    )
    
    data["nl"].append(question)
