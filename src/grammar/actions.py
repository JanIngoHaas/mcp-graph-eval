from dataclasses import dataclass, field
import random
from typing import Optional, List, Any, Set
from rdflib import URIRef, Literal
from src.grammar.ontology import OntologySampler, TypeNode, EntityNode, PropertyNode, PropertyRange
from src.grammar.language_utils import (
    classify_property, get_hop_phrases,
    format_property_as_noun_phrase, humanize_label,
    get_search_phrases, compose_question, compose_qb_question,
    pluralize
)
from src.vm.core import RetrySignal

# Singleton sampler for the knowledge graph
_sampler: Optional[OntologySampler] = None

def get_sampler() -> OntologySampler:
    global _sampler
    if _sampler is None:
        _sampler = OntologySampler()
    return _sampler

def _append_trace(data: dict, step: dict, required: bool = False, is_answer: bool = False) -> None:
    """Append only answer-bearing tool calls to trace."""
    if not is_answer:
        return
    entry = dict(step)
    entry["required"] = required
    entry["is_answer"] = bool(is_answer)
    data["trace"].append(entry)

# --- Configuration ---
PROB_QB_DEEP_FILTER = 0.30
UNIQUE_ANCHOR_MAX_ATTEMPTS = 120
HOPPABLE_SAMPLER_MAX_ATTEMPTS = 20

# --- State Objects ---

@dataclass
class EntityRef:
    """Canonical immutable entity identity used for working state."""
    uri: URIRef
    label: str
    type_uri: URIRef

    @classmethod
    def from_node(cls, node: EntityNode) -> "EntityRef":
        return cls(uri=node.uri, label=node.label, type_uri=node.type_uri)

    def to_dict(self) -> dict:
        return {
            "uri": str(self.uri),
            "label": self.label,
            "type_uri": str(self.type_uri),
        }


@dataclass
class WorkingEntity:
    """Runtime wrapper around EntityRef for traversal-local mutable state."""
    ref: EntityRef
    seen_properties: Set[URIRef] = field(default_factory=set)

    @classmethod
    def from_node(cls, node: EntityNode) -> "WorkingEntity":
        return cls(ref=EntityRef.from_node(node))

    @property
    def label(self): return self.ref.label
    @property
    def uri(self): return self.ref.uri
    @property
    def type_uri(self): return self.ref.type_uri

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
    return random.choice(["="]), val_str

@dataclass
class QueryBuilderState:
    """State for constructing a complex query."""
    root_type: URIRef
    filters: List[dict] = field(default_factory=list)
    projects: List[URIRef] = field(default_factory=list)
    anchor_entity: Optional[WorkingEntity] = None


def _peek(data: dict) -> WorkingEntity:
    stack = data.get("s_entities")
    if not stack: raise ValueError("Focal stack is empty")
    return stack[-1]


def _sample_unique_entity(require_hoppable: bool = False) -> EntityNode:
    """Sample an entity whose (label,type) pair is unique."""
    s = get_sampler()
    last_reason = "unknown"

    for _ in range(UNIQUE_ANCHOR_MAX_ATTEMPTS):
        try:
            if require_hoppable:
                node = s.get_random_hoppable_entity(max_attempts=HOPPABLE_SAMPLER_MAX_ATTEMPTS)
            else:
                node = s.get_random_entity()
        except RetrySignal as exc:
            last_reason = str(exc)
            continue

        primary_type = s.get_entity_primary_type(node.uri) or node.type_uri
        count = s.count_entities_by_label_and_type(node.label, primary_type)
        if count == 1:
            if primary_type != node.type_uri:
                node = EntityNode(node.uri, node.label, primary_type)
            return node
        last_reason = f"label='{node.label}', type='{primary_type}', count={count}"

    raise RetrySignal(f"Failed to find unique anchor after {UNIQUE_ANCHOR_MAX_ATTEMPTS} attempts ({last_reason})")

# --- Oracle Functions ---

def sel_random_entity(data: dict):
    """Samples a random entity and pushes it onto the focal stack."""
    node = _sample_unique_entity(require_hoppable=False)
    data["s_entities"].append(WorkingEntity.from_node(node))

def sel_hoppable_entity(data: dict):
    """Samples a random entity that has at least one outgoing object property."""
    node = _sample_unique_entity(require_hoppable=True)
    data["s_entities"].append(WorkingEntity.from_node(node))

def gen_random_facts(min_facts: int = 1, max_facts: int = 3):
    """Factory that returns an action sampling [min_facts, max_facts] for the focal entity."""
    def _gen_facts_logic(data: dict) -> Any:
        s = get_sampler()
        stack = data.get("s_entities") or []
        if not stack:
            raise RetrySignal("No focal entity to sample facts from")

        # For hop questions, stack contains [anchor, target1, target2, ...].
        # In multi-target hops we sample answer facts for all targets.
        entities: list[WorkingEntity] = stack[1:] if len(stack) > 1 else [stack[-1]]

        candidates_by_entity: list[list[PropertyNode]] = []
        for ent in entities:
            all_props = s.get_entity_data_properties(ent.uri)
            candidates = [p for p in all_props if p.uri not in ent.seen_properties]
            if not candidates:
                # continue
                raise RetrySignal(f"Entity {ent.label} has no unused properties")
            candidates_by_entity.append(candidates)

        sampled_by_entity: list[list[PropertyNode]] = []
        if len(entities) > 1:
            # Prefer a shared property to keep "all ..." hop questions coherent.
            common_uris = set(p.uri for p in candidates_by_entity[0])
            for candidates in candidates_by_entity[1:]:
                common_uris &= {p.uri for p in candidates}

            if not common_uris:
                raise RetrySignal(
                    "Multi-target hop has no shared property across targets; retrying for coherent 'all' semantics"
                )

            chosen_uri = random.choice(list(common_uris))
            for candidates in candidates_by_entity:
                picked = next(p for p in candidates if p.uri == chosen_uri)
                sampled_by_entity.append([picked])
        else:
            candidates = candidates_by_entity[0]
            upper_bound = min(max_facts, len(candidates))
            if upper_bound < min_facts:
                raise RetrySignal(
                    f"Entity {entities[0].label} has only {len(candidates)} properties, "
                    f"but {min_facts} are required"
                )
            num_to_sample = random.randint(min_facts, upper_bound)
            sampled_by_entity.append(random.sample(candidates, num_to_sample))

        for ent, sampled_props in zip(entities, sampled_by_entity):
            for prop in sampled_props:
                val = random.choice(prop.values) if prop.values else s.sample_random_literal_value(prop.uri)

                # Record answer-bearing fact trace.
                _append_trace(
                    data,
                    {
                        "tool": "fact",
                        "subject": str(ent.uri),
                        "predicate": str(prop.uri),
                        "object": str(val),
                    },
                    required=True,
                    is_answer=True,
                )

                data["s_facts"].append((prop, val))
                ent.seen_properties.add(prop.uri)
        return
    return _gen_facts_logic

def gen_impossible_fact(data: dict):
    """
    Selects a property that the current entity definitely DOES NOT have.
    Used for generating 'impossible' questions.
    """
    ent = _peek(data)
    if not ent: raise RetrySignal("No focal entity to sample impossible fact from")

    s = get_sampler()
    # 1. Get all properties ACTUALY present on the entity
    actual_props = s.get_entity_properties(ent.uri)
    actual_uris = {p.uri for p in actual_props}
    
    # 2. Pick a random property from the UNIVERSE that is NOT in actual_uris
    impossible_prop = s.get_random_property_excluding(actual_uris)
    
    # 3. Record trace - the agent effectively "checks" this property
    _append_trace(
        data,
        {
            "tool": "fact",
            "subject": str(ent.uri),
            "predicate": str(impossible_prop.uri),
            "object": "_",
        },
        required=True,
        is_answer=False,
    )
    
    # 4. Add to s_facts for question generation -> (prop, None) implies no value found
    data["s_facts"].append((impossible_prop, None))
    
    # 5. NO answer_triples (or maybe an explicit "I don't know" marker if needed later)
    # The evaluator should see empty answer triples and "I don't know" in the model response
    return

def sel_hop_target(data: dict) -> Any:
    """Transitions the focus by pushing a related entity onto the stack."""
    ent = _peek(data)
    if not ent: raise RetrySignal("No focal entity to hop from")

    s = get_sampler()
    # Get object properties actually used by THIS specific entity
    obj_props = s.get_entity_object_properties(ent.uri)
    random.shuffle(obj_props)

    for prop in obj_props:
        target_candidates = [uri for uri in dict.fromkeys(prop.values) if isinstance(uri, URIRef)]
        if not target_candidates:
            continue
        random.shuffle(target_candidates)
        hop_target_count = len(target_candidates)
        hop_scope = "one" if hop_target_count == 1 else "all"
        selected_targets = target_candidates[:1] if hop_scope == "one" else target_candidates

        resolved_targets: list[EntityNode] = []
        for target_uri in selected_targets:
            target_node = s.resolve_entity(target_uri)
            resolved_targets.append(target_node)

            # Record the bridge fact for each selected target.
            _append_trace(
                data,
                {
                    "tool": "fact",
                    "subject": str(ent.uri),
                    "predicate": str(prop.uri),
                    "object": str(target_node.uri),
                },
                required=True,
                is_answer=True,
            )

        # Avoid leaking concrete targets into the NL bridge.
        data["hop_bridge_predicate_uri"] = str(prop.uri)
        data["hop_target_count"] = hop_target_count
        data["hop_scope"] = hop_scope
        phrases = get_hop_phrases(prop.label, scope=hop_scope)
        data["nl"].append(random.choice(phrases))

        # Push selected target entities to support downstream answer-fact generation.
        for node in resolved_targets:
            data["s_entities"].append(WorkingEntity.from_node(node))
        return

    raise RetrySignal(f"Entity {ent.label} has no outgoing links (object properties) to hop to")


# --- Action Functions (for APPLY) ---

def gen_search(data: dict):
    ent = _peek(data)
    if not ent: return
    
    _append_trace(
        data,
        {"tool": "search", "query": ent.label},
        required=True,
        is_answer=False,
    )
    
    phrases = get_search_phrases(ent.label)
    data["nl"].append(random.choice(phrases))

def gen_inspect(data: dict):
    """Records an inspect tool call in the trace. No NL added (internal agent step)."""
    ent = _peek(data)
    if not ent: return
    _append_trace(
        data,
        {"tool": "inspect", "uri": ent.uri},
        required=True,
        is_answer=False,
    )

def add_type_to_question(qtype: str):
    def inner(data: dict):
        data["qtype"] = qtype
    return inner

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

        # Consolidate phrases while keeping first-seen order.
        prop_labels = []
        seen_labels = set()
        for p, _ in facts:
            label = p.label
            if label in seen_labels:
                continue
            seen_labels.add(label)
            prop_labels.append(label)
        
        question = compose_question(prop_labels, prefix, article)
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
        print(f"Existing paths: {existing_paths}")

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
            target_uri_selected = None
            
            for candidate_p1 in obj_props:
                if not candidate_p1.values: continue
                target_uri = random.choice(candidate_p1.values)
                target_uri_selected = target_uri
                target_props = s.get_entity_data_properties(target_uri)
                
                # Check for unique second hop
                for candidate_p2 in target_props:
                    path_uri = f"{candidate_p1.uri} -> {candidate_p2.uri}"
                    if path_uri not in existing_paths:
                        p1, p2 = candidate_p1, candidate_p2
                        val_term = random.choice(p2.values)
                        break
                if p1: break
            
            if not p1:
                raise RetrySignal("No unique deep paths found from anchor")
            
            op, val_str = resolve_operator_and_value(val_term)
            target_type_uri = None
            if target_uri_selected is not None:
                try:
                    target_entity = s.resolve_entity(target_uri_selected)
                    target_type_uri = str(target_entity.type_uri)
                except Exception:
                    target_type_uri = None

            qb.filters.append({
                "path_uri": f"{p1.uri} -> {p2.uri}",
                "path_display": f"{p1.label}->{p2.label}",
                "operator": op,
                "value": val_str,
                "type": "deep",
                "path_segments": [str(p1.uri), str(p2.uri)],
                "target_type_uri": target_type_uri,
            })
            
        else:
            # Direct Filter: ent -> p1 -> val
            all_props = s.get_entity_properties(ent.uri)
            if not all_props: 
                raise RetrySignal(f"Anchor {ent.label} has no properties for filter")
            
            candidates = [p for p in all_props if p.uri not in existing_paths]
            if not candidates:
                 raise RetrySignal(f"No unique properties left for filter on {ent.label}")

            print(candidates)
            p1 = random.choice(candidates)
            if not p1.values: 
                raise RetrySignal(f"Property {p1.label} on {ent.label} has no values")
            
            val_term = random.choice(p1.values)
            op, val_str = resolve_operator_and_value(val_term)
            
            qb.filters.append({
                "path_uri": p1.uri,
                "path_display": p1.label,
                "operator": op,
                "value": val_str,
                "type": "direct",
                "path_segments": [str(p1.uri)],
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
        
    # 0. Discovery steps (Enriching trace for explanation)
    s = get_sampler()
    type_label = s.get_label(qb.root_type)
    
    # Agent searches for the class
    _append_trace(
        data,
        {"tool": "search", "query": type_label},
        required=True,
        is_answer=False,
    )
    # Agent inspects the class to see properties
    _append_trace(
        data,
        {"tool": "inspect", "uri": str(qb.root_type)},
        required=True,
        is_answer=False,
    )

    # If filters require a hop, enforce inspection of the linking property,
    # the target type, and the second-hop property.
    for f in qb.filters:
        segments = f.get("path_segments") or []
        if len(segments) > 1:
            _append_trace(
                data,
                {"tool": "inspect", "uri": segments[0]},
                required=True,
                is_answer=False,
            )
            target_type_uri = f.get("target_type_uri")
            if target_type_uri:
                _append_trace(
                    data,
                    {"tool": "inspect", "uri": target_type_uri},
                    required=True,
                    is_answer=False,
                )
        elif len(segments) == 1:
            pass
    
    # Discovery NL
    discovery_phrases = [
        f"I'm looking into the available information for {pluralize(type_label)}. ",
        f"I am currently checking the database for {type_label} records and have a task for you. ",
        f"I'm curious about the {type_label} entries. "
    ]
    data["nl"].append(random.choice(discovery_phrases))

    # 1. Build JSON Tool Call
    # Map our internal state to the tool schema
    tool_filters = []
    for f in qb.filters:
        tool_filters.append({
            "path": f["path_uri"],
            "operator": f["operator"],
            "value": f["value"]
        })
    
    tool_call = {
        "tool": "query_builder",
        "type": str(qb.root_type),
        "filters": tool_filters,
        "project": [str(p) for p in qb.projects]
    }
    
    _append_trace(
        data,
        tool_call,
        required=True,
        is_answer=True,
    )

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
