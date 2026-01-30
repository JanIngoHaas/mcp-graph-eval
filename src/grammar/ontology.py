from ..vm.core import RetrySignal
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, cast
from enum import Enum, auto
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.term import Identifier
from rdflib.namespace import RDF, RDFS, OWL, XSD, split_uri
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import ResultRow
from dotenv import load_dotenv

from .language_utils import humanize_label, pluralize
from .sparql_utils import filter_out_boring_stuff, group_by_predicate

# Load .env file for configuration
load_dotenv()

@dataclass(frozen=True)
class TypeNode:
    uri: URIRef
    label: str

    @property
    def plural(self) -> str:
        return pluralize(self.label)

class PropertyRange(Enum):
    OBJECT = auto()
    DATATYPE = auto()

@dataclass(frozen=True)
class PropertyNode:
    uri: URIRef
    label: str
    range_kind: PropertyRange
    values: List[Identifier] # List of rdflib Identifiers (URIRef, Literal, etc.)

@dataclass(frozen=True)
class EntityNode:
    uri: URIRef
    label: str
    type_uri: URIRef



# Global persistent cache for all SPARQL queries
_QUERY_CACHE: Dict[str, List[ResultRow]] = {}

class OntologySampler:
    """
    Interface to an actual RDF Knowledge Graph via SPARQL using rdflib.
    """
    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("SPARQL_ENDPOINT")
        if not self.endpoint_url:
            raise ValueError("SPARQL_ENDPOINT must be specified in .env or constructor")
        
        self.store = SPARQLStore(self.endpoint_url)
        self.graph = Graph(self.store, identifier=None)
        
        # Rate-limiting delay from environment or default to 0.5s
        self.query_delay = float(os.getenv("SPARQL_QUERY_DELAY", 0.5))

        # Standard prefixes
        self.ns = {
            "rdf": RDF,
            "rdfs": RDFS,
            "owl": OWL,
            "xsd": XSD
        }
        for prefix, uri in self.ns.items():
            self.graph.bind(prefix, uri)

        self.hoppable_types = self._parse_type_list(os.getenv("SPARQL_HOPPABLE_TYPES", ""))

    def _query(self, sparql: str, use_cache: bool = False) -> List[ResultRow]:
        """Executes a SPARQL query via rdflib and returns list of rows."""
        if use_cache and sparql in _QUERY_CACHE:
            return _QUERY_CACHE[sparql]

        if self.query_delay > 0:
            time.sleep(self.query_delay)

        res = self.graph.query(sparql)
        rows = cast(List[ResultRow], list(res))
        if use_cache and rows:
            _QUERY_CACHE[sparql] = rows
        return rows

    def _parse_type_list(self, raw: str) -> List[URIRef]:
        """Parses a comma-separated list of type URIs (or prefixed names if known)."""
        if not raw:
            return []
        out: List[URIRef] = []
        for item in raw.split(","):
            token = item.strip()
            if not token:
                continue
            if token.startswith("<") and token.endswith(">"):
                token = token[1:-1].strip()
            if "://" in token:
                out.append(URIRef(token))
                continue
            if ":" in token:
                prefix, local = token.split(":", 1)
                ns = self.ns.get(prefix)
                if ns:
                    out.append(URIRef(str(ns) + local))
            # Silently skip unknown prefixes.
        return out

    def get_label(self, uri: URIRef) -> str:
        """Attempts to find a professional label for a URI using a prioritized SPARQL query."""
        sparql = f"""
        SELECT ?label WHERE {{
            <{uri}> rdfs:label ?label .
        }} LIMIT 1
        """
        results = self._query(sparql, use_cache=True)
        
        resolved_label = str(results[0][0]) if results else None
        if not resolved_label:
            resolved_label = self._get_local_name(uri)
        
        return humanize_label(resolved_label)

    def _get_local_name(self, uri: URIRef) -> str:
        if "schema#label" in uri:
            return "label"

        msg = "Failed to find label for URI: " + str(uri)
        raise RetrySignal(msg)

        # """Extracts and humanizes the local part of a URI."""
        # try:
        #     _, local = split_uri(uri)
        # except (ValueError, TypeError):
        #     # Handle opaque or non-conforming URIs
        #     s_uri = str(uri).rstrip("/#")
        #     local = s_uri.replace("#", "/").rsplit("/", 1)[-1]
        
        # return local

    def get_random_type(self) -> TypeNode:
        """
        Fetches a random RDF class directly (avoids bias from labeled entities).
        """
        sparql = """
        SELECT DISTINCT ?type ?label WHERE {
            ?s a ?type .
            ?type rdfs:label ?label .
        } ORDER BY RAND() LIMIT 1
        """
        rows = self._query(sparql, use_cache=False)
        if not rows:
            raise RetrySignal("No types found")

        type_uri, label_lit = rows[0]
        return TypeNode(cast(URIRef, type_uri), str(label_lit))

    def get_all_properties(self) -> List[PropertyNode]:
        """Returns ALL properties in the graph (both data and object properties) defined via types."""
        sparql = """
        SELECT DISTINCT ?p WHERE {
            VALUES ?t { owl:ObjectProperty owl:DatatypeProperty rdf:Property }
            ?p a ?t .
        }
        """
        results = self._query(sparql)
        
        props = []
        for row in results:
            p_uri = cast(URIRef, row[0])
            label = self.get_label(p_uri)
            props.append(PropertyNode(p_uri, label, PropertyRange.OBJECT, values=[]))
        return props

    def get_random_property_excluding(self, forbidden_uris: set[URIRef]) -> PropertyNode:
        """
        Returns a random property that is NOT in the forbidden set.
        Used for generating 'impossible' questions.
        """
        all_props = self.get_all_properties()
        
        candidates = [p for p in all_props if p.uri not in forbidden_uris]
        
        if not candidates:
            msg = "No disjoint properties found for impossible question"
            raise RetrySignal(msg)
            
        return random.choice(candidates)

    def get_entity_properties(self, entity_uri: URIRef) -> List[PropertyNode]:
        """Returns all properties (data and object) outgoing from this entity."""
        return self.get_entity_data_properties(entity_uri) + self.get_entity_object_properties(entity_uri)

    def get_entity_data_properties(self, entity_uri: URIRef) -> List[PropertyNode]:
        """Properties pointing to Literals or 'dead-end' IRIs."""
        return self._discover_properties(entity_uri, "out", "isLiteral(?o) || (isIRI(?o) && NOT EXISTS { ?o ?p2 ?o2 })", PropertyRange.DATATYPE)

    def get_entity_object_properties(self, entity_uri: URIRef) -> List[PropertyNode]:
        """Properties pointing to label-bearing IRI targets (single-hop)."""
        # Require a labeled IRI target; no extra structure needed for one hop.
        filter_expr = "isIRI(?o) && EXISTS { ?o rdfs:label ?ol }"
        return self._discover_properties(entity_uri, "out", filter_expr, PropertyRange.OBJECT)

    def get_incoming_properties(self, entity_uri: URIRef) -> List[PropertyNode]:
        """Properties and their subjects pointing TO this specific entity."""
        # Ensure the subject has at least one OTHER property so the hop is worth it (and is not a blank node)
        filter_expr = f"isIRI(?s) && EXISTS {{ ?s ?p2 ?o2 . FILTER(?o2 != <{entity_uri}>) }}"
        return self._discover_properties(entity_uri, "in", filter_expr, PropertyRange.OBJECT)

    def _discover_properties(self, entity_uri: URIRef, direction: str, sparql_filter: str, range_kind: PropertyRange) -> List[PropertyNode]:
        """Unified helper to fetch and group properties in either direction."""
        pattern = f"<{entity_uri}> ?p ?o" if direction == "out" else f"?s ?p <{entity_uri}>"
        
        sparql = f"SELECT DISTINCT ?p ?{'o' if direction == 'out' else 's'} WHERE {{ ?p rdfs:label ?l . {pattern} . FILTER({sparql_filter}) }}"
        results = self._query(sparql)
        
        # Group by property
        groups = group_by_predicate(results)
        
        props = []
        for p_uri, vals in groups.items():
            if filter_out_boring_stuff([str(p_uri)]):
                label = self.get_label(p_uri)
                props.append(PropertyNode(p_uri, label, range_kind, values=vals))
        return props

    def get_random_entity(self, type_uri: Optional[URIRef] = None) -> EntityNode:
        """
        Samples a random labeled entity (optionally of a given type) with ORDER BY RAND().
        """

        if type_uri:
            type_clause = f"?s a <{type_uri}> .\n            BIND(<{type_uri}> AS ?type)"
        else:
            type_clause = "?s a ?type ."

        sparql = f"""
        SELECT ?s ?type ?label WHERE {{
            ?s <{str(RDFS.label)}> ?label .
            FILTER(isIRI(?s))
            {type_clause}
        }} ORDER BY RAND() LIMIT 1
        """
        rows = self._query(sparql, use_cache=False)
        if not rows:
            msg = f"No entities found (type={type_uri})"
            raise RetrySignal(msg)

        uri, result_type, label_lit = rows[0]
        entity_type = result_type or type_uri
        if not entity_type:
            msg = "Couldn't retrieve a type for random entity"
            raise RetrySignal(msg)
        label = str(label_lit)
        return EntityNode(cast(URIRef, uri), label, cast(URIRef, entity_type))

    def get_random_hoppable_entity(self, type_uri: Optional[URIRef] = None, max_attempts: int = 5) -> EntityNode:
        """
        Samples an entity that has at least one outgoing object property.
        If SPARQL_HOPPABLE_TYPES is set, only those types are considered.
        """
        type_candidates = [type_uri] if type_uri else list(self.hoppable_types)
        if not type_candidates:
            type_candidates = [None]

        for _ in range(max_attempts):
            chosen = random.choice(type_candidates)
            node = self.get_random_entity(chosen)
            if self.get_entity_object_properties(node.uri):
                return node

        raise RetrySignal("Failed to find a hoppable entity after retries")

    def resolve_entity(self, uri: URIRef) -> EntityNode:
        """Fetches the label and type for a specific URI to construct a proper EntityNode."""
        sparql = f"""
        SELECT ?label ?type WHERE {{
            <{uri}> a ?type .
            OPTIONAL {{ <{uri}> rdfs:label ?label . }}
        }} LIMIT 1
        """
        results = self._query(sparql)
        if not results:
             # Fallback if no type is found
             label = self.get_label(uri)
             return EntityNode(uri, label, self.ns["owl"].Thing)
            
        # Filter types to avoid boring ones like owl:NamedIndividual
        type_uris = [row[1] for row in results]
        filtered_types = filter_out_boring_stuff([str(u) for u in type_uris])
        type_uri = URIRef(filtered_types[0]) if filtered_types else type_uris[0]
        
        row = results[0]
        label = str(row[0]) if row[0] else self.get_label(uri)
        return EntityNode(uri, label, cast(URIRef, type_uri))


    def sample_random_literal_value(self, prop_uri: URIRef) -> Literal:
        """Samples a literal value for a given property."""
        sparql = f"SELECT ?o WHERE {{ ?s <{prop_uri}> ?o . FILTER(isLiteral(?o)) }} ORDER BY RAND() LIMIT 1"
        rows = self._query(sparql, use_cache=False)
        if rows:
            return cast(Literal, rows[0])
        return Literal("value")
