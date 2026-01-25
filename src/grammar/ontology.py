import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum, auto
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.term import Identifier
from rdflib.namespace import RDF, RDFS, OWL, XSD, split_uri
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from dotenv import load_dotenv

from .language_utils import humanize_label
from .sparql_utils import filter_out_boring_stuff, group_by_predicate

# Load .env file for configuration
load_dotenv()

@dataclass(frozen=True)
class TypeNode:
    uri: URIRef
    label: str

    @property
    def plural(self) -> str:
        label = self.label.strip()
        if not label:
            return "items"
        if label.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return f"{label}es"
        if label.endswith('y') and not label.endswith(('ay', 'ey', 'iy', 'oy', 'uy')):
            return f"{label[:-1]}ies"
        return f"{label}s"

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
_QUERY_CACHE: Dict[str, List[tuple]] = {}

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

    def _query(self, sparql: str) -> List[tuple]:
        """Executes a SPARQL query via rdflib and returns list of rows."""
        if sparql in _QUERY_CACHE:
            print(f"DEBUG: Cache hit for query: {sparql}")
            return _QUERY_CACHE[sparql]

        if self.query_delay > 0:
            time.sleep(self.query_delay)
        print(f"DEBUG: Executing SPARQL query to {self.endpoint_url} via rdflib...")
        start_time = time.time()
        
        res = self.graph.query(sparql)
        rows = list(res)
        
        elapsed = time.time() - start_time
        print(f"DEBUG: Query finished in {elapsed:.2f}s")
        
        _QUERY_CACHE[sparql] = rows
        return rows

    def get_label(self, uri: URIRef) -> str:
        """Attempts to find a professional label for a URI using a prioritized SPARQL query."""
        sparql = f"""
        SELECT ?label WHERE {{
            <{uri}> rdfs:label ?label .
        }} LIMIT 1
        """
        results = self._query(sparql)
        
        resolved_label = str(results[0][0]) if results else None
        
        if not resolved_label:
            # 2. Syntactic Fallback
            resolved_label = self._get_local_name(uri)
        
        return humanize_label(resolved_label)

    def _get_local_name(self, uri: URIRef) -> str:
        if "schema#label" in uri:
            return "label"

        return uri
        # """Extracts and humanizes the local part of a URI."""
        # try:
        #     _, local = split_uri(uri)
        # except (ValueError, TypeError):
        #     # Handle opaque or non-conforming URIs
        #     s_uri = str(uri).rstrip("/#")
        #     local = s_uri.replace("#", "/").rsplit("/", 1)[-1]
        
        # return local

    def get_random_type(self) -> TypeNode:
        """Fetches a random RDF class (type) that has actual instances in the graph."""
        sparql = "SELECT DISTINCT ?type WHERE { ?s a ?type . }"
        results = self._query(sparql)
        if not results:
            raise RuntimeError(f"No classes found at {self.endpoint_url}")
        
        uris = [row[0] for row in results]
        filtered = filter_out_boring_stuff([str(u) for u in uris])
        uri = URIRef(random.choice(filtered)) if filtered else URIRef(uris[0])

        label = self.get_label(uri)
        return TypeNode(uri, label)

    def get_entity_properties(self, entity_uri: URIRef) -> List[PropertyNode]:
        """Returns all properties (data and object) outgoing from this entity."""
        return self.get_entity_data_properties(entity_uri) + self.get_entity_object_properties(entity_uri)

    def get_entity_data_properties(self, entity_uri: URIRef) -> List[PropertyNode]:
        """Properties pointing to Literals or 'dead-end' IRIs."""
        return self._discover_properties(entity_uri, "out", "isLiteral(?o) || (isIRI(?o) && NOT EXISTS { ?o ?p2 ?o2 })", PropertyRange.DATATYPE)

    def get_entity_object_properties(self, entity_uri: URIRef) -> List[PropertyNode]:
        """Properties pointing to entities with internal structure."""
        return self._discover_properties(entity_uri, "out", "isIRI(?o) && EXISTS { ?o ?p2 ?o2 }", PropertyRange.OBJECT)

    def get_incoming_properties(self, entity_uri: URIRef) -> List[PropertyNode]:
        """Properties and their subjects pointing TO this specific entity."""
        # Ensure the subject has at least one OTHER property so the hop is worth it (and is not a blank node)
        filter_expr = f"isIRI(?s) && EXISTS {{ ?s ?p2 ?o2 . FILTER(?o2 != <{entity_uri}>) }}"
        return self._discover_properties(entity_uri, "in", filter_expr, PropertyRange.OBJECT)

    def _discover_properties(self, entity_uri: URIRef, direction: str, sparql_filter: str, range_kind: PropertyRange) -> List[PropertyNode]:
        """Unified helper to fetch and group properties in either direction."""
        pattern = f"<{entity_uri}> ?p ?o" if direction == "out" else f"?s ?p <{entity_uri}>"
        
        sparql = f"SELECT DISTINCT ?p ?{'o' if direction == 'out' else 's'} WHERE {{ {pattern} . FILTER({sparql_filter}) }}"
        results = self._query(sparql)
        
        # Group by property
        groups = group_by_predicate(results)
        
        props = []
        for p_uri, vals in groups.items():
            if filter_out_boring_stuff([str(p_uri)]):
                label = self.get_label(p_uri)
                props.append(PropertyNode(p_uri, label, range_kind, values=vals))
        return props

    def get_random_entity(self, type_uri: URIRef) -> EntityNode:
        """Samples a random entity of the given type."""
        sparql = f"""
        SELECT ?s ?label WHERE {{
            ?s a <{type_uri}> .
            FILTER(isIRI(?s))
            # Skip skolemized/internal IDs that look like 'bn123' or 'node123'
            FILTER(!regex(str(?s), "(bn|node)[0-9]+|:_|#bn", "i"))
            OPTIONAL {{ ?s rdfs:label ?label }}
        }} ORDER BY RAND() LIMIT 1
        """
        results = self._query(sparql)
        if not results:
            raise ValueError(f"No entities of type {type_uri} found")
            
        uri, label_lit = results[0][0], results[0][1]
        label = str(label_lit) if label_lit else self.get_label(uri)
        return EntityNode(uri, label, type_uri)


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
        return EntityNode(uri, label, type_uri)


    def sample_random_literal_value(self, prop_uri: URIRef) -> Literal:
        """Samples a literal value for a given property."""
        sparql = f"SELECT ?o WHERE {{ ?s <{prop_uri}> ?o . FILTER(isLiteral(?o)) }} LIMIT 100"
        results = self._query(sparql)
        if results:
            return random.choice(results)[0]
        return Literal("value")
