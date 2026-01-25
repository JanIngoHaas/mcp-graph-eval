import os
import random
import time
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load .env file for configuration
load_dotenv()

@dataclass(frozen=True)
class ClassNode:
    uri: str
    label: str
    plural: str

@dataclass(frozen=True)
class PropertyNode:
    uri: str
    label: str
    range_type: str # 'Literal' or 'Class'
    values: List[str] # All values found directly from discovery

@dataclass(frozen=True)
class EntityNode:
    uri: str
    label: str
    type_uri: str
    uuid: str # The unique identifier (URI in real RDF)

from .language_utils import humanize_label
from .sparql_utils import filter_out_boring_stuff, group_properties

# Global persistent cache for all SPARQL queries
_QUERY_CACHE: Dict[str, List[Dict[Any, Any]]] = {}

class OntologySampler:
    """
    Interface to an actual RDF Knowledge Graph via SPARQL.
    Truly generalized discovery without graph-specific assumptions.
    """
    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("SPARQL_ENDPOINT")
        if not self.endpoint_url:
            raise ValueError("SPARQL_ENDPOINT must be specified in .env or constructor")
        
        # Standard prefixes only
        self.prefixes = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        }

    def _query(self, sparql: str) -> List[Dict[ Any, Any]]:
        """Executes a SPARQL query and returns the results as a list of dicts."""
        if sparql in _QUERY_CACHE:
            print(f"DEBUG: Cache hit for query: {sparql}")
            return _QUERY_CACHE[sparql]

        time.sleep(0.5)
        prefix_str = "\n".join([f"PREFIX {k}: <{v}>" for k, v in self.prefixes.items()])
        full_query = prefix_str + "\n" + sparql
        
        # Logs for visibility
        print(f"DEBUG: Executing SPARQL query to {self.endpoint_url}...")
        start_time = time.time()
        
        response = requests.post(
            self.endpoint_url,
            data={"query": full_query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=60
        )
        response.raise_for_status()
        
        count = len(response.json().get("results", {}).get("bindings", []))
        elapsed = time.time() - start_time
        print(f"DEBUG: Query finished in {elapsed:.2f}s (results: {count})")
        
        results = response.json().get("results", {}).get("bindings", [])
        _QUERY_CACHE[sparql] = results
        return results

    def get_label(self, uri: str) -> str:
        """Attempts to find a professional label for a URI."""
        sparql = f"""
        SELECT ?label WHERE {{
            <{uri}> rdfs:label ?label .
        }} LIMIT 1
        """
        try:
            results = self._query(sparql)
            if results:
                label = humanize_label(results[0]["label"]["value"])
                if label:
                    return label
        except Exception:
            pass
        
        # Fallback to local name
        raw = uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
        return humanize_label(raw) or raw

    def get_random_class(self) -> ClassNode:
        """Fetches a random class that has actual instances in the graph, skipping boring namespaces."""
        # Query cache handles the heavy lifting
        sparql = "SELECT DISTINCT ?type WHERE { ?s a ?type . }"
        results = self._query(sparql)
        if not results:
            raise RuntimeError(f"No classes found in knowledge graph at {self.endpoint_url}")
        
        uris = [res["type"]["value"] for res in results]
        filtered = filter_out_boring_stuff(uris)
        uri = random.choice(filtered) if filtered else uris[0]

        label = self.get_label(uri)
        plural = f"{label}s" if not label.endswith('s') else label
        return ClassNode(uri, label, plural)

    def get_entity_properties(self, entity_uri: str) -> List[PropertyNode]:
        """Returns all properties (data and object) outgoing from this entity."""
        return self.get_entity_data_properties(entity_uri) + self.get_entity_object_properties(entity_uri)

    def get_entity_data_properties(self, entity_uri: str) -> List[PropertyNode]:
        """Properties pointing to Literals or 'dead-end' IRIs."""
        return self._discover_properties(entity_uri, "out", "isLiteral(?o) || (isIRI(?o) && NOT EXISTS { ?o ?p2 ?o2 })", "Literal")

    def get_entity_object_properties(self, entity_uri: str) -> List[PropertyNode]:
        """Properties pointing to entities with internal structure."""
        return self._discover_properties(entity_uri, "out", "isIRI(?o) && EXISTS { ?o ?p2 ?o2 }", "Class")

    def get_incoming_properties(self, entity_uri: str) -> List[PropertyNode]:
        """Properties and their subjects pointing TO this specific entity."""
        # Ensure the subject has at least one OTHER property so the hop is worth it (and is not a blank node)
        filter_expr = f"isIRI(?s) && EXISTS {{ ?s ?p2 ?o2 . FILTER(?o2 != <{entity_uri}>) }}"
        return self._discover_properties(entity_uri, "in", filter_expr, "Class")

    def _discover_properties(self, entity_uri: str, direction: str, sparql_filter: str, range_label: str) -> List[PropertyNode]:
        """Unified helper to fetch and group properties in either direction."""
        val_key = "o" if direction == "out" else "s"
        pattern = f"<{entity_uri}> ?p ?o" if direction == "out" else f"?s ?p <{entity_uri}>"
        
        sparql = f"SELECT DISTINCT ?p ?{val_key} WHERE {{ {pattern} . FILTER({sparql_filter}) }}"
        results = self._query(sparql)
        groups = group_properties(results, val_key, self.prefixes)
        
        props = []
        for p_uri, nodes in groups.items():
            if filter_out_boring_stuff([p_uri]):
                label = self.get_label(p_uri)
                values = [n["value"] for n in nodes]
                props.append(PropertyNode(p_uri, label, range_label, values=values))
        return props

    def get_random_entity(self, type_uri: str) -> EntityNode:
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
            
        res = results[0]
        uri = res["s"]["value"]
        label = res.get("label", {}).get("value") or self.get_label(uri)
        return EntityNode(uri, label, type_uri, uri)


    def get_entity_node(self, uri: str) -> EntityNode:
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
             return EntityNode(uri, label, self.prefixes["owl"] + "Thing", uri)
            
        res = results[0]
        label = res.get("label", {}).get("value") or self.get_label(uri)
        type_uri = res["type"]["value"]
        return EntityNode(uri, label, type_uri, uri)


    def sample_random_literal_value(self, prop_uri: str) -> str:
        """Samples a literal value for a given property."""
        sparql = f"SELECT ?o WHERE {{ ?s <{prop_uri}> ?o . }} LIMIT 100"
        results = self._query(sparql)
        if results:
            return random.choice(results)["o"]["value"]
        return "value"
