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

def filter_out_boring_stuff(uris: List[str]) -> List[str]:
    """Filters out standard RDF/OWL/XSD namespaces from a list of URIs."""
    boring_ns = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2001/XMLSchema#"
    ]
    return [uri for uri in uris if not any(uri.startswith(ns) for ns in boring_ns)]

from .utils import humanize_label

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

    def _get_label(self, uri: str) -> str:
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
        sparql = "SELECT DISTINCT ?type WHERE { [] a ?type . }"
        results = self._query(sparql)
        if not results:
            raise RuntimeError(f"No classes found in knowledge graph at {self.endpoint_url}")
        
        uris = [res["type"]["value"] for res in results]
        filtered = filter_out_boring_stuff(uris)
        uri = random.choice(filtered) if filtered else uris[0]

        label = self._get_label(uri)
        plural = f"{label}s" if not label.endswith('s') else label
        return ClassNode(uri, label, plural)

    def get_entity_properties(self, entity_uri: str) -> List[PropertyNode]:
        """Returns properties and all their objects that are actually OUTGOING from this specific entity."""
        sparql = f"""
        SELECT DISTINCT ?p ?o WHERE {{
            <{entity_uri}> ?p ?o .
        }}
        """
        results = self._query(sparql)
        
        # Group by property to detect range and collect all values
        p_data = {}
        for res in results:
            p_uri = res["p"]["value"]
            if p_uri in [self.prefixes["rdf"] + "type", self.prefixes["rdfs"] + "label"]:
                continue
            
            val_node = res["o"]
            if p_uri not in p_data:
                p_data[p_uri] = {"is_uri": False, "values": []}
            
            p_data[p_uri]["values"].append(val_node["value"])
            if val_node["type"] == "uri":
                p_data[p_uri]["is_uri"] = True

        props = []
        for p_uri, info in p_data.items():
            if filter_out_boring_stuff([p_uri]):
                label = self._get_label(p_uri)
                range_type = "Class" if info["is_uri"] else "Literal"
                props.append(PropertyNode(p_uri, label, range_type, values=info["values"]))
        return props

    def get_random_entity(self, type_uri: str) -> EntityNode:
        """Samples a random entity of the given type."""
        sparql = f"""
        SELECT ?s ?label WHERE {{
            ?s a <{type_uri}> .
            OPTIONAL {{ ?s rdfs:label ?label }}
        }} ORDER BY RAND() LIMIT 1
        """
        results = self._query(sparql)
        if not results:
            raise ValueError(f"No entities of type {type_uri} found")
            
        res = results[0]
        uri = res["s"]["value"]
        label = res.get("label", {}).get("value") or self._get_label(uri)
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
             label = self._get_label(uri)
             return EntityNode(uri, label, self.prefixes["owl"] + "Thing", uri)
            
        res = results[0]
        label = res.get("label", {}).get("value") or self._get_label(uri)
        type_uri = res["type"]["value"]
        return EntityNode(uri, label, type_uri, uri)

    def sample_random_literal_value(self, prop_uri: str) -> str:
        """Samples a literal value for a given property."""
        sparql = f"SELECT ?o WHERE {{ ?s <{prop_uri}> ?o . }} LIMIT 100"
        results = self._query(sparql)
        if results:
            return random.choice(results)["o"]["value"]
        return "value"
