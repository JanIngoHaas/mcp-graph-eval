from typing import List, Dict, Any
import collections
from rdflib.namespace import RDF, RDFS, OWL, XSD

def filter_out_boring_stuff(uris: List[str]) -> List[str]:
    """Filters out standard RDF/OWL/XSD namespaces from a list of URIs."""
    boring_ns = {RDF, RDFS, OWL, XSD}
    
    filtered = []
    for val in uris:
        uri = str(val)
        # Skip standard technical namespaces
        if any(uri.startswith(str(ns)) for ns in boring_ns):
            continue
            
        # Skip common internal/blank-node ID patterns
        local_part = uri.split("/")[-1].split("#")[-1].lower()
        if local_part.startswith(("bn", "node", "_:")):
            continue
            
        filtered.append(uri)
    return filtered

def group_by_predicate(results: List[Any]) -> Dict[Any, List[Any]]:
    """Groups rdflib SPARQL results by the first element (predicate) and returns lists of the second element."""
    groups = collections.defaultdict(list)
    skip_uris = {RDF.type, RDFS.label}
    for row in results:
        p_uri, val = row[0], row[1]
        if p_uri in skip_uris:
            continue
        groups[p_uri].append(val)
    return dict(groups)
