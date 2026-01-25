import collections
from typing import List, Dict

def filter_out_boring_stuff(uris: List[str]) -> List[str]:
    """Filters out standard RDF/OWL/XSD namespaces from a list of URIs."""
    boring_ns = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2001/XMLSchema#"
    ]
    
    filtered = []
    for uri in uris:
        # Skip standard technical namespaces
        if any(uri.startswith(ns) for ns in boring_ns):
            continue
            
        # Skip common internal/blank-node ID patterns
        local_part = uri.split("/")[-1].split("#")[-1].lower()
        if local_part.startswith(("bn", "node", "_:")):
            continue
            
        filtered.append(uri)
    return filtered

def group_properties(results: List[Dict], val_key: str, prefixes: Dict[str, str]) -> Dict[str, List[Dict]]:
    """Groups SPARQL results by property (p) and returns a list of raw nodes for each."""
    groups = collections.defaultdict(list)
    skip_uris = {prefixes["rdf"] + "type", prefixes["rdfs"] + "label"}
    for res in results:
        p_uri = res["p"]["value"]
        if p_uri in skip_uris:
            continue
        groups[p_uri].append(res[val_key])
    return dict(groups)

