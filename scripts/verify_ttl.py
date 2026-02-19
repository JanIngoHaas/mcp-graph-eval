from rdflib import Graph
import sys

g = Graph()
try:
    print("Reading ontology_v2.ttl...")
    g.parse("datasets/wiproflex/ontology_v2.ttl", format="turtle")
    print("Reading instances_v2.ttl...")
    g.parse("datasets/wiproflex/instances_v2.ttl", format="turtle")
    print("Graph parsed successfully. Total triples:", len(g))
except Exception as e:
    print("Error parsing graph:", e)
    sys.exit(1)
