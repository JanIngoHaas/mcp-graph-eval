import os
import sys
from src.grammar.ontology import OntologySampler

def test():
    try:
        sampler = OntologySampler()
        print(f"Connecting to: {sampler.endpoint_url}")
        
        print("Discovering a random class...")
        cls = sampler.get_random_class()
        print(f"Selected Class: {cls.label} (<{cls.uri}>)")
        
        print(f"Discovering properties for {cls.label}...")
        props = sampler.get_valid_properties(cls.uri)
        if not props:
            print("  No properties found.")
        for p in props[:5]:
            print(f"  Property: {p.label} (<{p.uri}>) [{p.range_type}]")
            
        print(f"Sampling a random entity of type {cls.label}...")
        ent = sampler.get_random_entity(cls.uri)
        print(f"Selected Entity: {ent.label} (<{ent.uri}>)")
        
    except Exception as e:
        print(f"Error during discovery: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test()
