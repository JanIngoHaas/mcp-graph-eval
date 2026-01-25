from src.grammar.ontology import OntologySampler

sampler = OntologySampler()
print(f"Endpoint: {sampler.endpoint_url}")

print("Fetching random class...")
cls = sampler.get_random_class()
print(f"Class: {cls.label} ({cls.uri})")

print(f"Fetching properties for {cls.label}...")
props = sampler.get_valid_properties(cls.uri)
for p in props[:5]:
    print(f"  Property: {p.label} ({p.uri}) [{p.range_type}]")

print(f"Fetching random entity of type {cls.label}...")
ent = sampler.get_random_entity(cls.uri)
print(f"Entity: {ent.label} ({ent.uri})")
