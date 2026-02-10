from random import seed
from collections import Counter
import json
import traceback
from src.vm.core import GeneratorVM
import src.grammar.definitions as rules

def generate_samples(num_samples: int):
    vm = GeneratorVM(initial_ctx={
        "samples": [],
        "nl": [],
        "trace": [],
        "s_entities": [],
        "s_facts": [],
        "answer_triples": [],
        "qtype": None,
        "qb": None,
        "hop_bridge_predicate_uri": None,
        "hop_target_count": None,
        "hop_scope": None,
    })
    root_node = rules.root(num_samples)

    vm.reset()
    root_node.expand(vm)

    return vm.get_ctx("samples") or []

def main():
    seed(3952356)
    num_samples = 150
    print(f"Generating {num_samples} samples...")
    qtype_counts = Counter()

    try:
        samples = generate_samples(num_samples)
    except Exception as e:
        print(f"Error generating samples: {e}")
        traceback.print_exc()
        samples = []


    for i, sample in enumerate(samples, start=1):
        print(f"--- Generated Sample {i}/{num_samples} ---")
        if sample.get("qtype"):
            qtype_counts[sample["qtype"]] += 1
        print("Question:", sample.get("question", ""))

    output_file = "produced_samples.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"\nAll samples saved to {output_file}")
    print("\nQuestion type counts:")
    total = 0
    for qtype in ["direct", "hop", "impossible", "query_builder"]:
        single = qtype_counts.get(qtype, 0)
        print(f"- {qtype}: {single}")
        total += single
    print("Total: ", total)

if __name__ == "__main__":
    main()
