from random import seed
from collections import Counter
import json
import traceback
from src.vm.core import GeneratorVM
import src.grammar.definitions as rules

def generate_sample():
    vm = GeneratorVM(initial_ctx={
        'nl': [],
        'trace': [],
        's_entities': [],
        's_facts': [],
        'qtype': None
    })
    root_node = rules.root()
    
    vm.reset()
    root_node.expand(vm)
    
    # Construct final result
    nl_stack = vm.get_ctx('nl') or []
    nl_question = "".join(nl_stack)
    trace = vm.get_ctx('trace') or []
    qtype = vm.get_ctx('qtype')
    
    return {
        "question": nl_question,
        "trace": trace,
        "qtype": qtype
    }

def main():
    seed(395234)
    num_samples = 100
    print(f"Generating {num_samples} samples...")
    samples = []
    qtype_counts = Counter()
    
    for i in range(num_samples):
        print(f"--- Generating Sample {i+1}/{num_samples} ---")
        try:
            sample = generate_sample()
            if sample:
                samples.append(sample)
                if sample.get("qtype"):
                    qtype_counts[sample["qtype"]] += 1
                print("Question:", sample["question"])
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
            traceback.print_exc()

    output_file = "produced_samples.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"\nAll samples saved to {output_file}")
    print("\nQuestion type counts:")
    for qtype in ["direct", "hop", "impossible", "query_builder"]:
        print(f"- {qtype}: {qtype_counts.get(qtype, 0)}")

if __name__ == "__main__":
    main()
