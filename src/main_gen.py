import json
from src.vm.core import GeneratorVM, GrammarBreak
import src.grammar.definitions as rules

def generate_sample():
    vm = GeneratorVM()
    root_node = rules.root()
    
    # Retry a few times if the grammar breaks at the top level
    for _ in range(5):
        try:
            root_node.expand(vm)
            break
        except GrammarBreak:
            vm = GeneratorVM() # reset
            continue
        except Exception as e:
            raise e

    # Construct final result
    nl_stack = vm.get_ctx('nl') or []
    nl_question = "".join(nl_stack)
    trace = vm.get_ctx('trace') or []
    
    return {
        "question": nl_question,
        "trace": trace
    }

def main():
    num_samples = 5
    print(f"Generating {num_samples} samples...")
    samples = []
    
    for i in range(num_samples):
        print(f"--- Generating Sample {i+1}/{num_samples} ---")
        try:
            sample = generate_sample()
            samples.append(sample)
            print("Question:", sample["question"])
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")

    output_file = "produced_samples.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"\nAll samples saved to {output_file}")

if __name__ == "__main__":
    main()
