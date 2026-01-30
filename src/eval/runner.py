import json
import asyncio
import argparse
import os
import traceback
import Levenshtein
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime

from src.eval.langchain_adapter import LangChainAdapter
from src.eval.harness import AgentResult
from src.eval.ground_truth import execute_trace_ground_truth
from src.eval.ground_truth import extract_triples_from_subgraph
from src.eval.prompts import get_agent_system_prompt

EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "1"))

# --- Configurable List of Models to Evaluate ---
EVAL_MODELS = [
    "lfm2.5-1.2b-thinking",
    "mistralai/ministral-3-14b-reasoning",
    # Add more models here as needed
]

# --- Metrics Calculation ---

def calculate_triple_f1(gold_triples: List[Dict], predicted_triples: List[Dict]) -> Tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F1 for triples.
    Triples are normalized to (subject, predicate, object) strings for comparison.
    """
    def get_all_triples(t_list):
        all_triples = set()
        for t in t_list:
            subj = t.get("subject")
            pred = t.get("predicate")
            obj = t.get("object")
            if subj:
                all_triples.add((str(subj), str(pred), str(obj)))
            elif "ttl" in t:
                for s, p, o in extract_triples_from_subgraph(t["ttl"]):
                    all_triples.add((str(s), str(p), str(o)))
        return all_triples

    gold_set = get_all_triples(gold_triples)
    pred_set = get_all_triples(predicted_triples)

    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set) - tp
    fn = len(gold_set) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def calculate_trace_similarity(gold_trace: List[Dict], predicted_trace: List[Dict]) -> float:
    """
    Calculates a similarity score (0-1) between the gold usage trace and the predicted trace.
    Aligns steps fundamentally by Tool Name (with inspect==fact rule) and then compares arguments.
    """
    if not gold_trace and not predicted_trace:
        return 1.0
    if not gold_trace or not predicted_trace:
        return 0.0

    # 1. Normalize Tool Names
    def normalize_tool_name(tool: str) -> str:
        if tool == "inspect": return "fact"
        return tool

    # Simple approach: Longest Common Subsequence logic or similar alignment.
    # Given the sequential nature, we can iterate and look for best matches window-wise or just simple alignment.
    # For robustness, let's just do a greedy best-match for each gold step in the predicted trace order? 
    # Or strict sequence alignment?
    # Let's use Levenshtein distance on a "stringified" representation of the trace for a quick robust metric,
    # OR step-by-step comparison.
    
    # Step-by-step approach is more interpretable.
    
    score_sum = 0.0
    matches = 0
    
    # We will try to find the "best matching step" in the predicted trace for each gold step.
    # This ignores order slightly but allows for inserted/deleted steps by the agent.
    # BUT, to capture "Flow", strict order matters? 
    # Let's stick to the user's suggestion: "first check tool name... then syntactic similarity"
    
    # Let's map gold steps to predicted steps (greedy).
    used_pred_indices = set()
    
    for g_step in gold_trace:
        g_tool = normalize_tool_name(g_step.get("tool", ""))
        best_step_score = 0.0
        best_pred_idx = -1
        
        for i, p_step in enumerate(predicted_trace):
            if i in used_pred_indices: continue
            
            p_tool = normalize_tool_name(p_step.get("tool", ""))
            
            if g_tool == p_tool:
                # Calculate Detailed Argument Similarity
                # We serialize the arguments (excluding 'tool') to string
                def get_args_str(step):
                    return " ".join([f"{k}:{v}" for k, v in sorted(step.items()) if k not in ("tool", "explanation_key", "limit")]) # exclude internal keys and limit
                
                g_args = get_args_str(g_step)
                p_args = get_args_str(p_step)
                
                # Levenshtein Ratio: 0 to 1
                sim = Levenshtein.ratio(g_args, p_args)
                
                # Base score for matching tool is high? Or just use the arg similarity as the step score?
                # If args are totally different, maybe it shouldn't match.
                # Let's say: if tool matches, score is 0.5 + 0.5 * arg_sim
                current_score = 0.5 + (0.5 * sim)
                
                if current_score > best_step_score:
                    best_step_score = current_score
                    best_pred_idx = i
        
        if best_pred_idx != -1:
            score_sum += best_step_score
            used_pred_indices.add(best_pred_idx)
            matches += 1
            
    # Normalize by max length to penalize missing steps or extra hallucinations?
    # Or just by gold length (Recall-focused trace similarity)?
    # Let's use max(len_gold, len_pred) to valid total similarity.
    max_len = max(len(gold_trace), len(predicted_trace))
    final_score = score_sum / max_len if max_len > 0 else 0.0
    
    return final_score


# --- Main Runner ---

# --- Runner Helpers ---

def load_json(path: str) -> List:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return []

def save_json(path: str, data: List):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_task_list(data: List, output_path: str, limit: int, model_name: str) -> Tuple[List[int], Dict]:
    """Determine which sample IDs need to be processed, prioritizing unfinished ones."""
    existing_results = load_json(output_path)
    
    if existing_results:
        print(f"Loaded existing results from {output_path}.")
        # Validate model matches
        if existing_results.get("metadata", {}).get("model") != model_name:
            raise ValueError(f"Existing results are for model '{existing_results.get('metadata', {}).get('model')}', not '{model_name}'")
    else:
        print(f"No existing results found at {output_path}. Creating new file.")
        existing_results = {
            "metadata": {
                "model": model_name,

                "prompt": get_agent_system_prompt(),
                "created_at": datetime.now().isoformat(),
            },
            "output": []
        }

    processed_ids = {r["id"] for r in existing_results.get("output", []) if "id" in r}
    
    # Resume-friendly logic: Find the first X unfinished samples
    todo_ids = []
    for idx in range(len(data)):
        if idx not in processed_ids:
            todo_ids.append(idx)
        if len(todo_ids) >= limit:
            break

    if not todo_ids and processed_ids:
        choice = input(f"\nAll samples for model '{model_name}' are already processed.\nDo you want to FORCE re-run the first {limit} samples? [y/N]: ").lower()
        if choice == 'y':
            todo_ids = list(range(min(limit, len(data))))
            # Filter results to remove the ones we are about to overwrite
            existing_results["output"] = [r for r in existing_results.get("output", []) if r.get("id") not in todo_ids]
        else:
            print(f"Skipping model '{model_name}' - everything is up to date.")

    return todo_ids, existing_results

async def process_single_sample(
    idx: int, 
    sample: Dict, 
    adapter: LangChainAdapter, 
    results: List[Dict], 
    output_path: str, 
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock
):
    """Processes a single sample through the evaluation pipeline."""
    async with semaphore:
        print(f"--- Starting Sample {idx} ---")
        gold_trace = sample.get("trace", [])
        
        try:
            # 1. Compute ground truth authoritative triples
            gold_triples = await execute_trace_ground_truth(gold_trace, adapter.client)

            entry = {
                "id": idx,
                "question": sample.get("question"),
                "qtype": sample.get("qtype", "unknown"),
                "expected": {
                    "triples": gold_triples,
                    "trace": gold_trace
                },
                "received": {"triples": [], "trace": []},
                "answer": None,
                "error": None
            }

            # 2. Answer question via agent
            res = await adapter.answer_question(entry["question"])
            entry["received"] = {
                "triples": res.citations_data,
                "trace": res.explanation_data
            }
            entry["answer"] = res.answer
            print(f"--- Finished Sample {idx} (Received {len(res.citations_data)} triples) ---")
            
        except Exception as e:
            traceback.print_exc()
            entry = {
                "id": idx,
                "question": sample.get("question"),
                "qtype": sample.get("qtype", "unknown"),
                "expected": {"triples": [], "trace": gold_trace},
                "received": {"triples": [], "trace": []},
                "answer": None,
                "error": str(e)
            }
            print(f"--- Error on Sample {idx}: {e} ---")

        # 3. Thread-safe incremental save
        async with lock:
            results["output"].append(entry)
            results["output"].sort(key=lambda x: x.get("id", 0))
            save_json(output_path, results)

async def process_samples(todo_ids: List[int], all_data: List[Dict], results: Dict, output_path: str, model_name: str):
    """Core evaluation loop with concurrency support."""
    adapter = LangChainAdapter(model_name=model_name)
    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    lock = asyncio.Lock()
    
    print(f"\nEvaluation Plan (Concurrency={EVAL_CONCURRENCY}):")
    print(f"  - Model               : {model_name}")
    print(f"  - Total samples target: {len(todo_ids)}")
    print(f"  - Output file         : {output_path}\n")

    tasks = []
    for idx in todo_ids:
        sample = all_data[idx]
        tasks.append(process_single_sample(
            idx, sample, adapter, results, output_path, semaphore, lock
        ))
    
    await asyncio.gather(*tasks)

def get_output_path_for_model(model_name: str, base_output: str) -> str:
    """Generate output file path for a specific model."""
    # Convert model name to safe filename: "mistral:7b" -> "mistral_7b"
    safe_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
    base, ext = os.path.splitext(base_output)
    return f"{base}_{safe_name}{ext}"

async def main():
    parser = argparse.ArgumentParser(description="Run kg-mcp evaluation and save raw results.")
    parser.add_argument("--samples", type=str, default="produced_samples.json", help="Path to samples JSON")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to run per model")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Base path for output files (model name will be appended)")
    args = parser.parse_args()

    all_data = load_json(args.samples)
    if not all_data:
        print(f"Error: No data found in {args.samples}")
        return

    limit = len(all_data) if args.limit is None else args.limit
    
    print(f"\n{'='*60}")
    print(f"MCP Graph Evaluation Runner")
    print(f"{'='*60}")
    print(f"  Samples file: {args.samples}")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Limit per model: {limit}")
    print(f"  Models to evaluate: {len(EVAL_MODELS)}")
    for m in EVAL_MODELS:
        print(f"    - {m}")
    print(f"{'='*60}\n")

    for model_name in EVAL_MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        output_path = get_output_path_for_model(model_name, args.output)
        
        try:
            todo_ids, results = get_task_list(all_data, output_path, limit, model_name)
            
            if todo_ids:
                await process_samples(todo_ids, all_data, results, output_path, model_name)
            
            print(f"✓ Completed model: {model_name}")
        except Exception as e:
            print(f"✗ Error with model {model_name}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All models evaluated. Done.")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
