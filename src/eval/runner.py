import json
import asyncio
import argparse
import random
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import Levenshtein
from sklearn.metrics import precision_recall_fscore_support
from abc import ABC, abstractmethod

from src.eval.langchain_adapter import LangChainAdapter
from src.eval.harness import AgentResult

# --- Metrics Calculation ---

def calculate_triple_f1(gold_triples: List[Dict], predicted_triples: List[Dict]) -> Tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F1 for triples.
    Triples are normalized to (subject, predicate, object) strings for comparison.
    """
    def normalize_triple(t):
        return (str(t.get("subject")), str(t.get("predicate")), str(t.get("object")))

    gold_set = set(normalize_triple(t) for t in gold_triples)
    pred_set = set(normalize_triple(t) for t in predicted_triples)

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

async def main():
    parser = argparse.ArgumentParser(description="Evaluate kg-mcp using LangChain adapter.")
    parser.add_argument("--samples", type=str, default="produced_samples.json", help="Path to samples JSON")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to run")
    parser.add_argument("--offset", type=int, default=0, help="Offset to start from")
    args = parser.parse_args()

    print(f"Loading samples from {args.samples}...")
    with open(args.samples, "r") as f:
        data = json.load(f)

    # Filter samples that actually implement answer_triples (some might be empty check first)
    # Actually, we should run on the valid ones.
    
    samples_to_run = data[args.offset : args.offset + args.limit]
    
    adapter = LangChainAdapter()
    
    total_f1 = 0.0
    total_trace_sim = 0.0
    success_count = 0
    
    print(f"\nStarting Evaluation of {len(samples_to_run)} samples...\n")
    
    results = []

    for i, sample in enumerate(samples_to_run):
        idx = args.offset + i
        question = sample.get("question")
        gold_trace = sample.get("trace", [])
        gold_triples = sample.get("answer_triples", [])
        
        print(f"--- Sample {idx} ---")
        print(f"Q: {question}")
        
        try:
            result = await adapter.answer_question(question)
            
            # 1. Triple Evaluation
            if gold_triples:
                prec, rec, f1 = calculate_triple_f1(gold_triples, result.citations_data)
                f1_str = f"F1: {f1:.4f} (P: {prec:.4f}, R: {rec:.4f})"
            else:
                f1 = 0.0 # Or None? Let's treat as N/A but 0 for sum if simplistic
                prec, rec = 0.0, 0.0
                f1_str = "F1: N/A (No Gold Triples)"
            
            # 2. Trace Evaluation
            trace_sim = calculate_trace_similarity(gold_trace, result.explanation_data)
            
            print(f"A: {result.answer[:100]}...")
            print(f"   -> {f1_str}")
            print(f"   -> Trace Sim: {trace_sim:.4f}")
            
            # Only count F1 in average if applicable? 
            # For simplicity, if no gold triples, F1 is technically undefined or 1.0 if we cited nothing?
            # Let's say: if gold is empty, and we cited nothing -> F1=1.0. If we cited something -> F1=0.
            # But the user said "trace is enough". So let's just log it.
            
            if gold_triples:
                total_f1 += f1
                valid_f1_samples = locals().get("valid_f1_samples", 0) + 1
            else:
                 valid_f1_samples = locals().get("valid_f1_samples", 0)
            
            total_trace_sim += trace_sim
            success_count += 1
            
            results.append({
                "id": idx,
                "f1": f1 if gold_triples else None,
                "trace_sim": trace_sim,
                "error": None
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "id": idx,
                "f1": 0.0,
                "trace_sim": 0.0,
                "error": str(e)
            })

    # Global Metrics
    n = len(samples_to_run)
    n_f1 = locals().get("valid_f1_samples", 0)
    
    avg_f1 = total_f1 / n_f1 if n_f1 > 0 else 0.0
    avg_trace_sim = total_trace_sim / n if n > 0 else 0.0
    success_rate = (success_count / n) * 100 if n > 0 else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Samples: {n}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Avg Triple F1: {avg_f1:.4f} (over {n_f1} samples)")
    print(f"Avg Trace Sim: {avg_trace_sim:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
