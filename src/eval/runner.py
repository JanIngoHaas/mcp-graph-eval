import argparse
import asyncio
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> bool:
        return False

from src.eval.ground_truth import (
    execute_trace_ground_truth,
)
from src.eval.langchain_adapter import (
    EVAL_RECURSION_LIMIT,
    LangChainAdapter,
    LLM_IS_DETERMINISTIC,
)
from src.eval.prompts import get_agent_system_prompt
from src.scoring.io_utils import dump_data, infer_format, load_data

EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "1"))

# --- Models ---
EVAL_MODELS = [
    # "qwen3:4b-instruct-2507-q8_0",
    # "qwen3:8b-q8_0",
    # "ministral-3:14b",
    # "devstral-small-2:24b",
    # "nemotron-3-nano:30b",
    # "glm-4.7-flash:q8_0",
    # "gpt-oss:120b",
    # "devstral-2:123b",
     "gemini-3-flash-preview:cloud",  # previously active
     "kimi-k2.5:cloud",  # previously active
     "glm-4.7:cloud",  # previously active
     "devstral-2:123b-cloud",  # previously active
     "gpt-oss:120b-cloud",  # previously active
     "nemotron-3-nano:30b-cloud",  # previously active
     "qwen3-coder-next:cloud",  # previously active
     "ministral-3:3b-cloud",  # previously active
     "ministral-3:8b-cloud",
     "ministral-3:14b-cloud"  # previously active
]


def load_json(path: str) -> List:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return []


def _normalize_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens", usage.get("total"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    normalized: Dict[str, int] = {}
    if input_tokens is not None:
        normalized["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        normalized["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        normalized["total_tokens"] = int(total_tokens)
    return normalized


def _summarize_token_usage(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    found = False
    for entry in entries:
        raw_list: List[Dict[str, Any]] = []
        raw = entry.get("usage") or entry.get("token_usage") or {}
        if isinstance(raw, list):
            raw_list.extend(raw)
        elif raw:
            raw_list.append(raw)
        if not raw_list:
            for event in entry.get("runtime_trace") or []:
                if isinstance(event, dict) and event.get("type") == "token_usage":
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        raw_list.append(usage)

        for item in raw_list:
            usage = _normalize_usage(item)
            if not usage:
                continue
            found = True
            for key, value in usage.items():
                totals[key] = totals.get(key, 0) + value
    return totals if found else {}


def get_task_list(data: List, output_path: str, limit: int, model_name: str) -> Tuple[List[int], Dict]:
    """Determine which sample IDs need to be processed, prioritizing unfinished ones."""
    if os.path.exists(output_path):
        try:
            existing_results = load_data(Path(output_path))
        except Exception as e:
            print(f"Warning: Could not load {output_path}: {e}")
            existing_results = {}
    else:
        existing_results = {}

    if existing_results:
        print(f"Loaded existing results from {output_path}.")
        if existing_results.get("metadata", {}).get("model") != model_name:
            raise ValueError(
                f"Existing results are for model '{existing_results.get('metadata', {}).get('model')}', "
                f"not '{model_name}'"
            )
    else:
        print(f"No existing results found at {output_path}. Creating new file.")
        existing_results = {
            "metadata": {
                "model": model_name,
                "prompt": get_agent_system_prompt(),
                "deterministic": LLM_IS_DETERMINISTIC,
                "created_at": datetime.now().isoformat(),
                "token_usage": {},
            },
            "output": [],
        }

    existing_results.setdefault("metadata", {})
    existing_results["metadata"]["token_usage"] = _summarize_token_usage(existing_results.get("output", []))

    processed_ids = {r["id"] for r in existing_results.get("output", []) if "id" in r}

    todo_ids = []
    for idx in range(len(data)):
        if idx not in processed_ids:
            todo_ids.append(idx)
        if len(todo_ids) >= limit:
            break

    if not todo_ids and processed_ids:
        choice = input(
            f"\nAll samples for model '{model_name}' are already processed.\n"
            f"Do you want to FORCE re-run the first {limit} samples? [y/N]: "
        ).lower()
        if choice == "y":
            todo_ids = list(range(min(limit, len(data))))
            existing_results["output"] = [
                r for r in existing_results.get("output", []) if r.get("id") not in todo_ids
            ]
        else:
            print(f"Skipping model '{model_name}' - everything is up to date.")

    return todo_ids, existing_results


def _expected_metadata_for_output(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hop_bridge_predicate_uri": sample.get("hop_bridge_predicate_uri"),
        "hop_target_count": sample.get("hop_target_count"),
        "hop_scope": sample.get("hop_scope"),
    }


async def process_single_sample(
    idx: int,
    sample: Dict[str, Any],
    adapter: LangChainAdapter,
    results: Dict,
    output_path: str,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
):
    """Process one sample end-to-end and persist incrementally."""
    async with semaphore:
        print(f"--- Starting Sample {idx} ---")
        start_time = time.perf_counter()
        gold_trace = list(sample.get("trace", []))
        qtype = sample.get("qtype", "unknown")

        try:
            gold_triples = await execute_trace_ground_truth(gold_trace, adapter.client)

            entry = {
                "id": idx,
                "question": sample.get("question"),
                "qtype": qtype,
                "expected": {
                    "triples": gold_triples,
                    "trace": gold_trace,
                    **_expected_metadata_for_output(sample),
                },
                "received": {"triples": [], "trace": []},
                "runtime_trace": [],
                "answer": None,
                "error": None,
                "usage": [],
            }

            res = await adapter.answer_question(entry["question"])
            entry["received"] = {
                "triples": res.citations_data,
                "trace": res.explanation_data,
            }
            entry["runtime_trace"] = list(res.runtime_trace or [])
            entry["answer"] = res.answer
            entry["usage"] = res.token_usage

            if res.answer is not None:
                print("--- Model Answer ---")
                print(res.answer)
                print("--- End Model Answer ---")
            print(f"--- Finished Sample {idx} (Received {len(res.citations_data)} triples) ---")

        except Exception as e:
            traceback.print_exc()
            entry = {
                "id": idx,
                "question": sample.get("question"),
                "qtype": qtype,
                "expected": {
                    "triples": [],
                    "trace": gold_trace,
                    **_expected_metadata_for_output(sample),
                },
                "received": {"triples": [], "trace": []},
                "runtime_trace": [],
                "answer": None,
                "error": str(e),
                "usage": [],
            }
            print(f"--- Error on Sample {idx}: {e} ---")
        finally:
            entry["elapsed_s"] = time.perf_counter() - start_time

        async with lock:
            results["output"].append(entry)
            results["output"].sort(key=lambda x: x.get("id", 0))
            results.setdefault("metadata", {})
            results["metadata"]["token_usage"] = _summarize_token_usage(results["output"])
            fmt = infer_format(Path(output_path), None)
            dump_data(results, Path(output_path), fmt)


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
        tasks.append(
            process_single_sample(
                idx,
                sample,
                adapter,
                results,
                output_path,
                semaphore,
                lock,
            )
        )

    await asyncio.gather(*tasks)


def get_output_path_for_model(model_name: str, base_output: str) -> str:
    safe_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
    base, ext = os.path.splitext(base_output)
    return f"{base}_{safe_name}{ext}"


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run kg-mcp evaluation and save raw results.")
    parser.add_argument("--samples", type=str, default="produced_samples.json", help="Path to samples JSON")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to run per model")
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.toml",
        help="Base path for output files (model name will be appended)",
    )
    args = parser.parse_args()

    all_data = load_json(args.samples)
    if not all_data:
        print(f"Error: No data found in {args.samples}")
        return

    limit = len(all_data) if args.limit is None else args.limit

    print(f"\n{'=' * 60}")
    print("MCP Graph Evaluation Runner")
    print(f"{'=' * 60}")
    print("DISCLAIMER: Experimental test run. Only the models listed below are evaluated.")
    print(f"  Samples file: {args.samples}")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Limit per model: {limit}")
    print(f"  Agent recursion_limit: {EVAL_RECURSION_LIMIT}")
    print(f"  Models to evaluate: {len(EVAL_MODELS)}")
    for m in EVAL_MODELS:
        print(f"    - {m}")
    print(f"{'=' * 60}\n")

    for model_name in EVAL_MODELS:
        print(f"\n{'=' * 60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'=' * 60}")

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

    print(f"\n{'=' * 60}")
    print("All models evaluated. Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
