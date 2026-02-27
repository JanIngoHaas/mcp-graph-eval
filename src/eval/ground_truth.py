import json
import re
import time
from typing import Dict, List, Set, Tuple

from langchain_mcp_adapters.client import MultiServerMCPClient
from rdflib import Graph


def extract_triples_from_subgraph(subgraph_ttl: str) -> Set[Tuple[str, str, str]]:
    """Extract all triples from a Turtle-formatted subgraph."""
    triples = set()
    try:
        g = Graph()
        g.parse(data=subgraph_ttl, format="turtle")
        for s, p, o in g:
            triples.add((str(s), str(p), str(o)))
    except Exception as e:
        print("WARNING: Error parsing TTL: ", e)
    return triples


def extract_citations(citations_data: list) -> List[Dict[str, str]]:
    """Extract triples from the citation resource data."""
    triples = set()
    for entry in citations_data:
        if isinstance(entry, dict) and "ttl" in entry:
            ttl_str = entry["ttl"]
            if ttl_str:
                try:
                    triples.update(extract_triples_from_subgraph(ttl_str))
                except Exception as e:
                    print("WARNING: Error parsing citation TTL: ", e)
    return [{"subject": s, "predicate": p, "object": o} for s, p, o in triples]


def _dedupe_triples(triples: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for t in triples:
        tup = (str(t.get("subject", "")), str(t.get("predicate", "")), str(t.get("object", "")))
        if tup in seen:
            continue
        seen.add(tup)
        out.append({"subject": tup[0], "predicate": tup[1], "object": tup[2]})
    return out


def _short(value: object, max_len: int = 140) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


async def execute_trace_ground_truth(trace: List[Dict], mcp_client: MultiServerMCPClient) -> List[Dict]:
    """
    Executes answer-bearing fact/query_builder calls in trace against MCP server
    and retrieves citation triples.
    """
    t_start = time.perf_counter()
    answer_bearing_tools = {"fact", "query_builder"}
    citation_keys: set[str] = set()
    executed_calls = 0
    selected_calls = [
        step
        for step in trace
        if step.get("tool") in answer_bearing_tools and bool(step.get("is_answer"))
    ]

    if not selected_calls:
        print("[ground_truth] no answer-bearing calls in trace", flush=True)
        return []

    print(f"[ground_truth] trace_len={len(trace)}", flush=True)
    for i, step in enumerate(trace):
        tool = step.get("tool")
        is_answer = step.get("is_answer")
        if tool not in answer_bearing_tools or not is_answer:
            continue
        summary = {
            "tool": tool,
            "is_answer": is_answer,
            "subject": _short(step.get("subject", "")),
            "predicate": _short(step.get("predicate", "")),
            "object": _short(step.get("object", "")),
            "type": _short(step.get("type", "")),
            "filters": _short(step.get("filters", "")),
            "project": _short(step.get("project", "")),
        }
        print(f"[ground_truth] trace_step[{i}] {summary}", flush=True)
    print("[ground_truth] opening MCP session...", flush=True)

    async with mcp_client.session("kg-mcp") as session:
        print("[ground_truth] MCP session opened", flush=True)
        for tool_call in selected_calls:
            if "is_answer" not in tool_call:
                raise ValueError(f"Trace step missing is_answer: {tool_call}")

            tool_name = tool_call.get("tool")
            args = {
                k: v
                for k, v in tool_call.items()
                if k not in {"tool", "required", "is_answer"}
            }
            if tool_name == "fact":
                # For expected-triple extraction, wildcard object avoids brittle literal/escaping mismatches.
                args["object"] = "_"
            print(
                f"[ground_truth] executing {tool_name} args={_short(args, 220)}",
                flush=True,
            )

            try:
                t_call = time.perf_counter()
                res = await session.call_tool(tool_name, args)
                executed_calls += 1
                print(
                    f"[ground_truth] call_tool {tool_name} took {time.perf_counter() - t_call:.3f}s",
                    flush=True,
                )
                content = "".join([c.text for c in res.content if hasattr(c, "text")])
                for citation_key in re.findall(r"Citation Key:\s*([\w-]+)", content):
                    if citation_key in citation_keys:
                        continue
                    citation_keys.add(citation_key)
                    t_cite = time.perf_counter()
                    await session.call_tool("cite", {"key": citation_key})
                    print(
                        f"[ground_truth] cite took {time.perf_counter() - t_cite:.3f}s",
                        flush=True,
                    )
            except Exception as e:
                print(f"Error executing ground truth tool call ({tool_name}): {e}")

        try:
            t_read = time.perf_counter()
            citation_result = await session.read_resource("citation://session")
            print(
                f"[ground_truth] read_resource(citation://session) took {time.perf_counter() - t_read:.3f}s",
                flush=True,
            )
            if not citation_result or not citation_result.contents:
                return []
            text = citation_result.contents[0].text
            citations_json = json.loads(text)
            all_triples = extract_citations(citations_json)
        except Exception as e:
            print(f"Error reading citation session: {e}")
            return []

    deduped = _dedupe_triples(all_triples)
    print(
        f"[ground_truth] total took {time.perf_counter() - t_start:.3f}s; "
        f"executed_calls={executed_calls}; citation_keys={len(citation_keys)}; "
        f"triples={len(deduped)}",
        flush=True,
    )
    return deduped
