import json
import re
from typing import List, Dict, Set, Tuple
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
    """
    Extract triples from the citation resource data.
    """
    triples = set()
    for entry in citations_data:
        if isinstance(entry, dict) and 'ttl' in entry:
            ttl_str = entry['ttl']
            if ttl_str:
                try:
                    triples.update(extract_triples_from_subgraph(ttl_str))
                except Exception as e:
                    print("WARNING: Error parsing citation TTL: ", e)
    # Return as list of dicts for evaluation
    return [{"subject": s, "predicate": p, "object": o} for s, p, o in triples]



async def execute_trace_ground_truth(trace: List[Dict], mcp_client: MultiServerMCPClient) -> List[Dict]:
    """
    Executes all relevant tool calls in a trace (query_builder, fact) 
    against a fresh MCP server session to retrieve ground truth triples.
    """
    all_triples = []
    
    async with mcp_client.session("kg-mcp") as session:
        for tool_call in trace:
            if "is_answer" not in tool_call:
                raise ValueError(f"Trace step missing is_answer: {tool_call}")

            tool_name = tool_call.get("tool")
            if tool_name not in ["query_builder", "fact"]:
                continue
            if not tool_call["is_answer"]:
                continue
                
            # 1. Prepare arguments (remove "tool" key)
            args = {k: v for k, v in tool_call.items() if k not in {"tool", "required", "is_answer"}}
            
            # 2. Call the tool
            try:
                res = await session.call_tool(tool_name, args)
                
                # 3. Handle citation logic for explainable tools
                content = "".join([c.text for c in res.content if hasattr(c, "text")])
                match = re.search(r"Citation Key: ([a-f0-9\-]{36})", content)
                
                if match:
                    citation_key = match.group(1)
                    await session.call_tool("cite", {"key": citation_key})
                
                # 4. Read the citation resource
                citation_result = await session.read_resource("citation://session")
                # Resource contents is a list of resource parts, usually one part with text
                text = citation_result.contents[0].text
                citations_json = json.loads(text)
                output_triples = extract_citations(citations_json)
                all_triples.extend(output_triples)                

            except Exception as e:
                print(f"Error executing ground truth tool call ({tool_name}): {e}")

    return all_triples
