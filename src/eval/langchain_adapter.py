import json
import os
import traceback
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from src.eval.harness import AgentAdapter, AgentResult
from src.eval.prompts import get_agent_system_prompt
from src.eval.ground_truth import extract_citations

load_dotenv()

# LLM Configuration
LLM_API_KEY = os.getenv("EVAL_LLM_API_KEY", "ollama")
LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", "http://localhost:11434/v1")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp")

def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

# Determinism toggle (only switch we expose)
LLM_IS_DETERMINISTIC = _parse_bool(os.getenv("EVAL_DETERMINISTIC"), default=True)

# Hardcoded decoding defaults
LLM_TEMPERATURE_DET = 0.0
LLM_TEMPERATURE_NDET = 0.7
LLM_TOP_P = 1.0
LLM_MAX_TOKENS = None
LLM_N = 1
LLM_FREQUENCY_PENALTY = 0.0
LLM_PRESENCE_PENALTY = 0.0
LLM_TOP_K = None
class LangChainAdapter(AgentAdapter):
    """Adapter using LangChain ReAct/Tool-calling agent via LangGraph."""

    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("model_name is required")
        
        self.model_name = model_name
        
        model_kwargs = {}
        llm_kwargs = {}
        if LLM_IS_DETERMINISTIC:
            if LLM_TOP_K is not None:
                model_kwargs["top_k"] = LLM_TOP_K
            llm_kwargs = {
                "temperature": LLM_TEMPERATURE_DET,
                "top_p": LLM_TOP_P,
                "n": LLM_N,
                "max_tokens": LLM_MAX_TOKENS,
                "frequency_penalty": LLM_FREQUENCY_PENALTY,
                "presence_penalty": LLM_PRESENCE_PENALTY,
            }
        
        llm_kwargs["model_kwargs"] = model_kwargs

        self.llm = ChatOpenAI(
            model=self.model_name,
            seed=0,
	        api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            **llm_kwargs,
        )

        self.client = MultiServerMCPClient({
            "kg-mcp": {
                "transport": "http",
                "url": MCP_SERVER_URL,
            }
        })

        # Load system prompt
        self.system_prompt = get_agent_system_prompt()

    async def answer_question(self, question: str) -> AgentResult:
        """Answer question using LangChain agent with MCP tools and return raw citation data."""

        try:
            # Connect to MCP server via Streamable HTTP (SSE)
            async with self.client.session("kg-mcp") as session:
                # 1. Load tools from MCP session
                tools = await load_mcp_tools(session)
                if not tools:
                    return AgentResult(
                        answer="Error: No tools available from MCP server.",
                        citations_data=[],
                        explanation_data=[]
                    )

                # Enable error handling for all tools
                for tool in tools:
                    tool.handle_tool_error = True

                # 2. Create Agent using LangGraph with system prompt
                agent = create_agent(model=self.llm, tools=tools, system_prompt=self.system_prompt)

                # 3. Invoke Agent with a step limit to prevent infinite loops
                token_handler = TokenUsageCallback()
                trace_handler = LiveTraceCallback()
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={"recursion_limit": 50, "callbacks": [token_handler, trace_handler]},
                )

                # 4. Extract answer
                answer = result["messages"][-1].content
                token_usage = token_handler.get_usage()

                # 5. Get raw citation and explanation data from resources
                citations_data = []
                explanation_data = []
                try:
                    citation_result = await session.read_resource("citation://session")
                    if citation_result and citation_result.contents:
                        raw_citations = json.loads(citation_result.contents[0].text)
                        # Extract actual triples from the TTL data
                        citations_data = extract_citations(raw_citations)

                    explanation_result = await session.read_resource("explanation://session")
                    if explanation_result and explanation_result.contents:
                        explanation_data = json.loads(explanation_result.contents[0].text)

                except Exception as e:
                    print(f"WARNING: Could not fetch resources: {e}")

                return AgentResult(
                    answer=answer,
                    citations_data=citations_data,
                    explanation_data=explanation_data,
                    token_usage=token_usage
                )
                
        except Exception as e:
            # Ensure we always return an AgentResult, never None
            error_msg = f"Agent execution failed: {type(e).__name__}: {e}"
            print(f"ERROR: {error_msg}")
            # Surface the underlying cause (esp. ExceptionGroup from async TaskGroup).
            traceback.print_exc()
            if hasattr(e, "exceptions"):
                try:
                    for idx, sub in enumerate(getattr(e, "exceptions") or [], start=1):
                        print(f"ERROR: ExceptionGroup sub-exception {idx}: {type(sub).__name__}: {sub}")
                except Exception:
                    pass
            return AgentResult(
                answer=error_msg,
                citations_data=[],
                explanation_data=[],
                token_usage=[]
            )

class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            generations = getattr(response, "generations", []) or []
            for gen_list in generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    usage = getattr(msg, "usage_metadata", None)
                    print(f"[TokenUsageCallback] usage_metadata={usage}")
                    if not isinstance(usage, dict):
                        continue
                    self._entries.append(usage)
        except Exception:
            pass

    def get_usage(self) -> List[Dict[str, Any]]:
        return self._entries


class LiveTraceCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self._step = 0

    def _bump(self, label: str) -> None:
        self._step += 1
        print(f"[LiveTrace] {self._step:02d} {label}")

    def _format_tool_input_full(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload[:400] + "..." if len(payload) > 400 else payload
        try:
            if isinstance(payload, dict) or isinstance(payload, list):
                text = json.dumps(payload, ensure_ascii=True)
                return text[:400] + "..." if len(text) > 400 else text
            text = json.dumps(payload, ensure_ascii=True)
            return text[:400] + "..." if len(text) > 400 else text
        except Exception:
            text = str(payload)
            return text[:400] + "..." if len(text) > 400 else text

    def _summarize_output(self, output: Any) -> str:
        text = str(output)
        return text[:200] + "..." if len(text) > 200 else text

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._bump("LLM start")

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            generations = getattr(response, "generations", []) or []
            for gen_list in generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    content = getattr(msg, "content", None)
                    if isinstance(content, str) and content.strip():
                        text = content.strip()
                        if len(text) > 300:
                            text = text[:300] + "..."
                        self._bump(f"Reasoning: {text}")
        except Exception:
            self._bump("Reasoning step completed")

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = serialized.get("name") if isinstance(serialized, dict) else None
        self._bump(f"Tool call: {name or 'unknown'} | {self._format_tool_input_full(input_str)}")

    def on_tool_end(self, output, **kwargs) -> None:
        self._bump(f"Tool result: {self._summarize_output(output)}")

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        self._bump(f"Tool call: {action.tool} | {self._format_tool_input_full(action.tool_input)}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        self._bump("Agent finish")
