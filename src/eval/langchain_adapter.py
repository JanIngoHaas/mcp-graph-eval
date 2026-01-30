import json
import os
import traceback
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from src.eval.harness import AgentAdapter, AgentResult
from src.eval.prompts import get_agent_system_prompt
from src.eval.ground_truth import extract_citations

load_dotenv()

# LLM Configuration
LLM_API_KEY = os.getenv("EVAL_LLM_API_KEY", "ollama")
LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", "http://localhost:11434/v1")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp")

# Deterministic decoding (hardcoded)
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 1.0
LLM_MAX_TOKENS = None
LLM_N = 1
LLM_SEED = None
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
        if LLM_SEED is not None:
            model_kwargs["seed"] = LLM_SEED
        if LLM_TOP_K is not None:
            model_kwargs["top_k"] = LLM_TOP_K

        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            n=LLM_N,
            max_tokens=LLM_MAX_TOKENS,
            frequency_penalty=LLM_FREQUENCY_PENALTY,
            presence_penalty=LLM_PRESENCE_PENALTY,
            model_kwargs=model_kwargs
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
                agent = create_react_agent(self.llm, tools=tools, prompt=self.system_prompt)

                # 3. Invoke Agent with a step limit to prevent infinite loops
                token_handler = TokenUsageCallback()
                inputs = {"messages": [HumanMessage(content=question)]}
                result = await agent.ainvoke(
                    inputs,
                    config={"recursion_limit": 50, "callbacks": [token_handler]},
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
