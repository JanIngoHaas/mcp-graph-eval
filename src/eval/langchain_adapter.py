import json
import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from src.eval.harness import AgentAdapter, AgentResult
from src.eval.prompts import get_agent_system_prompt
from src.eval.ground_truth import extract_citations

load_dotenv()

# LLM Configuration
LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "ministral-3:14b")
LLM_API_KEY = os.getenv("EVAL_LLM_API_KEY", "dummy") # Default to dummy for local models
LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", "http://localhost:11434/v1")
LLM_TEMPERATURE = float(os.getenv("EVAL_LLM_TEMPERATURE", "0.0"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp")

class LangChainAdapter(AgentAdapter):
    """Adapter using LangChain ReAct/Tool-calling agent via LangGraph."""

    def __init__(self):
        if not LLM_API_KEY:
            raise ValueError(
                "LLM API Key is missing. Please set EVAL_LLM_API_KEY or LLM_API_KEY in your .env file."
            )

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
            seed=54837347
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
            inputs = {"messages": [HumanMessage(content=question)]}
            result = await agent.ainvoke(inputs, config={"recursion_limit": 50})

            # 4. Extract answer
            answer = result["messages"][-1].content

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
                print("WARNING: Could not fetch resources: ", e)
                pass

            return AgentResult(
                answer=answer,
                citations_data=citations_data,
                explanation_data=explanation_data
            )