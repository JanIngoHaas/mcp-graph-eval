from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AgentResult:
    answer: str
    citations_data: List[Dict[str, Any]] = field(default_factory=list)
    explanation_data: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: List[Dict[str, Any]] = field(default_factory=list)
    runtime_trace: List[Dict[str, Any]] = field(default_factory=list)

class AgentAdapter:
    async def answer_question(self, question: str) -> AgentResult:
        raise NotImplementedError
