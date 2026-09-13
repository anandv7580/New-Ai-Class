"""
Orchestrator Agent

The top-level controller implementing UdaPlay's two-tier decision
workflow:

    User Query -> RAG Agent (retrieve_game)
               -> Validation Agent (evaluate_retrieval)
               -> [sufficient?]
                     yes -> synthesize_final_answer (Internal Game Database)
                     no  -> Web Search Agent (game_web_search)
                            -> synthesize_final_answer (Web Search via Tavily)

It also maintains short-term conversational memory across turns
(a running transcript of query/answer pairs) so multi-turn sessions can
refer back to earlier games without repeating themselves, and it logs
every decision point so a full session produces a readable trace.
"""

from dataclasses import dataclass, field
from typing import List

from src.config import CONFIDENCE_THRESHOLD
from src.logging_config import get_logger
from src.schemas import FinalAnswer
from src.agents.rag_agent import build_rag_agent, run_rag_agent
from src.agents.validation_agent import build_validation_agent, run_validation_agent
from src.agents.web_search_agent import build_web_search_agent, run_web_search_agent
from src.tools.overall_tool import synthesize_final_answer

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    query: str
    answer: FinalAnswer


@dataclass
class OrchestratorState:
    """Short-term conversational memory for a single session."""

    history: List[ConversationTurn] = field(default_factory=list)

    def add(self, query: str, answer: FinalAnswer) -> None:
        self.history.append(ConversationTurn(query=query, answer=answer))

    def as_context_string(self, max_turns: int = 5) -> str:
        if not self.history:
            return ""
        recent = self.history[-max_turns:]
        lines = ["Recent conversation history:"]
        for turn in recent:
            lines.append(f"- Q: {turn.query}\n  A: {turn.answer.answer}")
        return "\n".join(lines)


class OrchestratorAgent:
    """Coordinates the RAG, Validation, and Web Search agents for one session."""

    def __init__(self) -> None:
        logger.info("Initializing Orchestrator Agent and its sub-agents")
        self.rag_executor = build_rag_agent()
        self.validation_executor = build_validation_agent()
        self.web_search_executor = build_web_search_agent()
        self.state = OrchestratorState()

    def ask(self, query: str) -> FinalAnswer:
        logger.info("=" * 60)
        logger.info("Orchestrator received new query: %r", query)

        # Step 1: internal retrieval
        internal_context = run_rag_agent(self.rag_executor, query)
        logger.info("Step 1/4 complete: RAG Agent returned internal context")

        # Step 2: evaluate that retrieval
        evaluation = run_validation_agent(self.validation_executor, query, internal_context)
        logger.info(
            "Step 2/4 complete: Validation Agent -> is_sufficient=%s confidence=%.2f (%s)",
            evaluation.is_sufficient,
            evaluation.confidence_score,
            evaluation.reasoning,
        )

        # Step 3: branch on sufficiency
        use_internal = evaluation.is_sufficient and evaluation.confidence_score >= CONFIDENCE_THRESHOLD

        if use_internal:
            logger.info("Step 3/4: internal context judged SUFFICIENT -> skipping web fallback")
            context = internal_context
            source = "Internal Game Database"
        else:
            logger.info("Step 3/4: internal context judged INSUFFICIENT -> triggering Web Search Agent")
            web_context = run_web_search_agent(self.web_search_executor, query)
            if internal_context and "No relevant results" not in internal_context:
                context = f"Internal DB context:\n{internal_context}\n\nWeb search context:\n{web_context}"
                source = "Internal Game Database + Web Search via Tavily"
            else:
                context = web_context
                source = "Web Search via Tavily"

        # Step 4: synthesize the final structured, cited answer
        result_dict = synthesize_final_answer.invoke(
            {
                "query": query,
                "context": context,
                "source": source,
                "confidence": evaluation.confidence_score,
            }
        )
        final_answer = FinalAnswer(**result_dict)
        logger.info("Step 4/4 complete: final answer synthesized | source=%s", source)

        self.state.add(query, final_answer)
        return final_answer
