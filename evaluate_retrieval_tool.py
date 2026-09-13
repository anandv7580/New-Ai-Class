"""
Tool 2: evaluate_retrieval

Evaluates whether the content returned by retrieve_game is relevant and
sufficient to answer the user's query. Internally this uses an LLM bound
to a Pydantic schema (structured output) so the result is always a
validated {confidence_score, is_sufficient, reasoning} payload.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_CHAT_MODEL
from src.logging_config import get_logger
from src.schemas import RetrievalEvaluation

logger = get_logger(__name__)

_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval-quality evaluator for a video game research "
            "assistant. Given a user query and the content retrieved from an internal "
            "vector database, decide whether that content is relevant AND sufficient "
            "to fully answer the query on its own, with no outside information. "
            "Be conservative: if the retrieved content does not mention the exact "
            "game/entity/fact the query asks about, or the query asks about something "
            "current/recent/future ('right now', 'upcoming', 'latest'), treat it as "
            "NOT sufficient.",
        ),
        (
            "human",
            "User query:\n{query}\n\nRetrieved content:\n{retrieved_docs}\n\n"
            "Evaluate this retrieval.",
        ),
    ]
)


def _get_evaluator_llm():
    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    return llm.with_structured_output(RetrievalEvaluation)


@tool
def evaluate_retrieval(query: str, retrieved_docs: str) -> dict:
    """Evaluate whether retrieved_docs (from retrieve_game) are relevant and
    sufficient to answer the query.

    Returns a dict with keys: confidence_score (0.0-1.0), is_sufficient (bool),
    and reasoning (str). Always call this after retrieve_game, before deciding
    whether a web search fallback is needed.
    """
    logger.info("evaluate_retrieval called for query=%r", query)
    structured_llm = _get_evaluator_llm()
    chain = _EVAL_PROMPT | structured_llm

    try:
        evaluation: RetrievalEvaluation = chain.invoke({"query": query, "retrieved_docs": retrieved_docs})
    except Exception as exc:  # noqa: BLE001
        logger.exception("evaluate_retrieval failed; defaulting to insufficient")
        evaluation = RetrievalEvaluation(
            confidence_score=0.0,
            is_sufficient=False,
            reasoning=f"Evaluation call failed ({exc}); defaulting to insufficient to trigger web fallback.",
        )

    logger.info(
        "evaluate_retrieval result: is_sufficient=%s confidence=%.2f",
        evaluation.is_sufficient,
        evaluation.confidence_score,
    )
    return evaluation.model_dump()
