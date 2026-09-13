"""
Tool 4: synthesize_final_answer ("overall tool")

Takes the winning context (internal or web), the query, the source label,
and the retrieval evaluation, and produces one final, structured
FinalAnswer: a natural-language answer, highlighted Key Details
(Platform, Publisher, Release Date), and an explicit source citation.

This is the single place final formatting happens, so every answer the
orchestrator returns -- regardless of which path it took -- has the same
shape and always cites its source, per the project's output-format rules.
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_CHAT_MODEL
from src.logging_config import get_logger
from src.schemas import FinalAnswer, KeyDetails

logger = get_logger(__name__)

_SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are UdaPlay's answer synthesizer. Given a user query and supporting "
            "context, write a clear, concise natural-language answer. Then extract, if "
            "known from the context, the game's Platform, Publisher, and Release Date "
            "into key_details. If a detail isn't present in the context, leave it null "
            "rather than guessing.",
        ),
        (
            "human",
            "User query:\n{query}\n\nSupporting context:\n{context}",
        ),
    ]
)


class _SynthOutput(FinalAnswer):
    # Reuse FinalAnswer's shape for structured-output binding, minus fields
    # (source, confidence) that the orchestrator fills in itself rather than
    # trusting the LLM to report.
    pass


def _get_synth_llm():
    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.2,
    )
    return llm.with_structured_output(_SynthOutput)


@tool
def synthesize_final_answer(query: str, context: str, source: str, confidence: Optional[float] = None) -> dict:
    """Combine retrieved context (internal and/or web) into one structured final
    answer with highlighted Key Details and an explicit Source citation.

    `source` must be one of: "Internal Game Database", "Web Search via Tavily",
    or "Internal Game Database + Web Search via Tavily".
    """
    logger.info("synthesize_final_answer called | source=%s | query=%r", source, query)
    structured_llm = _get_synth_llm()
    chain = _SYNTH_PROMPT | structured_llm

    try:
        draft: _SynthOutput = chain.invoke({"query": query, "context": context})
        key_details = draft.key_details
        answer_text = draft.answer
    except Exception as exc:  # noqa: BLE001
        logger.exception("synthesize_final_answer LLM formatting failed; using raw context as fallback")
        answer_text = context
        key_details = KeyDetails()

    final = FinalAnswer(
        query=query,
        answer=answer_text,
        key_details=key_details,
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
    )
    logger.info("synthesize_final_answer produced final answer for query=%r", query)
    return final.model_dump()
