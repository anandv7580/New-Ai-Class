"""
Validation Agent

A LangChain tool-calling agent whose only capability is evaluate_retrieval.
It decides, given the query and the RAG Agent's raw output, whether that
internal context is trustworthy enough to answer from directly.
"""

import json

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_CHAT_MODEL
from src.logging_config import get_logger
from src.schemas import RetrievalEvaluation
from src.tools.evaluate_retrieval_tool import evaluate_retrieval

logger = get_logger(__name__)

_VALIDATION_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Validation Agent for UdaPlay. You have exactly one "
            "tool, evaluate_retrieval. You will be given a user query and the "
            "content retrieved by the RAG Agent, formatted as:\n"
            "QUERY: <query>\nRETRIEVED_DOCS: <content>\n\n"
            "Always call evaluate_retrieval with those two values, then return "
            "the tool's JSON result verbatim as your final answer. Do not add "
            "commentary.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def build_validation_agent() -> AgentExecutor:
    """Construct the Validation Agent's AgentExecutor (single tool: evaluate_retrieval)."""
    logger.info("Building Validation Agent (tool: evaluate_retrieval)")
    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    tools = [evaluate_retrieval]
    agent = create_tool_calling_agent(llm, tools, _VALIDATION_AGENT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def run_validation_agent(executor: AgentExecutor, query: str, retrieved_docs: str) -> RetrievalEvaluation:
    logger.info("Validation Agent evaluating retrieval for query=%r", query)
    agent_input = f"QUERY: {query}\nRETRIEVED_DOCS: {retrieved_docs}"

    try:
        result = executor.invoke({"input": agent_input})
        raw_output = result.get("output", "")
        # The agent is instructed to echo the tool's dict/JSON verbatim.
        if isinstance(raw_output, dict):
            data = raw_output
        else:
            data = json.loads(raw_output)
        evaluation = RetrievalEvaluation(**data)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Validation Agent output could not be parsed (%s); calling evaluate_retrieval directly as a safety net.",
            exc,
        )
        evaluation = RetrievalEvaluation(**evaluate_retrieval.invoke({"query": query, "retrieved_docs": retrieved_docs}))

    logger.info(
        "Validation Agent decision: is_sufficient=%s confidence=%.2f",
        evaluation.is_sufficient,
        evaluation.confidence_score,
    )
    return evaluation
