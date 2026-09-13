"""
RAG Agent

A LangChain tool-calling agent whose only capability is retrieve_game.
Its job is narrow on purpose: given a query, decide how to phrase the
FAISS lookup and return the raw internal context. It never fabricates
game facts and never talks to the web.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_CHAT_MODEL
from src.logging_config import get_logger
from src.tools.retrieve_game_tool import retrieve_game

logger = get_logger(__name__)

_RAG_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the RAG Agent for UdaPlay. You have exactly one tool, "
            "retrieve_game, which searches an internal FAISS database of video "
            "games. Always call retrieve_game with the user's query (rephrased "
            "to focus on the game title/entity if helpful). Return the tool's "
            "raw output verbatim as your final answer -- do not summarize, do "
            "not add outside knowledge, and do not say anything else.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def build_rag_agent() -> AgentExecutor:
    """Construct the RAG Agent's AgentExecutor (single tool: retrieve_game)."""
    logger.info("Building RAG Agent (tool: retrieve_game)")
    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    tools = [retrieve_game]
    agent = create_tool_calling_agent(llm, tools, _RAG_AGENT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def run_rag_agent(executor: AgentExecutor, query: str) -> str:
    logger.info("RAG Agent received query=%r", query)
    result = executor.invoke({"input": query})
    output = result.get("output", "")
    logger.info("RAG Agent finished for query=%r", query)
    return output
