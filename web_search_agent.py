"""
Web Search Agent

A LangChain tool-calling agent whose only capability is game_web_search
(Tavily). Invoked by the orchestrator only when the Validation Agent
decides the internal FAISS result is insufficient.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_CHAT_MODEL
from src.logging_config import get_logger
from src.tools.web_search_tool import game_web_search

logger = get_logger(__name__)

_WEB_SEARCH_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Web Search Agent for UdaPlay. You have exactly one "
            "tool, game_web_search, backed by the Tavily API. Always call it "
            "with the user's query. Return the tool's raw output verbatim as "
            "your final answer -- do not summarize and do not add outside "
            "knowledge beyond what the tool returns.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def build_web_search_agent() -> AgentExecutor:
    """Construct the Web Search Agent's AgentExecutor (single tool: game_web_search)."""
    logger.info("Building Web Search Agent (tool: game_web_search)")
    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    tools = [game_web_search]
    agent = create_tool_calling_agent(llm, tools, _WEB_SEARCH_AGENT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def run_web_search_agent(executor: AgentExecutor, query: str) -> str:
    logger.info("Web Search Agent received query=%r", query)
    result = executor.invoke({"input": query})
    output = result.get("output", "")
    logger.info("Web Search Agent finished for query=%r", query)
    return output
