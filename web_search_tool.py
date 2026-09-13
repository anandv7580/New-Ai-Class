"""
Tool 3: game_web_search

Falls back to the Tavily Search API when the internal FAISS database
lacks relevant or sufficient information (e.g. recent releases or
"what is X working on right now" style questions).

Prefers the modern `langchain-tavily` package; falls back to the older
(deprecated but still functional) `langchain_community` integration if
that package is unavailable, so this keeps working across course
environments pinned to slightly different LangChain versions.
"""

from langchain_core.tools import tool

from src.config import TAVILY_API_KEY
from src.logging_config import get_logger

logger = get_logger(__name__)

_tavily_client_singleton = None


def _get_tavily_client():
    global _tavily_client_singleton
    if _tavily_client_singleton is not None:
        return _tavily_client_singleton

    try:
        from langchain_tavily import TavilySearch

        _tavily_client_singleton = TavilySearch(max_results=5, topic="general", api_key=TAVILY_API_KEY)
        logger.info("Using langchain_tavily.TavilySearch client")
    except ImportError:
        from langchain_community.tools.tavily_search import TavilySearchResults

        _tavily_client_singleton = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)
        logger.info("langchain_tavily not installed; falling back to langchain_community.TavilySearchResults")

    return _tavily_client_singleton


def _format_results(raw) -> str:
    # langchain_tavily returns {"results": [...], "answer": ...}
    # langchain_community's TavilySearchResults returns a list of dicts directly.
    if isinstance(raw, dict):
        items = raw.get("results", [])
        answer = raw.get("answer")
    else:
        items = raw
        answer = None

    blocks = []
    if answer:
        blocks.append(f"Quick answer: {answer}")

    for item in items:
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")
        blocks.append(f"[{title}]({url})\n{content}")

    return "\n\n---\n\n".join(blocks) if blocks else "No web results found."


@tool
def game_web_search(query: str) -> str:
    """Search the live web (via Tavily) for gaming news, upcoming releases,
    or any information not found in the internal database.

    Use this ONLY after retrieve_game + evaluate_retrieval indicate the
    internal database result is missing or insufficient.
    """
    logger.info("game_web_search called with query=%r", query)
    client = _get_tavily_client()

    try:
        raw = client.invoke({"query": query})
    except Exception as exc:  # noqa: BLE001
        logger.exception("game_web_search failed")
        return f"[game_web_search error] Tavily search failed: {exc}"

    formatted = _format_results(raw)
    logger.info("game_web_search returning formatted web results for query=%r", query)
    return formatted
