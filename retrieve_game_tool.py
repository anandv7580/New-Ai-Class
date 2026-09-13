"""
Tool 1: retrieve_game

Performs semantic search on the local FAISS index and returns formatted
game details. This is the only tool the RAG agent has access to.
"""

from langchain_core.tools import tool

from src.logging_config import get_logger
from src.rag.vector_store import similarity_search

logger = get_logger(__name__)


def _format_match(doc, score: float) -> str:
    meta = doc.metadata
    return (
        f"Title: {meta.get('Name')}\n"
        f"Platform: {meta.get('Platform')}\n"
        f"Genre: {meta.get('Genre')}\n"
        f"Publisher: {meta.get('Publisher')}\n"
        f"Release Date: {meta.get('ReleaseYear')}\n"
        f"Description: {meta.get('Description')}\n"
        f"(similarity score: {score:.3f})"
    )


@tool
def retrieve_game(query: str) -> str:
    """Search the internal FAISS game database for entries relevant to the query.

    Use this FIRST for any question about a specific video game's platform,
    genre, publisher, release date, or description. Returns formatted game
    details for the top matches, or a clear "no relevant results" message.
    """
    logger.info("retrieve_game called with query=%r", query)
    try:
        results = similarity_search(query, k=3)
    except Exception as exc:  # noqa: BLE001
        logger.exception("retrieve_game failed")
        return f"[retrieve_game error] Could not query the FAISS index: {exc}"

    if not results:
        logger.info("retrieve_game found no matches for query=%r", query)
        return "No relevant results found in the internal game database."

    formatted = "\n\n---\n\n".join(_format_match(doc, score) for doc, score in results)
    logger.info("retrieve_game returning %d matches for query=%r", len(results), query)
    return formatted
