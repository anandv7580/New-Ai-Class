"""
FAISS vector store construction, persistence, and reload utilities,
plus a small similarity-search wrapper used both by notebook 01 for
verification and by the retrieve_game tool in notebook 02 / the agent.
"""

import os
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import FAISS_INDEX_PATH, GAMES_DATA_PATH, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL
from src.logging_config import get_logger
from src.rag.data_loader import load_game_records, records_to_documents

logger = get_logger(__name__)

_embeddings_singleton: Optional[OpenAIEmbeddings] = None
_vector_store_singleton: Optional[FAISS] = None


def get_embeddings() -> OpenAIEmbeddings:
    """Lazily instantiate a shared OpenAIEmbeddings client."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        logger.info("Instantiating OpenAIEmbeddings (model=%s)", OPENAI_EMBEDDING_MODEL)
        _embeddings_singleton = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    return _embeddings_singleton


def build_vector_store(
    data_path: str = GAMES_DATA_PATH,
    index_path: str = FAISS_INDEX_PATH,
    persist: bool = True,
) -> FAISS:
    """Build a fresh FAISS index from data/games.json and (optionally) persist it."""
    records = load_game_records(data_path)
    documents = records_to_documents(records)

    logger.info("Embedding %d documents and building FAISS index ...", len(documents))
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    logger.info("FAISS index built with %d vectors", len(documents))

    if persist:
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        logger.info("Persisted FAISS index to disk at '%s'", index_path)

    global _vector_store_singleton
    _vector_store_singleton = vector_store
    return vector_store


def load_vector_store(index_path: str = FAISS_INDEX_PATH) -> FAISS:
    """Reload a previously persisted FAISS index from disk."""
    global _vector_store_singleton
    if _vector_store_singleton is not None:
        return _vector_store_singleton

    if not os.path.isdir(index_path):
        logger.warning("No FAISS index found at '%s'; building a new one from raw data.", index_path)
        return build_vector_store(index_path=index_path)

    logger.info("Loading persisted FAISS index from '%s'", index_path)
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we generated this index ourselves
    )
    _vector_store_singleton = vector_store
    return vector_store


def similarity_search(query: str, k: int = 3) -> List[Tuple[Document, float]]:
    """Run a top-k similarity search with relevance scores, for verification and retrieval."""
    vector_store = load_vector_store()
    logger.info("Running similarity search for query=%r (k=%d)", query, k)
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    for doc, score in results:
        logger.debug("  match=%r score=%.4f", doc.metadata.get("Name"), score)
    return results
