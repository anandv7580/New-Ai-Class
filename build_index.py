"""
Standalone script: build (or rebuild) the persisted FAISS index from
data/games.json. Run this once before using the agent, and again
whenever data/games.json changes.

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

# Allow running as `python scripts/build_index.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import verify_environment, GAMES_DATA_PATH, FAISS_INDEX_PATH  # noqa: E402
from src.logging_config import get_logger  # noqa: E402
from src.rag.vector_store import build_vector_store, similarity_search  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    verify_environment()
    logger.info("Building FAISS index from %s", GAMES_DATA_PATH)
    build_vector_store(data_path=GAMES_DATA_PATH, index_path=FAISS_INDEX_PATH, persist=True)

    logger.info("Running a verification similarity search")
    sample_query = "Pokémon Red launch platform"
    results = similarity_search(sample_query, k=3)
    print(f"\nTop matches for verification query: {sample_query!r}\n")
    for doc, score in results:
        print(f"- {doc.metadata.get('Name')} (score={score:.3f})")


if __name__ == "__main__":
    main()
