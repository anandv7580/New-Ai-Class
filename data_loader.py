"""
Load and validate the raw video game JSON data, converting each record
into a LangChain Document ready for embedding.
"""

import json
from typing import List

from langchain_core.documents import Document

from src.logging_config import get_logger
from src.schemas import GameRecord

logger = get_logger(__name__)


def load_game_records(path: str) -> List[GameRecord]:
    """Load and validate raw JSON into a list of GameRecord models."""
    logger.info("Loading raw game data from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    records: List[GameRecord] = []
    for i, item in enumerate(raw_items):
        try:
            records.append(GameRecord(**item))
        except Exception as exc:  # noqa: BLE001 - we want to log & skip bad rows
            logger.warning("Skipping malformed record at index %d: %s", i, exc)

    logger.info("Loaded and validated %d/%d game records", len(records), len(raw_items))
    return records


def records_to_documents(records: List[GameRecord]) -> List[Document]:
    """Convert validated GameRecords into embedding-ready Documents.

    The page_content is a natural-language blob (used for semantic
    similarity search); the structured fields are preserved in metadata
    so the retrieve_game tool can render them cleanly.
    """
    documents: List[Document] = []
    for record in records:
        page_content = (
            f"{record.Name} is a {record.Genre} game for {record.Platform}, "
            f"published by {record.Publisher} in {record.ReleaseYear}. "
            f"{record.Description}"
        )
        metadata = record.model_dump()
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info("Converted %d game records into Document objects", len(documents))
    return documents
