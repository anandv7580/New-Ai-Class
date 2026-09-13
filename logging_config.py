"""
Centralized logging setup for UdaPlay.

Every module (tools, agents, orchestrator, scripts) calls
`get_logger(__name__)` to get a logger that writes to both the console
and a shared rotating log file, so a full run of the multi-agent
pipeline produces one readable trace of every step: retrieval calls,
evaluation scores, fallback triggers, and final answers.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import LOG_DIR, LOG_LEVEL

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "udaplay.log"

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger("udaplay")
    root.setLevel(LOG_LEVEL)
    root.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (e.g. 'udaplay.tools.retrieve_game')."""
    _configure_root()
    short_name = name.replace("src.", "").replace("__main__", "main")
    return logging.getLogger(f"udaplay.{short_name}")
