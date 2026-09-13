"""
Central configuration for UdaPlay.

Loads environment variables from config.env (falls back to .env) and
exposes them as module-level constants so every other module imports
its settings from one place instead of re-reading the environment.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels up from this file (src/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Try config.env first (matches the original project spec), then .env.
_ENV_CANDIDATES = [PROJECT_ROOT / "config.env", PROJECT_ROOT / ".env"]
for _env_path in _ENV_CANDIDATES:
    if _env_path.exists():
        load_dotenv(_env_path)
        break
else:
    # No env file found on disk; still call load_dotenv() in case the
    # variables are already exported in the shell / Vocareum workspace.
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index_udaplay")
FAISS_INDEX_PATH = str(PROJECT_ROOT / FAISS_INDEX_DIR)

GAMES_DATA_PATH = str(PROJECT_ROOT / "data" / "games.json")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = PROJECT_ROOT / "logs"

# Confidence threshold used by the orchestrator if the validation agent's
# structured output ever needs a numeric fallback comparison.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))


def verify_environment() -> None:
    """Raise a clear error early if required secrets are missing."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Create a config.env (or .env) file from config.env.example and fill it in."
        )
