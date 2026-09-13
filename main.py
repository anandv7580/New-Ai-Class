"""
UdaPlay CLI entry point.

Usage:
    python -m src.main
    python -m src.main --query "When was Pokémon Red launched, and on what platform?"

Run scripts/build_index.py once beforehand to create the persisted FAISS
index (or let the first query build it automatically).
"""

import argparse

from src.config import verify_environment
from src.logging_config import get_logger
from src.agents.orchestrator_agent import OrchestratorAgent

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="UdaPlay AI Research Agent")
    parser.add_argument("--query", type=str, default=None, help="Run a single query and exit.")
    args = parser.parse_args()

    verify_environment()
    orchestrator = OrchestratorAgent()

    if args.query:
        answer = orchestrator.ask(args.query)
        print(answer.to_display_string())
        return

    print("UdaPlay AI Research Agent -- type a question, or 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer = orchestrator.ask(query)
        print("\nUdaPlay:")
        print(answer.to_display_string())
        print()


if __name__ == "__main__":
    main()
