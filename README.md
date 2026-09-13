# UdaPlay — AI Gaming Research Agent

UdaPlay is a multi-agent research assistant for a gaming analytics platform.
It answers questions from executives, analysts, and gamers by first checking
an internal FAISS knowledge base of video games, and automatically falling
back to a live web search (Tavily) whenever the internal data is missing or
low-confidence.

## Architecture

Four specialized agents, coordinated by an Orchestrator, each wrapping one
focused LangChain tool:

```
                         ┌─────────────────────┐
                         │   Orchestrator       │
                         │   Agent (state       │
                         │   machine + memory)  │
                         └──────────┬───────────┘
                                    │
                 1. query           │
                                    ▼
                         ┌─────────────────────┐
                         │   RAG Agent          │
                         │   tool: retrieve_game│──▶ FAISS vector DB
                         └──────────┬───────────┘
                                    │ internal context
                                    ▼
                         ┌─────────────────────┐
                         │  Validation Agent    │
                         │  tool: evaluate_     │
                         │  retrieval           │
                         └──────────┬───────────┘
                                    │ {confidence, is_sufficient, reasoning}
                          ┌─────────┴─────────┐
                     sufficient           insufficient
                          │                   │
                          │                   ▼
                          │        ┌─────────────────────┐
                          │        │  Web Search Agent    │
                          │        │  tool: game_web_search│──▶ Tavily API
                          │        └──────────┬───────────┘
                          │                   │ web context
                          └─────────┬─────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  "Overall" Tool:      │
                         │  synthesize_final_    │
                         │  answer (structured,  │
                         │  cited FinalAnswer)   │
                         └─────────────────────┘
```

- **RAG Agent** — the only tool it can call is `retrieve_game`, which runs
  semantic similarity search against the local FAISS index.
- **Validation Agent** — the only tool it can call is `evaluate_retrieval`,
  which uses an LLM with a **structured (Pydantic) output** to score whether
  the retrieved content actually answers the query.
- **Web Search Agent** — the only tool it can call is `game_web_search`,
  a Tavily-backed fallback used only when validation fails.
- **Orchestrator Agent** — a plain Python state machine (not itself an LLM
  agent) that runs the steps above in order, keeps short-term conversation
  memory across turns, logs every decision, and calls the **overall tool**
  (`synthesize_final_answer`) to produce one consistently-formatted,
  cited answer regardless of which path was taken.

Every tool call, evaluation score, and fallback decision is logged to both
the console and `logs/udaplay.log` (see `src/logging_config.py`).

## Project Structure

```
UdaPlay/
├── config.env.example      # copy to config.env (or .env) and fill in keys
├── requirements.txt
├── data/
│   └── games.json          # sample internal game dataset (24 titles)
├── src/
│   ├── config.py            # env loading / constants
│   ├── logging_config.py    # shared logger setup
│   ├── schemas.py            # Pydantic v2 models (GameRecord, RetrievalEvaluation, FinalAnswer, ...)
│   ├── rag/
│   │   ├── data_loader.py   # JSON -> validated GameRecord -> Document
│   │   └── vector_store.py  # FAISS build / persist / reload / similarity_search
│   ├── tools/
│   │   ├── retrieve_game_tool.py       # Tool 1
│   │   ├── evaluate_retrieval_tool.py  # Tool 2
│   │   ├── web_search_tool.py          # Tool 3 (Tavily)
│   │   └── overall_tool.py             # Tool 4 ("overall" synthesizer)
│   ├── agents/
│   │   ├── rag_agent.py
│   │   ├── validation_agent.py
│   │   ├── web_search_agent.py
│   │   └── orchestrator_agent.py
│   └── main.py               # CLI entry point
├── scripts/
│   └── build_index.py        # one-off / repeatable FAISS index builder
├── notebooks/
│   ├── Udaplay_01_solution_project.ipynb   # Part 1: RAG pipeline
│   └── Udaplay_02_solution_project.ipynb   # Part 2: multi-agent workflow
└── logs/
    └── udaplay.log            # created at runtime
```

## Setup (Vocareum / local)

1. **Unzip** this project and open it in your workspace (Vocareum or local Jupyter/VS Code).

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**: copy `config.env.example` to `config.env`
   (the code also accepts a plain `.env` file) and fill in real values:
   ```
   OPENAI_API_KEY="sk-..."
   TAVILY_API_KEY="tvly-..."
   OPENAI_BASE_URL="https://openai.vocareum.com/v1"
   ```
   `config.env` / `.env` are already git-ignored — never commit real keys.

4. **Build the FAISS index** (embeds `data/games.json`):
   ```bash
   python scripts/build_index.py
   ```
   This creates a persisted index folder (`faiss_index_udaplay/`) that
   `FAISS.load_local(...)` reloads on future runs without re-embedding.

5. **Run the agent**:
   ```bash
   # interactive multi-turn session
   python -m src.main

   # or a single one-shot query
   python -m src.main --query "When was Pokémon Red launched, and on what platform?"
   ```

6. **Or work through the notebooks** in order:
   - `notebooks/Udaplay_01_solution_project.ipynb` — builds and verifies the FAISS index.
   - `notebooks/Udaplay_02_solution_project.ipynb` — runs the full multi-agent workflow against the three sample test scenarios below, showing reasoning traces and cited answers.

## Sample Test Scenarios

| # | Query | Expected Path |
|---|-------|----------------|
| 1 | "When was Pokémon Red launched, and on what platform?" | FAISS hit → Validation passes → **Internal Game Database** citation |
| 2 | "What is Rockstar Games working on right now?" | Not in local data → Validation fails → **Web Search via Tavily** fallback |
| 3 | "When was God of War Ragnarok released?" | Not in the sample dataset (only the 2018 *God of War* is indexed) → Validation fails → Web fallback completes the answer |

## Notes on the Sample Dataset

`data/games.json` intentionally does **not** include *God of War Ragnarok*
or any "what is [studio] currently working on" information, so scenarios 2
and 3 genuinely exercise the web-search fallback path rather than being
answerable from local data alone. Swap in your own dataset at any time —
just re-run `scripts/build_index.py` afterward.

## Bonus Enhancements Included / Possible Next Steps

- ✅ **Structured outputs everywhere**: `RetrievalEvaluation` and
  `FinalAnswer` are validated Pydantic v2 models end-to-end.
- ✅ **Full step-by-step logging**: every tool call, evaluation score, and
  fallback decision is logged (console + `logs/udaplay.log`).
- ⬜ **Long-term memory** (suggested next step): after a successful web
  search, embed the new fact and `vector_store.add_documents(...)` it back
  into the FAISS index so the internal database grows over time.
- ⬜ **Rich metadata filters** (suggested next step): FAISS supports a
  `filter=` argument on `similarity_search`; you can extend
  `retrieve_game_tool.py` to accept optional `publisher` / `platform`
  arguments and pass them through as a metadata filter.

## Troubleshooting

- `EnvironmentError: Missing required environment variable(s)` — make sure
  `config.env` (or `.env`) exists in the project root and both
  `OPENAI_API_KEY` and `TAVILY_API_KEY` are set.
- `ImportError` for `langchain_tavily` — the web search tool automatically
  falls back to the older `langchain_community` Tavily integration if
  `langchain-tavily` isn't installed; either works.
- Re-running `scripts/build_index.py` always rebuilds and overwrites
  `faiss_index_udaplay/` from the current `data/games.json`.
