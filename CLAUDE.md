# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A CLI tool for querying LLMs via multiple OpenAI-compatible API routers (Google AI, OpenRouter, and easily extensible). It sends questions to models listed in per-router CSV catalogs, streams responses with rich markdown rendering, and tracks token usage/cost.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Unix/macOS
pip install -r requirements.txt
```

Requires an API key for the active router:
- **Google AI**: `GOOGLE_API_KEY` env var or `.GOOGLE_API_KEY` file
- **OpenRouter**: `OPENROUTER_API_KEY` env var or `.OPENROUTER_API_KEY` file

## Running

```bash
python3 chat.py "your question here"
python3 chat.py --question "your question" --model "deepseek-ai/DeepSeek-V3.2"
python3 chat.py --list-models          # show available models
python3 chat.py --switch-model         # interactively change default model
python3 chat.py --switch-router        # interactively change API router
python3 chat.py "question" --choose-model # interactively pick a model for this run
python3 chat.py "question" --context   # continue conversation from last message
python3 chat.py "question" -c          # short form for continuing conversation
python3 chat.py "question"             # start fresh (overwrites previous context)
```

There are no tests, linter, or build system.

## Architecture

Single source file (`chat.py`) with no package structure:

- **`chat.py`** — Entry point and all core logic. Contains key structures:
  - `RouterConfig` — Dataclass defining a router (base URL, API key env/file, CSV path, defaults)
  - `ROUTERS` — Registry dict mapping keys ("google", "openrouter") to `RouterConfig` instances
  - `ModelRegistry` — Loads the active router's `models.csv`, finds cheapest provider for a given model name
  - `ContextManager` — Persists multi-turn chat history as JSON files in system temp dir
  - `LLMClient` — OpenAI SDK wrapper parameterized with base URL and API key
  - `App` — Orchestrates everything: resolves router/model, calls API, streams/renders response with `rich.Live`, displays token cost

## Key Details

- Active router and per-router default models stored in `.llm_config.json`
- Router selection: `--switch-router` interactive command, persisted in config
- Model resolution priority: `--model` arg > `.llm_config.json[router].default_model` > `RouterConfig.default_model`
- Each router has its own `routers/<key>/models.csv` with provider/pricing/context metadata
- API model IDs use the model name directly
- All console output uses `rich` library with color markup (`[cyan]`, `[red]`, etc.)
- Context behavior: every chat message is saved to a temp file; `--context` flag continues from last message, without it starts fresh
- To add a new router: add entry to `ROUTERS` dict, create `routers/<key>/models.csv`, add API key file to `.gitignore`

## Code Style

- 2-space indentation
- Private functions prefixed with `_`
- Type hints on function signatures
- Standard import order: stdlib, third-party, local
