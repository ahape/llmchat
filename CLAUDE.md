# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A CLI tool for querying LLMs via the HuggingFace Router API (OpenAI-compatible). It sends questions to models listed on HuggingFace's inference providers, streams responses with rich markdown rendering, and tracks token usage/cost.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Unix/macOS
pip install -r requirements.txt
```

Requires `HF_TOKEN` environment variable (HuggingFace API token). Alternatively, place token in a `.HF_TOKEN` file in the project root.

## Running

```bash
python3 chat.py "your question here"
python3 chat.py --question "your question" --model "deepseek-ai/DeepSeek-V3.2"
python3 chat.py --list-models          # show available models
python3 chat.py --switch-model         # interactively change default model
python3 chat.py "question" --context   # maintain chat history across calls
python3 chat.py "question" -c new      # start fresh context
```

There are no tests, linter, or build system.

## Architecture

Four source files, no package structure:

- **`chat.py`** — Entry point and all core logic. Contains four classes:
  - `ModelRegistry` — Loads `models.csv`, finds cheapest provider for a given model name
  - `ContextManager` — Persists multi-turn chat history as JSON files in system temp dir
  - `LLMClient` — OpenAI SDK wrapper pointing at `router.huggingface.co/v1`
  - `App` — Orchestrates everything: resolves model, calls API, streams/renders response with `rich.Live`, displays token cost
- **`arguments.py`** — argparse setup, returns an `Args` dataclass. Question can be positional or `--question`/`-q`
- **`console.py`** — Single shared `rich.Console` instance
- **`utilities.py`** — `@log_timing` decorator for measuring function execution time

## Key Details

- Default model is stored in `.hf_config.json` (persisted via `--switch-model`), falling back to `HF_MODEL` env var, then `MODEL_DEFAULT` constant
- Model resolution priority: `--model` arg > `HF_MODEL` env > `.hf_config.json` > hardcoded default (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`)
- `models.csv` contains provider/pricing/context metadata; `ModelRegistry.find_best_provider()` picks cheapest by default
- API model IDs use `model_name:provider` format (e.g., `deepseek-ai/DeepSeek-V3.2:novita`)
- All console output uses `rich` library with color markup (`[cyan]`, `[red]`, etc.)

## Code Style

- 2-space indentation
- Private functions prefixed with `_`
- Type hints on function signatures
- Standard import order: stdlib, third-party, local
