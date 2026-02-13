#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export CHAT_LLM_CALLER_DIR="$PWD"
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python3" "$PROJECT_DIR/chat.py" "$@"
