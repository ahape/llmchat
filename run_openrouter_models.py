#!/usr/bin/env python3
"""Run chat.py against every OpenRouter model in parallel."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
CHAT_SCRIPT = ROOT / "chat.py"
DEFAULT_MODELS_FILE = ROOT / "routers" / "openrouter" / "models.csv"


@dataclass(slots=True)
class RunResult:
  model: str
  success: bool
  returncode: Optional[int]
  message: str
  log_path: Optional[Path]


def _default_workers() -> int:
  cpu = os.cpu_count() or 4
  return max(2, min(32, cpu * 2))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Invoke chat.py for every OpenRouter model in parallel.",
  )
  parser.add_argument(
    "question",
    help="Question to send to each model. Use '-' to read from stdin.",
  )
  parser.add_argument(
    "--models-file",
    type=Path,
    default=DEFAULT_MODELS_FILE,
    help="Path to the OpenRouter models.csv file.",
  )
  parser.add_argument(
    "--router",
    default="openrouter",
    help="Router key to pass through to chat.py (default: openrouter).",
  )
  parser.add_argument(
    "--max-workers",
    type=int,
    help="Maximum number of concurrent chat.py processes (default: 2 * CPU, capped at 32).",
  )
  parser.add_argument(
    "--timeout",
    type=int,
    default=180,
    help="Seconds to wait before timing out a model invocation.",
  )
  parser.add_argument(
    "--include-duplicates",
    action="store_true",
    help="Keep duplicate model names (provider variants).",
  )
  parser.add_argument(
    "--limit",
    type=int,
    help="Only run the first N models (after filtering/uniquing).",
  )
  parser.add_argument(
    "--log-dir",
    type=Path,
    help="Optional directory where per-model stdout/stderr logs will be written.",
  )
  return parser.parse_args()


def _read_question(question_arg: str) -> str:
  if question_arg == "-":
    data = sys.stdin.read().strip()
    if not data:
      raise ValueError("No question provided on stdin")
    return data
  return question_arg


def load_models(csv_path: Path, include_duplicates: bool) -> List[str]:
  if not csv_path.exists():
    raise FileNotFoundError(f"Models file not found: {csv_path}")

  models: List[str] = []
  seen = set()
  with open(csv_path, "r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      name = (row.get("Model") or "").strip()
      if not name:
        continue
      if not include_duplicates and name in seen:
        continue
      seen.add(name)
      models.append(name)
  return models


def _sanitize_filename(name: str) -> str:
  safe = name.replace("/", "__").replace(" ", "_").replace(":", "-")
  return "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-", "."})


def _write_log(log_dir: Optional[Path], model: str, stdout: str, stderr: str) -> Optional[Path]:
  if not log_dir:
    return None
  log_dir.mkdir(parents=True, exist_ok=True)
  filename = f"{_sanitize_filename(model)}.log"
  log_path = log_dir / filename
  combined = ["# stdout\n", stdout, "\n# stderr\n", stderr]
  log_path.write_text("".join(combined), encoding="utf-8")
  return log_path


def run_model(
  model: str,
  question: str,
  router: str,
  timeout: int,
  log_dir: Optional[Path],
) -> RunResult:
  if not CHAT_SCRIPT.exists():
    return RunResult(model, False, None, f"chat.py not found at {CHAT_SCRIPT}", None)

  cmd = [
    sys.executable,
    str(CHAT_SCRIPT),
    question,
    "--router",
    router,
    "--model",
    model,
  ]

  try:
    completed = subprocess.run(
      cmd,
      cwd=ROOT,
      capture_output=True,
      text=True,
      encoding="utf-8",
      errors="replace",
      timeout=timeout,
    )
  except subprocess.TimeoutExpired as exc:
    log_path = _write_log(log_dir, model, exc.stdout or "", exc.stderr or "")
    return RunResult(model, False, None, f"timed out after {timeout}s", log_path)
  except OSError as exc:
    return RunResult(model, False, None, str(exc), None)

  stdout = completed.stdout or ""
  stderr = completed.stderr or ""
  log_path = _write_log(log_dir, model, stdout, stderr)

  if completed.returncode == 0:
    snippet = stdout.strip().splitlines()[-1] if stdout.strip() else "ok"
    return RunResult(model, True, 0, snippet, log_path)

  message = stderr.strip() or stdout.strip() or f"exit code {completed.returncode}"
  return RunResult(model, False, completed.returncode, message, log_path)


def summarize(results: Iterable[RunResult]) -> None:
  total = 0
  succeeded = 0
  try:
    for result in results:
      total += 1
      status = "SUCCESS" if result.success else "FAIL"
      log_hint = f" (log: {result.log_path})" if result.log_path else ""
      print(f"[{status}] {result.model}: {result.message}{log_hint}")
      if result.success:
        succeeded += 1
  finally:
    if total > 0:
      failed = total - succeeded
      print("-" * 60)
      print(f"Completed {total} models | {succeeded} succeeded | {failed} failed")


def main() -> int:
  args = parse_args()

  if not args.log_dir:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tmp = Path.home() / ".gemini" / "tmp" / "llmchat"
    args.log_dir = tmp / f"llm_responses_{timestamp}"
    print(f"Logging outputs to {args.log_dir}", file=sys.stderr)

  try:
    question = _read_question(args.question)
  except ValueError as err:
    print(err, file=sys.stderr)
    return 1

  try:
    models = load_models(args.models_file, args.include_duplicates)
  except (OSError, FileNotFoundError) as err:
    print(err, file=sys.stderr)
    return 1

  if not models:
    print("No models found in the provided CSV", file=sys.stderr)
    return 1

  if args.limit:
    models = models[:args.limit]

  workers = args.max_workers or _default_workers()
  if workers < 1:
    print("max-workers must be >= 1", file=sys.stderr)
    return 1

  with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {
      executor.submit(run_model, model, question, args.router, args.timeout, args.log_dir): model
      for model in models
    }

    def _generate_results() -> Iterable[RunResult]:
      for future in as_completed(futures):
        model = futures[future]
        try:
          yield future.result()
        except Exception as exc:  # pragma: no cover - defensive
          yield RunResult(model, False, None, f"worker raised: {exc}", None)

    try:
      summarize(_generate_results())
    except KeyboardInterrupt:
      print("\n[!] Interrupted by user. Cancelling pending tasks...", file=sys.stderr)
      for f in futures:
        f.cancel()
      return 130

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
