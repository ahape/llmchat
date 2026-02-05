#!/usr/bin/env python3
import argparse, sys, os
from console import console

SYSTEM_PROMPTS = {}
SYSTEM_PROMPTS.setdefault("cli",
"""

  You are a command line expert (zsh, macOS, tmux, vim). Reply ONLY with the
  command or minimal config needed. No preamble. No numbered steps. No "you can
  use". Format: command/config first, then ONE short sentence explaining if
  necessary. If it's a config change, show the line(s) to add.

""".rstrip())

def parse_arguments():
  parser = argparse.ArgumentParser(
    description="Process a question with an optional model parameter",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
 Examples:
 %(prog)s "why is the sky blue?"
 %(prog)s --question "why is the sky blue?"
 %(prog)s --model "deepseek-ai/DeepSeek-V3.2" --question "..."
 %(prog)s --question "..." --model "deepseek-ai/DeepSeek-V3.2"
 %(prog)s "..." --model "deepseek-ai/DeepSeek-V3.2"
    """
  )

  # Mutually exclusive group for list-models
  group = parser.add_mutually_exclusive_group()

  group.add_argument(
    "--list-models",
    action="store_true",
    help="List available models and exit"
  )

  # Optional arguments (only valid when not using --list-models)
  group.add_argument(
    "--question",
    "-q",
    type=str,
    help="The question to process"
  )

  parser.add_argument(
    "--model",
    "-m",
    type=str,
    help="Model to use for processing (default: env['HF_MODEL'])"
  )

  parser.add_argument(
    "-s", "--system",
    type=str,
    default=None,
    help="System prompt to use"
  )

  parser.add_argument(
    "--cli",
    action="store_true",
    help="Use CLI guru system prompt"
  )

  # Positional argument (will be used if --question is not provided)
  group.add_argument(
    "positional_question",
    nargs="?",
    type=str,
    help="Question as a positional argument (alternative to --question)"
  )

  args = parser.parse_args()

  # Handle list-models argument
  if args.list_models:
    return None, None, None, True  # Special return for list-models

  # Determine which question to use
  if args.question:
    question = args.question
  elif args.positional_question:
    question = args.positional_question
  else:
    parser.error("No question provided. Use either positional argument or --question flag.")

  if question == "-":
    question = sys.stdin.read().strip()

  if args.cli:
    system_prompt = SYSTEM_PROMPTS["cli"]
  elif args.system:
    system_prompt = SYSTEM_PROMPTS.get(args.system, args.system)  # Allow preset name OR custom string
  else:
    system_prompt = None

  if system_prompt:
    console.print("using the following system_prompt", style="yellow")
    console.print(system_prompt, style="yellow")

  # Then return it with False to indicate not list-models mode
  return question, args.model, system_prompt, False
