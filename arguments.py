#!/usr/bin/env python3
import argparse, sys, os
from dataclasses import dataclass, field
from console import console

@dataclass
class Args:
  question: str = None
  list_models: bool = False
  switch_model: bool = False
  switch_router: bool = False
  model: str = None
  context: str = None

  def __post_init__(self):
    if self.question == "-":
      self.question = sys.stdin.read().strip()

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
    "-l",
    "-lm",
    action="store_true",
    help="List available models that are compatible with Huggle Face router/providers, then exit"
  )

  group.add_argument(
    "--switch-model",
    "-s",
    action="store_true",
    help="Interactively switch the default model, then exit"
  )

  group.add_argument(
    "--switch-router",
    "-sr",
    action="store_true",
    help="Interactively switch the active API router (e.g., HuggingFace, Google), then exit"
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
    "-c", "--context",
    nargs="?",
    const="default",
    metavar="CONTEXT_ID",
    help="Maintain chat context in temp folder (optionally specify 'new' for fresh context)"
  )

  # Positional argument (will be used if --question is not provided)
  group.add_argument(
    "positional_question",
    nargs="?",
    type=str,
    help="Question as a positional argument (alternative to --question)"
  )

  args = parser.parse_args()
  return Args(
    list_models=bool(args.list_models),
    switch_model=bool(args.switch_model),
    switch_router=bool(args.switch_router),
    question=args.question or args.positional_question,
    model=args.model,
    context=args.context
  )
