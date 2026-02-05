#!/usr/bin/env python3
import os, sys
from pathlib import Path

from rich.markdown import Markdown
from rich.live import Live
from openai import OpenAI, APIError

from arguments import parse_arguments
from console import console
from utilities import log_timing

#MODEL_DEFAULT = "EssentialAI/rnj-1-instruct:together"
MODEL_DEFAULT = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"


_client = None  # Reusable client instance

def build_messages(question, system_prompt=None):
  messages = []
  if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
  messages.append({"role": "user", "content": question})
  return messages


def get_api_key():
  token = os.getenv("HF_TOKEN")
  if token:
    return token

  token_path = Path(".HF_TOKEN")
  if token_path.exists():
    return token_path.read_text().strip()
  raise SystemExit("ERROR: Need to configure your HF_TOKEN")


def get_client():
  global _client
  if _client is None:
    _client = OpenAI(
      base_url="https://router.huggingface.co/v1",
      api_key=get_api_key(),
    )
  return _client


def list_models():
  from models import all_models
  print(all_models)


def set_default_model(model):
  os.environ["HF_MODEL"] = model


def get_default_model():
  return os.getenv("HF_MODEL", MODEL_DEFAULT)


def prompt(question, model=None, stream=True, system_prompt=None):
  if not model:
    model = get_default_model()
  if not question:
    raise SystemExit("ERROR: You gotta ask a question")

  console.print("\nProcessing...\n", style="yellow")
  console.print(f"  Question: {question}")
  console.print(f"  Model: {model}\n")

  try:
    if stream:
      answer = stream_answer(question, model, system_prompt)
    else:
      answer = get_answer(question, model, system_prompt)
      console.print(Markdown(answer))
  except APIError as e:
    console.print(f"[red]API Error: {e.message}[/red]")
    return

  console.print("\n")


@log_timing
def get_answer(question, model, system_prompt=None):
  messages = []
  if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
  messages.append({"role": "user", "content": question})

  completion = get_client().chat.completions.create(
    model=model,
    messages=build_messages(question, system_prompt),
  )
  return completion.choices[0].message.content


@log_timing
def stream_answer(question, model, system_prompt=None):
  stream = get_client().chat.completions.create(
    model=model,
    messages=build_messages(question, system_prompt),
    stream=True,
  )

  full_response = ""

  with Live(console=console, refresh_per_second=10) as live:
    for chunk in stream:
      if chunk.choices and chunk.choices[0].delta.content:
        full_response += chunk.choices[0].delta.content
        live.update(Markdown(full_response))

  return full_response


if __name__ == "__main__":
  question, model, system_prompt = parse_arguments()
  prompt(question, model, system_prompt)
