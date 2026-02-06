#!/usr/bin/env python3
import os
import csv
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Generator, Tuple
from datetime import datetime

# Third-party imports
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table
from rich.console import Console
from openai import OpenAI, APIError

# Local imports
try:
  from arguments import parse_arguments
  from utilities import log_timing
except ImportError:
  # Fallback for standalone testing
  import argparse
  def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", help="The question to ask")
    parser.add_argument("--model", "-m", help="Model name")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--context", "-c", help="Context ID", default=None)
    return parser.parse_args()
  def log_timing(func):
    return func

# Initialize Console globally for UI
console = Console()

# --- Configuration & Data Structures ---

MODEL_DEFAULT = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
CONFIG_PATH = Path(__file__).parent / ".hf_config.json"

def load_config() -> dict:
  if CONFIG_PATH.exists():
    try:
      return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
      return {}
  return {}

def save_config(config: dict):
  CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

@dataclass
class ModelInfo:
  """Represents a row from the models CSV for type safety."""
  name: str
  provider: str
  tags: str
  input_cost: float
  output_cost: float
  context_length: str
  latency: str
  throughput: str
  tools: str
  structured: str

  @property
  def total_cost(self) -> float:
    return self.input_cost + self.output_cost

  @property
  def full_id(self) -> str:
    """Returns the format expected by the API (model:provider)."""
    if ":" in self.name:
      return self.name
    return f"{self.name}:{self.provider}"

# --- Managers (Logic Layer) ---

class ModelRegistry:
  """Handles loading and searching for models."""
  
  def __init__(self, csv_path: str = "models.csv"):
    self.csv_path = Path(csv_path)
    self.models: List[ModelInfo] = self._load_models()

  def _load_models(self) -> List[ModelInfo]:
    if not self.csv_path.exists():
      return []
    
    models = []
    try:
      with open(self.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
          # Safe conversion of costs
          i_cost = float(row.get('Input $/1M', 0)) if row.get('Input $/1M') not in ('-', '') else 0.0
          o_cost = float(row.get('Output $/1M', 0)) if row.get('Output $/1M') not in ('-', '') else 0.0
          
          models.append(ModelInfo(
            name=row['Model'],
            provider=row['Provider'],
            tags=row.get('Tags', ''),
            input_cost=i_cost,
            output_cost=o_cost,
            context_length=row.get('Context', '-'),
            latency=row.get('Latency(s)', '-'),
            throughput=row.get('Throughput(t/s)', '-'),
            tools=row.get('Tools', ''),
            structured=row.get('Structured', '')
          ))
    except Exception as e:
      console.print(f"[yellow]Error parsing CSV: {e}[/yellow]")
    return models

  def find_best_provider(self, model_name: str, strategy: str = "cheapest") -> Optional[ModelInfo]:
    """Finds a model, prioritizing specific providers or cost."""
    matches = [m for m in self.models if model_name.lower() in m.name.lower()]
    
    if not matches:
      return None

    # Determine strategy
    if strategy == "cheapest":
      # Filter those with valid pricing (non-zero or known)
      priced = [m for m in matches if m.input_cost > 0 or m.output_cost > 0]
      if priced:
        return min(priced, key=lambda m: m.total_cost)
      return matches[0]
    
    # Add 'fastest' logic here if needed
    return matches[0]

  def get_specific_model(self, model_name: str, provider: str) -> Optional[ModelInfo]:
    for m in self.models:
      if m.name == model_name and m.provider == provider:
        return m
    return None

class ContextManager:
  """Handles loading and saving chat history."""
  
  def __init__(self, context_id: str = None):
    self.base_dir = Path(tempfile.gettempdir()) / "huggingchat_contexts"
    self.base_dir.mkdir(parents=True, exist_ok=True)
    self.context_id = context_id
    self.messages: List[Dict[str, str]] = []
    
    if self.context_id == "new":
      self.save()
      console.print("[dim]Cleared default context history ('--context new' argument specified)[/dim]")
      self.context_id = "default"

    if self.context_id:
      self._load()

  def _load(self):
    file_path = self.context_path
    if file_path.exists():
      try:
        with open(file_path, 'r', encoding='utf-8') as f:
          data = json.load(f)
          self.messages = data.get('messages', [])
        console.print(f"[dim]Loaded {len(self.messages)} messages from context '{file_path}'[/dim]")
      except Exception as e:
        console.print(f"[red]Failed to load context: {e}[/red]")

  @property
  def context_path(self):
    return self.base_dir / f"{self.context_id}.json"

  def save(self):
    if not self.context_id:
      return

    data = {
      "id": self.context_id,
      "timestamp": datetime.now().isoformat(),
      "messages": self.messages
    }
    with open(self.context_path, 'w', encoding='utf-8') as f:
      json.dump(data, f, indent=2, ensure_ascii=False)

  def add_message(self, role: str, content: str):
    self.messages.append({"role": role, "content": content})

  def get_messages_for_api(self, current_question: str) -> List[Dict[str, str]]:
    # Return history + current question
    msgs = list(self.messages)
    msgs.append({"role": "user", "content": current_question})
    return msgs

class LLMClient:
  """Wrapper for the API interactions."""
  
  def __init__(self):
    self.api_key = self._get_api_key()
    self.client = OpenAI(
      base_url="https://router.huggingface.co/v1",
      api_key=self.api_key,
    )

  @staticmethod
  def _get_api_key() -> str:
    token = os.getenv("HF_TOKEN")
    if token:
      return token
    
    token_path = Path(".HF_TOKEN")
    if token_path.exists():
      return token_path.read_text().strip()
      
    console.print("[red]ERROR: HF_TOKEN not found in env or .HF_TOKEN file[/red]")
    sys.exit(1)

  @log_timing
  def chat(self, model_id: str, messages: List[Dict[str, str]], stream: bool = True) -> Generator[str, None, None] | str:
    try:
      stream_options = {"include_usage": True} if stream else None
      response = self.client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=stream,
        stream_options=stream_options,
      )
      
      if stream:
        return response
      else:
        return response.choices[0].message.content
    except APIError as e:
      console.print(f"[red]API Error: {e.message}[/red]")
      return ""

# --- Presentation Layer (UI) ---

class App:
  def __init__(self):
    self.registry = ModelRegistry()
    self.llm = LLMClient()

  def list_models(self, search: str = None):
    models = self.registry.models
    if search:
      models = [m for m in models if search.lower() in m.name.lower()]

    if not models:
      console.print("[yellow]No models found[/yellow]")
      return

    table = Table(title="Available Models")
    table.add_column("Model", style="cyan", no_wrap=False)
    table.add_column("Provider", style="green")
    table.add_column("Input $/1M", style="blue")
    table.add_column("Output $/1M", style="blue")
    table.add_column("Context", style="magenta")

    # Deduplicate by showing the cheapest provider for each unique model name
    seen = set()
    for m in models:
      if m.name not in seen:
        seen.add(m.name)
        best = self.registry.find_best_provider(m.name)
        if best:
          table.add_row(
            best.name, best.provider, 
            f"{best.input_cost}", f"{best.output_cost}", 
            best.context_length
          )
    console.print(table)

  def switch_model(self):
    """Interactively switch the default model."""
    # Deduplicate models, showing cheapest provider for each
    seen = {}
    for m in self.registry.models:
      if m.name not in seen:
        best = self.registry.find_best_provider(m.name)
        if best:
          seen[m.name] = best

    models = list(seen.values())
    if not models:
      console.print("[yellow]No models found[/yellow]")
      return

    # Show current default
    config = load_config()
    current = config.get("default_model", MODEL_DEFAULT)
    console.print(f"\n[bold]Current default model:[/bold] [cyan]{current}[/cyan]\n")

    # Display numbered list
    table = Table(title="Available Models")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Model", style="cyan", no_wrap=False)
    table.add_column("Provider", style="green")
    table.add_column("Input $/1M", style="blue")
    table.add_column("Output $/1M", style="blue")
    table.add_column("Context", style="magenta")

    for i, m in enumerate(models, 1):
      table.add_row(str(i), m.name, m.provider, f"{m.input_cost}", f"{m.output_cost}", m.context_length)

    console.print(table)

    # Prompt for selection
    console.print(f"\nEnter a number (1-{len(models)}) to select a model, or 'q' to cancel:")
    try:
      choice = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
      console.print("\n[dim]Cancelled.[/dim]")
      return

    if choice.lower() == "q":
      console.print("[dim]Cancelled.[/dim]")
      return

    try:
      idx = int(choice) - 1
      if not (0 <= idx < len(models)):
        raise ValueError
    except ValueError:
      console.print("[red]Invalid selection.[/red]")
      return

    selected = models[idx]
    config["default_model"] = selected.name
    save_config(config)
    console.print(f"\n[bold green]Default model switched to:[/bold green] [cyan]{selected.name}[/cyan] ({selected.provider})")

  def _display_model_details(self, model: ModelInfo):
    """Displays technical details about the selected model."""
    console.print(f"  [cyan]Cost:[/cyan]     [magenta]${model.input_cost}/1M in, ${model.output_cost}/1M out[/magenta]")
    console.print(f"  [cyan]Details:[/cyan]  [magenta]{model.context_length} | Latency: {model.latency}s | Throughput: {model.throughput} t/s[/magenta]\n")

  def run_prompt(self, question: str, model_name: str = None, provider: str = None, context_id: str = None):
    if not question:
      console.print("[red]Error: Question is required.[/red]")
      return

    # 1. Resolve Configuration
    config_default = load_config().get("default_model", MODEL_DEFAULT)
    model_name = model_name or os.getenv("HF_MODEL", config_default)
    provider = provider or os.getenv("HF_PROVIDER")

    # 2. Resolve Model Info
    selected_model: Optional[ModelInfo] = None
    if provider:
      selected_model = self.registry.get_specific_model(model_name, provider)
    else:
      selected_model = self.registry.find_best_provider(model_name)

    # Fallback if model not in CSV but passed as arg
    full_model_id = selected_model.full_id if selected_model else (f"{model_name}:{provider}" if provider else model_name)

    # 3. Setup Context
    ctx_mgr = ContextManager(context_id)
    
    # 4. Display UI Status
    console.print("\n[bold]Processing...[/bold]\n", style="yellow")
    console.print(f"  [cyan]Question:[/cyan] {question}")
    console.print(f"  [cyan]Target:[/cyan]   {full_model_id}")
    if context_id:
      console.print(f"  [cyan]Context:[/cyan]  {context_id}")
    
    if selected_model:
      self._display_model_details(selected_model)
    else:
      console.print(f"[yellow]Warning: Model '{model_name}' not found in CSV. Using raw string.[/yellow]\n")

    # 5. Execute API Call
    messages = ctx_mgr.get_messages_for_api(question)
    response_stream = self.llm.chat(full_model_id, messages, stream=True)

    full_response = ""
    usage = None

    # 6. Render Response
    with Live(console=console, refresh_per_second=10) as live:
      for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta.content:
          content = chunk.choices[0].delta.content
          full_response += content
          live.update(Markdown(full_response))
        if chunk.usage:
          usage = chunk.usage

    console.print("\n")

    # 6b. Display Cost
    if usage:
      prompt_tokens = usage.prompt_tokens or 0
      completion_tokens = usage.completion_tokens or 0
      total_tokens = prompt_tokens + completion_tokens
      cost_line = f"[dim]Tokens: {prompt_tokens} in + {completion_tokens} out = {total_tokens} total[/dim]"
      if selected_model and (selected_model.input_cost > 0 or selected_model.output_cost > 0):
        input_cost = (prompt_tokens / 1_000_000) * selected_model.input_cost
        output_cost = (completion_tokens / 1_000_000) * selected_model.output_cost
        total_cost = input_cost + output_cost
        cost_line += f"[dim] | Cost: ${total_cost:.6f}[/dim]"
      console.print(cost_line)

    # 7. Save Context
    if context_id and full_response:
      ctx_mgr.add_message("user", question)
      ctx_mgr.add_message("assistant", full_response)
      ctx_mgr.save()
      console.print(f"[dim]Context saved to '{ctx_mgr.context_path}'[/dim]")

if __name__ == "__main__":
  import sys
  
  # Simple argument parsing wrapper if not imported
  args = parse_arguments()
  app = App()

  if args.list_models:
    app.list_models()
  elif args.switch_model:
    app.switch_model()
  else:
    app.run_prompt(args.question, args.model, context_id=args.context)
