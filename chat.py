#!./.venv/bin/python3
import os
import csv
import json
import time
import tempfile
import threading
import argparse
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

# Initialize Console globally for UI
console = Console()

@dataclass
class Args:
  question: str = None
  list_models: bool = False
  switch_model: bool = False
  switch_router: bool = False
  model: str = None
  context: str = None
  compose: bool = False

  def __post_init__(self):
    if self.compose:
      import tempfile
      import subprocess
      # Create a temp file with .md extension for better syntax highlighting
      with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        temp_path = f.name

      try:
        # Open nvim and wait for it to close
        result = subprocess.run(['nvim', temp_path], check=False)
        if result.returncode == 0:
          # Read the composed content
          with open(temp_path, 'r', encoding='utf-8') as f:
            composed = f.read().strip()
          if composed:
            self.question = composed
          else:
            console.print("[yellow]No content composed. Exiting.[/yellow]")
            sys.exit(0)
        else:
          console.print("[red]Editor exited with error.[/red]")
          sys.exit(1)
      finally:
        # Clean up temp file
        if os.path.exists(temp_path):
          os.remove(temp_path)
    elif self.question == "-":
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
 %(prog)s --compose                              # Opens Neovim to compose question
 %(prog)s --compose --model "..."                # Compose with specific model
    """
  )

  # Mutually exclusive group for list-models
  group = parser.add_mutually_exclusive_group()

  group.add_argument(
    "--list-models",
    "-l",
    "-lm",
    action="store_true",
    help="List available models for the active router, then exit"
  )

  group.add_argument(
    "--switch-model",
    "-s",
    "-sm",
    action="store_true",
    help="Interactively switch the default model, then exit"
  )

  group.add_argument(
    "--switch-router",
    "-sr",
    action="store_true",
    help="Interactively switch the active API router, then exit"
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
    help="Model to use for processing"
  )

  parser.add_argument(
    "--context",
    "-c",
    nargs="?",
    const="default",
    metavar="CONTEXT_ID",
    help="Maintain chat context in temp folder (optionally specify 'new' for fresh context)"
  )

  parser.add_argument(
    "--compose",
    "-vim",
    action="store_true",
    help="Open Neovim to compose your question, then continue normally"
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
    context=args.context,
    compose=bool(args.compose)
  )

# --- Configuration & Data Structures ---

@dataclass
class RouterConfig:
  key: str              # "google", "openrouter"
  name: str             # "Google AI", "OpenRouter"
  base_url: str
  api_key_env: str      # env var name for API key
  api_key_file: str     # fallback file for API key
  csv_path: str         # relative path to models CSV
  default_model: str    # fallback default model

ROUTERS: Dict[str, RouterConfig] = {
  "google": RouterConfig(
    key="google", name="Google AI",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key_env="GOOGLE_API_KEY", api_key_file=".GOOGLE_API_KEY",
    csv_path="routers/google/models.csv",
    default_model="gemini-2.0-flash",
  ),
  "openrouter": RouterConfig(
    key="openrouter", name="OpenRouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY", api_key_file=".OPENROUTER_API_KEY",
    csv_path="routers/openrouter/models.csv",
    default_model="anthropic/claude-3.5-sonnet",
  ),
}
ROUTER_DEFAULT = "openrouter"

CONFIG_PATH = Path(__file__).parent / ".llm_config.json"

def load_config() -> dict:
  if CONFIG_PATH.exists():
    try:
      return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
      return {}
  return {}

def save_config(config: dict):
  CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

def _get_api_key(router: RouterConfig) -> str:
  """Read API key from env var or file for the given router."""
  token = os.getenv(router.api_key_env)
  if token:
    return token

  token_path = Path(router.api_key_file)
  if token_path.exists():
    return token_path.read_text().strip()

  console.print(f"[red]ERROR: {router.api_key_env} not found in env or {router.api_key_file} file[/red]")
  import sys
  sys.exit(1)

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

# --- Managers (Logic Layer) ---

class ModelRegistry:
  """Handles loading and searching for models."""

  def __init__(self, csv_path: str):
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
            structured=row.get('Structured', ''),
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
    self.base_dir = Path(tempfile.gettempdir()) / "llm_chat_contexts"
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

  def __init__(self, base_url: str, api_key: str):
    self.api_key = api_key
    self.client = OpenAI(
      base_url=base_url,
      api_key=api_key,
    )

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
    config = load_config()
    router_key = config.get("router", ROUTER_DEFAULT)
    self.router = ROUTERS.get(router_key, ROUTERS[ROUTER_DEFAULT])
    self.registry = ModelRegistry(csv_path=self.router.csv_path)
    self._llm: Optional[LLMClient] = None

  @property
  def llm(self) -> LLMClient:
    if self._llm is None:
      api_key = _get_api_key(self.router)
      self._llm = LLMClient(base_url=self.router.base_url, api_key=api_key)
    return self._llm

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
    router_config = config.get(self.router.key, {})
    current = router_config.get("default_model", self.router.default_model)
    console.print(f"\n[bold]Router:[/bold] [green]{self.router.name}[/green]")
    console.print(f"[bold]Current default model:[/bold] [cyan]{current}[/cyan]\n")

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
    if self.router.key not in config:
      config[self.router.key] = {}
    config[self.router.key]["default_model"] = selected.name
    save_config(config)
    console.print(f"\n[bold green]Default model switched to:[/bold green] [cyan]{selected.name}[/cyan] ({selected.provider})")

  def switch_router(self):
    """Interactively switch the active router."""
    routers = list(ROUTERS.values())
    config = load_config()
    current_key = config.get("router", ROUTER_DEFAULT)

    table = Table(title="Available Routers")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Active", style="yellow", justify="center")

    for i, r in enumerate(routers, 1):
      active = "*" if r.key == current_key else ""
      table.add_row(str(i), r.key, r.name, active)

    console.print(table)
    console.print(f"\nEnter a number (1-{len(routers)}) to select a router, or 'q' to cancel:")
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
      if not (0 <= idx < len(routers)):
        raise ValueError
    except ValueError:
      console.print("[red]Invalid selection.[/red]")
      return

    selected = routers[idx]
    config["router"] = selected.key
    save_config(config)
    console.print(f"\n[bold green]Router switched to:[/bold green] [cyan]{selected.name}[/cyan]")

  def run_prompt(self, question: str, model_name: str = None, provider: str = None, context_id: str = None):
    if not question:
      console.print("[red]Error: Question is required.[/red]")
      return

    # 1. Resolve Configuration
    config = load_config()
    router_config = config.get(self.router.key, {})
    config_default = router_config.get("default_model", self.router.default_model)
    model_name = model_name or config_default

    # 2. Resolve Model Info
    selected_model: Optional[ModelInfo] = None
    if provider:
      selected_model = self.registry.get_specific_model(model_name, provider)
    else:
      selected_model = self.registry.find_best_provider(model_name)

    # Fallback if model not in CSV but passed as arg
    full_model_id = selected_model.name if selected_model else model_name

    # 3. Setup Context
    ctx_mgr = ContextManager(context_id)

    # 4. Display UI Status
    console.print(f"  [cyan]Slug:[/cyan]   {full_model_id}")
    console.print(f"  [cyan]Router:[/cyan]   {self.router.key}")
    if context_id:
      console.print(f"  [cyan]Context ID:[/cyan]  {context_id}")

    # 5. Execute API Call (in separate thread to allow CTRL+C)
    console.print("\n[dim italic]Connecting to API...[/dim italic]\n")
    messages = ctx_mgr.get_messages_for_api(question)

    # Thread-safe container for the response stream
    response_container = {"stream": None, "error": None, "ready": False}

    def _make_request():
      try:
        response_container["stream"] = self.llm.chat(full_model_id, messages, stream=True)
      except Exception as e:
        response_container["error"] = e
      finally:
        response_container["ready"] = True

    # Start request in background thread
    request_thread = threading.Thread(target=_make_request, daemon=True)
    request_thread.start()

    # Wait for request to complete, but allow CTRL+C to interrupt
    try:
      while not response_container["ready"]:
        time.sleep(0.1)  # Check every 100ms
    except KeyboardInterrupt:
      console.print("\n\n[yellow]Request cancelled by user (CTRL+C)[/yellow]")
      return

    # Check if request failed
    if response_container["error"]:
      console.print(f"[red]Request failed: {response_container['error']}[/red]")
      return

    response_stream = response_container["stream"]
    full_response = ""
    usage = None
    token_count = 0
    first_token_received = False
    start_time = time.time()
    last_token_time = start_time

    # 6. Render Response
    try:
      with Live(console=console, refresh_per_second=10) as live:
        # Show waiting status until first token
        live.update("[dim italic]Waiting for response...[/dim italic]")

        for chunk in response_stream:
          current_time = time.time()

          if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            token_count += 1
            last_token_time = current_time

            # Mark first token received
            if not first_token_received:
              first_token_received = True
              time_to_first_token = current_time - start_time
              console.print(f"[dim]First token received in {time_to_first_token:.2f}s[/dim]\n")

            # Update with response content
            live.update(Markdown(full_response))

          # Detect potential hang (no tokens for 30+ seconds after first token)
          if first_token_received and (current_time - last_token_time) > 30:
            live.update(Markdown(full_response + "\n\n[yellow italic]⚠ No tokens received for 30s, stream may be stalled...[/yellow italic]"))

          if chunk.usage:
            usage = chunk.usage
    except KeyboardInterrupt:
      console.print("\n\n[yellow]Response cancelled by user (CTRL+C)[/yellow]")
      if full_response:
        console.print(f"[dim]Partial response received ({len(full_response)} characters)[/dim]")
      return

    # Calculate final timing stats
    end_time = time.time()
    total_time = end_time - start_time

    console.print("\n")

    # 6b. Display Cost and Performance
    if usage:
      prompt_tokens = usage.prompt_tokens or 0
      completion_tokens = usage.completion_tokens or 0
      total_tokens = prompt_tokens + completion_tokens
      tokens_per_sec = completion_tokens / total_time if total_time > 0 else 0
      cost_line = f"[dim]Tokens: {prompt_tokens} in + {completion_tokens} out = {total_tokens} total[/dim]"
      if selected_model and (selected_model.input_cost > 0 or selected_model.output_cost > 0):
        input_cost = (prompt_tokens / 1_000_000) * selected_model.input_cost
        output_cost = (completion_tokens / 1_000_000) * selected_model.output_cost
        total_cost = input_cost + output_cost
        cost_line += f"[dim] | Cost: ${total_cost:.6f}[/dim]"
      cost_line += f"[dim] | Time: {total_time:.2f}s ({tokens_per_sec:.1f} tokens/s)[/dim]"
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
  elif args.switch_router:
    app.switch_router()
  else:
    app.run_prompt(args.question, args.model, context_id=args.context)
