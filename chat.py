#!/usr/bin/env python3
import os, sys, csv
from pathlib import Path
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table
from openai import OpenAI, APIError
from arguments import parse_arguments
from console import console
from utilities import log_timing

MODEL_DEFAULT = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
_client = None  # Single reusable client instance

def list_models(search=None):
  """List available models from CSV"""
  models = _load_models_csv()

  if search:
    models = [m for m in models if search.lower() in m['Model'].lower()]

  if not models:
    console.print("[yellow]No models found[/yellow]")
    return

  table = Table(title="Available Models")
  table.add_column("Model", style="cyan", no_wrap=False)
  table.add_column("Provider", style="green")
  table.add_column("Tags", style="yellow")
  table.add_column("Input $/1M", style="blue")
  table.add_column("Output $/1M", style="blue")
  table.add_column("Context", style="magenta")

  # Group by model name and show cheapest
  seen_models = set()
  for model_info in models:
    model_name = model_info['Model']
    if model_name not in seen_models:
      seen_models.add(model_name)
      cheapest = _get_cheapest_provider(model_name)
      if cheapest:
        table.add_row(
          cheapest['Model'],
          cheapest['Provider'],
          cheapest['Tags'],
          cheapest['Input $/1M'],
          cheapest['Output $/1M'],
          cheapest['Context']
        )

  console.print(table)

def set_default_model(model, provider=None):
  """Set the default model and optionally provider"""
  console.print("\n[dim]FYI: You can also change the model via the environment variable HF_MODEL[/dim]")

  os.environ["HF_MODEL"] = model
  if provider:
    os.environ["HF_PROVIDER"] = provider

  # Display model info
  if provider:
    matches = _find_model_info(model, provider)
    model_info = matches[0] if matches else None
  else:
    model_info = _get_cheapest_provider(model)

  if model_info:
    console.print(f"\n[green]✓[/green] Model set to: {model}")
    if provider:
      console.print(f"[green]✓[/green] Provider set to: {provider}")
    _display_model_info(model_info)
  else:
    console.print(f"\n[yellow]⚠[/yellow] Model set to: {model} (not found in CSV)")

def get_default_model():
  """Get the current default model"""
  return os.getenv("HF_MODEL", MODEL_DEFAULT)

def get_default_provider():
  """Get the current default provider"""
  return os.getenv("HF_PROVIDER", None)

def prompt(question, model=None, stream=True, system_prompt=None, provider=None):
  if not model:
    model = get_default_model()
  if not provider:
    provider = get_default_provider()

  if not question:
    raise SystemExit("ERROR: You gotta ask a question")

  # Find model info and determine provider
  model_info = None
  if not provider:
    model_info = _get_cheapest_provider(model)
    if model_info:
      provider = model_info['Provider']
    else:
      console.print(f"[yellow]Warning: Model '{model}' not found in CSV, using model as-is[/yellow]")
  else:
    # Validate provider is available for this model
    matches = _find_model_info(model, provider)
    if matches:
      model_info = matches[0]
    else:
      console.print(f"[yellow]Warning: Provider '{provider}' not found for model '{model}' in CSV[/yellow]")

  # Format model with provider suffix
  if provider:
    model_with_provider = _format_model_with_provider(model, provider)
  else:
    model_with_provider = model

  console.print("\n[bold]Processing...[/bold]\n", style="yellow")
  console.print(f"  [cyan]Question:[/cyan] {question}")
  console.print(f"  [cyan]Model:[/cyan] {model}")
  if provider:
    console.print(f"  [cyan]Provider:[/cyan] {provider}")
  console.print(f"  [cyan]Full Model String:[/cyan] {model_with_provider}\n")

  # Display model details
  if model_info:
    console.print("[dim]Model Details:[/dim]")
    console.print(f"  Cost: ${model_info.get('Input $/1M', 'N/A')}/1M in, ${model_info.get('Output $/1M', 'N/A')}/1M out")
    console.print(f"  Context: {model_info.get('Context', 'N/A')} | Latency: {model_info.get('Latency(s)', 'N/A')}s | Throughput: {model_info.get('Throughput(t/s)', 'N/A')} t/s\n")

  try:
    if stream:
      answer = _stream_answer(question, model_with_provider, system_prompt)
    else:
      answer = _get_answer(question, model_with_provider, system_prompt)
      console.print(Markdown(answer))
  except APIError as e:
    console.print(f"[red]API Error: {e.message}[/red]")
    return

  console.print("\n")

def _load_models_csv(csv_path="models.csv"):
  """Load models from CSV file"""
  models = []
  csv_file = Path(csv_path)

  if not csv_file.exists():
    console.print(f"[yellow]Warning: {csv_path} not found[/yellow]")
    return models

  with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
      models.append(row)

  return models

def _find_model_info(model_name, provider=None):
  """Find model information from CSV, optionally filtered by provider"""
  models = _load_models_csv()

  # Try exact match first
  matches = [m for m in models if m['Model'] == model_name]

  # Filter by provider if specified
  if provider and matches:
    matches = [m for m in matches if m['Provider'] == provider]

  if not matches:
    # Try partial match
    matches = [m for m in models if model_name in m['Model']]
    if provider and matches:
      matches = [m for m in matches if m['Provider'] == provider]

  return matches

def _get_cheapest_provider(model_name):
  """Find the cheapest provider for a given model"""
  matches = _find_model_info(model_name)
  if not matches:
    return None

  # Filter out entries without pricing
  priced = [m for m in matches if m['Input $/1M'] != '-' and m['Output $/1M'] != '-']
  if not priced:
    return matches[0]  # Return first available if no pricing

  # Calculate total cost (input + output) and find minimum
  cheapest = min(priced, key=lambda m: float(m['Input $/1M']) + float(m['Output $/1M']))
  return cheapest

def _get_fastest_provider(model_name):
  """Find the fastest provider for a given model"""
  matches = _find_model_info(model_name)
  if not matches:
    return None

  # Filter out entries without latency data
  with_latency = [m for m in matches if m['Latency(s)'] != '-']
  if not with_latency:
    return matches[0]  # Return first available if no latency data

  # Find minimum latency
  fastest = min(with_latency, key=lambda m: float(m['Latency(s)']))
  return fastest

def _display_model_info(model_info):
  """Display model information in a nice table"""
  table = Table(title=f"Model: {model_info['Model']}")
  table.add_column("Property", style="cyan")
  table.add_column("Value", style="green")

  table.add_row("Provider", model_info['Provider'])
  if model_info['Tags']:
    table.add_row("Tags", model_info['Tags'])
  table.add_row("Input Cost", f"${model_info['Input $/1M']}/1M tokens" if model_info['Input $/1M'] != '-' else 'N/A')
  table.add_row("Output Cost", f"${model_info['Output $/1M']}/1M tokens" if model_info['Output $/1M'] != '-' else 'N/A')
  table.add_row("Context Length", model_info['Context'] if model_info['Context'] != '-' else 'N/A')
  table.add_row("Latency", f"{model_info['Latency(s)']}s" if model_info['Latency(s)'] != '-' else 'N/A')
  table.add_row("Throughput", f"{model_info['Throughput(t/s)']} t/s" if model_info['Throughput(t/s)'] != '-' else 'N/A')
  table.add_row("Tools Support", model_info['Tools'])
  table.add_row("Structured Output", model_info['Structured'])

  console.print(table)

def _build_messages(question, system_prompt=None):
  messages = []
  if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
  messages.append({"role": "user", "content": question})
  return messages

def _get_api_key():
  """Get HuggingFace API key"""
  token = os.getenv("HF_TOKEN")
  if token:
    return token

  token_path = Path(".HF_TOKEN")
  if token_path.exists():
    return token_path.read_text().strip()

  raise SystemExit("ERROR: Need to configure your HF_TOKEN")

def _get_client():
  """Get or create OpenAI client for HuggingFace router"""
  global _client

  if _client is None:
    _client = OpenAI(
      base_url="https://router.huggingface.co/v1",
      api_key=_get_api_key(),
    )

  return _client

@log_timing
def _get_answer(question, model, system_prompt=None):
  messages = _build_messages(question, system_prompt)

  completion = _get_client().chat.completions.create(
    model=model,
    messages=messages,
  )
  return completion.choices[0].message.content

@log_timing
def _stream_answer(question, model, system_prompt=None):
  messages = _build_messages(question, system_prompt)

  stream = _get_client().chat.completions.create(
    model=model,
    messages=messages,
    stream=True,
  )

  full_response = ""
  with Live(console=console, refresh_per_second=10) as live:
    for chunk in stream:
      if chunk.choices and chunk.choices[0].delta.content:
        full_response += chunk.choices[0].delta.content
        live.update(Markdown(full_response))

  return full_response

def _format_model_with_provider(model, provider):
  """Format model string with provider suffix for HuggingFace router"""
  if ':' in model:
    # Already has provider suffix
    return model
  return f"{model}:{provider}"

if __name__ == "__main__":
  result = parse_arguments()

  # Check if we're in list-models mode
  if len(result) == 4 and result[3] is True:  # list-models mode
    list_models()
  else:
    question, model, system_prompt = result[0], result[1], result[2]
    prompt(question, model, system_prompt)
