# AGENTS.md for HuggingChat Repository

## Build, Lint, and Test Commands

### Installation and Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Unix/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Windows
setx HF_TOKEN "your_token_value"
setx HF_MODEL "default_model_name"
# Unix/macOS
export HF_TOKEN="your_token_value"
export HF_MODEL="default_model_name"
```

### Running the Application
```bash
# Basic usage
python chat.py "What is the capital of South Sudan?"

# With specific model
python chat.py "What is the capital of South Sudan?" --model "deepseek-ai/DeepSeek-V3.2"

# Streaming output (default)
python chat.py --question "What is Python?" --model "Qwen/Qwen2.5-72B-Instruct"

# Non-streaming with markdown rendering
python chat.py --question "What is Python?" --no-stream

# CLI guru mode
python chat.py "How do I configure zsh?" --cli
```

## Code Style Guidelines

### Python Version
- Target Python 3.7+ (uses f-strings, type hints, and modern syntax)
- Use `#!/usr/bin/env python3` shebang

### Import Organization
Imports should be grouped in this order:
1. Standard library imports
2. Third-party imports (OpenAI, rich, etc.)
3. Local application imports

```python
import os
import sys
from pathlib import Path

from openai import OpenAI, APIError
from rich.console import Console
from rich.markdown import Markdown

from arguments import parse_arguments
from console import console
from utilities import log_timing
```

### Function and Variable Naming
- Use `snake_case` for functions and variables
- Use `PascalCase` for class names (if any classes are added)
- Use `UPPER_CASE` for constants
- Use descriptive, meaningful names

```python
def get_default_model():
  """Get the current default model"""
  return os.getenv("HF_MODEL", MODEL_DEFAULT)

MODEL_DEFAULT = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
```

### Type Hints
Always include type hints for function parameters and return values

```python
def prompt(question: str, model: str = None, stream: bool = True, 
           system_prompt: str = None, provider: str = None) -> None:
  """Process a question through the specified model"""
```

### Docstrings
Use triple-quoted docstrings with descriptive text

```python
def _load_models_csv(csv_path: str = "models.csv") -> List[Dict]:
  """Load models from CSV file and return as list of dictionaries"""
```

### Error Handling
Use `try/except` blocks with specific exception types

```python
try:
  # Code that might raise an exception
  answer = _get_answer(question, model, system_prompt)
except APIError as e:
  console.print(f"[red]API Error: {e.message}[/red]")
  return
except SystemExit as e:
  raise
```

### Console Output
Use `rich` console for all user-facing output
- Use color codes for emphasis: `[yellow]`, `[red]`, `[green]`, `[cyan]`
- Use `console.print()` for all output
- Use `console.print()` for tables and structured output

```python
console.print("[bold]Processing...[/bold]", style="yellow")
console.print(f"  [cyan]Question:[/cyan] {question}")

table = Table(title="Available Models")
table.add_column("Model", style="cyan", no_wrap=False)
```

### Code Organization
- Keep functions focused and small
- Use private functions (prefixed with `_`) for internal logic
- Use decorators for cross-cutting concerns (e.g., `@log_timing`)
- Place imports at the top of files
- Keep related functionality together
- Use two spaces for tabs

```python
def prompt(...):
  # Public API
  pass

def _load_models_csv(...):
  # Internal helper
  pass

def _get_cheapest_provider(...):
  # Internal helper
  pass
```

### Command-Line Interface
- Use `argparse` for argument parsing
- Provide clear help messages with examples
- Support both flags and positional arguments
- Use `.rstrip()` for docstrings to avoid trailing whitespace

```python
parser = argparse.ArgumentParser(
  description="Process a question with an optional model parameter",
  formatter_class=argparse.RawDescriptionHelpFormatter,
  epilog="""Examples:
%(prog)s "why is the sky blue?"
%(prog)s --question "why is the sky blue?"
  """
)
```

### String Formatting
- Use f-strings for variable interpolation
- Use `.rstrip()` on strings to remove trailing whitespace
- Use `.strip()` on user input

```python
question = question.strip()
console.print(f"Question: {question}")
```

### Conditional Logic
- Use `if/elif/else` for branching
- Use `isinstance()` for type checking
- Use `os.getenv()` for environment variable access

```python
if provider:
  model_with_provider = _format_model_with_provider(model, provider)
else:
  model_with_provider = model
```

### Logging and Timing
- Use `log_timing` decorator for performance monitoring
- Use `monotonic()` from `time` module for timing

```python
@log_timing
def _get_answer(question, model, system_prompt=None):
  # Implementation
```

## Environment Configuration

### Required Environment Variables
- `HF_TOKEN`: HuggingFace API token for model access
- `HF_MODEL`: Default model to use (optional, defaults to `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`)
- `HF_PROVIDER`: Default provider for the model (optional)

### Optional Environment Variables
- `.HF_TOKEN` file: Alternative to environment variable for HF_TOKEN

## Additional Notes

- The `models.csv` file contains model information including pricing, context length, and provider details
- Uses OpenAI-compatible API format through HuggingFace router
- Models can be filtered by provider for cost optimization
