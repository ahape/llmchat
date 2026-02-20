# Setup

Step 1: Set the API key for your chosen router as an env var or dotfile:

- **Google AI**: `GOOGLE_API_KEY` env var or `.GOOGLE_API_KEY` file
- **OpenRouter**: `OPENROUTER_API_KEY` env var or `.OPENROUTER_API_KEY` file

Step 2: Create and activate the virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate   # Unix/macOS
# .venv\Scripts\activate    # Windows
```

Step 3: Install the packages

```sh
pip install -r requirements.txt
```

# Usage

```sh
python3 chat.py "What is the capital of South Sudan?"
python3 chat.py "What is the capital of South Sudan?" --model "gemini-2.0-flash"
python3 chat.py --switch-router     # interactively change API router
python3 chat.py --switch-model      # interactively change default model
python3 chat.py --list-models       # show available models
python3 chat.py --fast "Need a quick answer"   # shortcut for OpenRouter + Gemini Flash Lite
```

# Other stuff

## Gemini 3 Pro's recs for model usage (2026-02-05)

### 1. The Best "Bang for Your Buck" (General Intelligence)
If you need a smart model that is extremely cheap to run, these are the standouts:

*   **openai/gpt-oss-120b (Novita):**
  *   **Why:** At **$0.05/$0.25**, this is incredibly cheap for a 120B parameter model. If this "open source GPT" performs anywhere near GPT-4 class, it is the best value on the list.
*   **meta-llama/Llama-4-Scout-17B-16E-Instruct (nscale):**
  *   **Why:** The "Scout" series usually punches above its weight. With a massive **890k context window** and pricing of **$0.09/$0.29**, this is perfect for summarizing large documents or analyzing books on a budget.
*   **Qwen/Qwen3-235B-A22B-Instruct-2507 (Novita):**
  *   **Why:** A massive 235B model for only **$0.09** input? This is likely the smartest model for the lowest price in the high-intelligence category.

### 2. Best for Coding
*   **Qwen/Qwen2.5-Coder-32B-Instruct (nscale):**
  *   **Why:** The Qwen Coder series is currently state-of-the-art for open weights. The 32B version is the "sweet spot" between speed and smarts. At **$0.06/$0.20**, it is a steal.
*   **Qwen/Qwen3-Coder-480B-A35B-Instruct (Novita):**
  *   **Why:** If you have a very complex architectural problem and money is less of an object, this 480B parameter monster is the heavy hitter.

### 3. Best for Complex Reasoning & Math
*   **deepseek-ai/DeepSeek-R1 (Novita):**
  *   **Why:** The "R1" series is optimized for "Chain of Thought" reasoning. Use this for math, logic puzzles, or complex agentic workflows.
*   **Qwen/Qwen3-Next-80B-A3B-Thinking (Hyperbolic):**
  *   **Why:** Tagged as "Cheapest Fastest" in the reasoning category (**$0.30/$0.30**).

### 4. Fastest / Lowest Latency (Chat & Simple Tasks)
*   **meta-llama/Llama-3.2-3B-Instruct (Together):**
  *   **Why:** Tagged "Cheapest Fastest." At 3B parameters, this will reply instantly. Use this for simple classification, extraction, or chat bots.
*   **zai-org/GLM-4.7-Flash (Novita):**
  *   **Why:** Flash models are optimized for speed. At **$0.07/$0.40** with a 200k context, it's a great fast reader.

### 5. Best for Vision (Images/Multimodal)
*   **Qwen/Qwen2.5-VL-72B-Instruct (Hyperbolic):**
  *   **Why:** Qwen's Vision Language (VL) models are excellent. This 72B model is tagged "Cheapest Fastest" (**$0.60/$0.60**) and is powerful enough to handle complex OCR and chart analysis.

### Summary Recommendation
*   **Start with:** **`Qwen/Qwen2.5-72B-Instruct`** (or the Qwen3 equivalent). It is generally the best all-rounder for logic, prose, and instruction following.
*   **For pure economy:** **`meta-llama/Llama-3.1-8B-Instruct`** ($0.02 input).
*   **For Massive Context:** **`MiniMax-M1-80k`** (1 Million context) or **`Llama-4-Scout`** (890k context).
