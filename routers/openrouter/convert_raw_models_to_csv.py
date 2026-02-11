import json
import csv
from pathlib import Path

SCHEMA = [
    "Model",
    "Provider",
    "Tags",
    "Input $/1M",
    "Output $/1M",
    "Context",
    "Latency(s)",
    "Throughput(t/s)",
    "Tools",
    "Structured",
]

def to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def dollars_per_million(raw_price):
    """
    raw_price is usually a string like "0.0000012" ($ per token, typically).
    We convert to $ per 1M tokens.
    """
    f = to_float(raw_price)
    return None if f is None else f * 1_000_000

def join_tags(tags):
    # CSV-friendly tag string; tweak to your preference
    return ";".join([t for t in tags if t])

def build_row(item: dict) -> dict:
    endpoint = item.get("endpoint") or {}

    model_slug = item.get("slug", "")
    provider = (
        endpoint.get("provider_display_name")
        or endpoint.get("provider_name")
        or (endpoint.get("provider_info") or {}).get("displayName")
        or (endpoint.get("provider_info") or {}).get("name")
        or ""
    )

    pricing = endpoint.get("pricing") or {}
    input_per_m = dollars_per_million(pricing.get("prompt"))
    output_per_m = dollars_per_million(pricing.get("completion"))

    context = item.get("context_length") or endpoint.get("context_length") or ""

    supported_params = endpoint.get("supported_parameters") or []
    supported_params_set = set(supported_params)

    tools_supported = any(p in supported_params_set for p in ("tools", "tool_choice"))
    structured_supported = any(p in supported_params_set for p in ("structured_outputs", "response_format"))

    # Build some reasonable tags (customize as you like)
    tags = []
    tags.append(item.get("group"))  # e.g., "Qwen"
    if item.get("supports_reasoning"):
        tags.append("reasoning")
    if item.get("hidden"):
        tags.append("hidden")
    # Modalities
    in_mods = item.get("input_modalities") or []
    out_mods = item.get("output_modalities") or []
    if in_mods:
        tags.append("in:" + ",".join(in_mods))
    if out_mods:
        tags.append("out:" + ",".join(out_mods))
    # Variant / quantization if present
    if endpoint.get("variant"):
        tags.append(f"variant:{endpoint['variant']}")
    if endpoint.get("quantization"):
        tags.append(f"quant:{endpoint['quantization']}")

    row = {
        "Model": model_slug,
        "Provider": provider,
        "Tags": join_tags(tags),
        "Input $/1M": "" if input_per_m is None else f"{input_per_m:.6f}",
        "Output $/1M": "" if output_per_m is None else f"{output_per_m:.6f}",
        "Context": context,
        "Latency(s)": "",          # not in your sample JSON
        "Throughput(t/s)": "",     # not in your sample JSON
        "Tools": "yes" if tools_supported else "no",
        "Structured": "yes" if structured_supported else "no",
    }
    return row

def json_to_csv(json_path: str, csv_path: str):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    items = data.get("data") or []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        for item in items:
            writer.writerow(build_row(item))

if __name__ == "__main__":
    json_to_csv("models.raw.json", "models.csv")
    print("Wrote models.csv")
