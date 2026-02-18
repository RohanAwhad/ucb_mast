"""Demo of mast-evaluator library with model selection."""

import argparse
import json
from pathlib import Path

from mast import evaluate, ModelName

parser = argparse.ArgumentParser(description="Run MAST evaluator on a transcript.")
parser.add_argument(
    "transcript", type=Path, help="Path to transcript file (.txt or .json)"
)
parser.add_argument(
    "--model",
    type=str,
    choices=["openai", "vertex"],
    default="openai",
    help="Model to use (default: openai)",
)
args = parser.parse_args()

model: ModelName = (
    "vertex/claude-opus-4-6" if args.model == "vertex" else "openai/gpt-5.1"
)

# Load trace
path: Path = args.transcript
if path.suffix == ".json":
    data = json.loads(path.read_text())
    trace = json.dumps(data.get("trajectory", data), indent=2)
else:
    trace = path.read_text()

print("=" * 70)
print(f"MAST Evaluator Demo (model: {model})")
print("=" * 70)

# Evaluate
results = evaluate([trace], model_name=model)
result = results[0]

print("\n--- A. Summary ---")
print(result.summary)

print("\n--- B. Task Completed ---")
print(result.task_completed)

print("\n--- C. Detected Failure Modes (with definitions) ---")
for fm in result.failure_modes.get_detected():
    print(f"\n  [{fm['code']}] {fm['name']}")
    print(f"      {fm['definition']}")

print("\n" + "=" * 70)
