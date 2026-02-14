# MAST Evaluator

A Python library for evaluating multi-agent system traces using the [MAST (Multi-Agent Systems Failure Taxonomy)](https://github.com/multi-agent-systems-failure-taxonomy/MAST) framework.

## What is MAST?

MAST is a taxonomy of failure modes and inefficiencies in multi-agent systems, developed by researchers at UC Berkeley. It defines 14 failure modes across 3 categories:

**Category 1: Agent-Level Failures**
- 1.1 Disobey Task Specification
- 1.2 Disobey Role Specification
- 1.3 Step Repetition
- 1.4 Loss of Conversation History
- 1.5 Unaware of Termination Conditions

**Category 2: Inter-Agent Failures**
- 2.1 Conversation Reset
- 2.2 Fail to Ask for Clarification
- 2.3 Task Derailment
- 2.4 Information Withholding
- 2.5 Ignored Other Agent's Input
- 2.6 Action-Reasoning Mismatch

**Category 3: System-Level Failures**
- 3.1 Premature Termination
- 3.2 No or Incorrect Verification
- 3.3 Weak Verification

## Installation

```bash
pip install -e .
# or with uv
uv pip install -e .
```

## Usage

```python
from mast import evaluate, FailureModes

# Your trace as a string (JSON, text, etc.)
trace = """[agent conversation trace here]"""

# Evaluate using OpenAI GPT-5.1 (default)
results = evaluate([trace], model_name="openai/gpt-5.1")

# Or use Anthropic Claude via Vertex AI
results = evaluate([trace], model_name="vertex/claude-opus-4-5")

# Access results
result = results[0]
print(result.summary)        # Freeform summary of issues
print(result.task_completed) # True/False

# Get detected failure modes with definitions
for fm in result.failure_modes.get_detected():
    print(f"[{fm['code']}] {fm['name']}")
    print(f"    {fm['definition']}")

# Look up any failure mode definition
FailureModes.get_definition("1.3")
# {'name': 'Step Repetition', 'definition': 'Agent unnecessarily repeats...'}
```

## Environment Variables

**For OpenAI:**
```bash
export OPENAI_API_KEY=your-api-key
```

**For Anthropic Vertex AI:**
```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
# Also requires: gcloud auth application-default login
```

## API Reference

### `evaluate(traces, *, model_name, api_key, project_id, region, max_trace_length)`

Evaluate multi-agent system traces for failure modes.

**Parameters:**
- `traces`: List of trace strings to evaluate
- `model_name`: `"openai/gpt-5.1"` or `"vertex/claude-opus-4-5"` (default: openai)
- `api_key`: OpenAI API key (optional, defaults to env var)
- `project_id`: Google Cloud project ID for Vertex (optional, defaults to env var)
- `region`: GCP region for Vertex AI (default: `"us-east5"`)
- `max_trace_length`: Maximum trace length before truncation (default: 1048570)

**Returns:** `list[EvaluationResult]`

### `EvaluationResult`

```python
@dataclass
class EvaluationResult:
    summary: str           # Freeform analysis summary
    task_completed: bool   # Whether the task succeeded
    failure_modes: FailureModes  # Detected failure modes
    raw_response: str      # Raw LLM response
```

### `FailureModes`

```python
@dataclass
class FailureModes:
    # 14 boolean fields for each failure mode
    disobey_task_specification: bool  # 1.1
    # ... etc

    def to_dict() -> dict[str, bool]  # {"1.1": True, "1.2": False, ...}
    def get_detected() -> list[dict]  # List of detected modes with definitions
    @staticmethod
    def get_definition(code: str) -> dict  # Look up any mode's definition
```

## Running Tests

```bash
# Unit tests only
uv run pytest tests/ -m "not integration"

# All tests (requires API keys)
uv run pytest tests/
```

## License

MIT
