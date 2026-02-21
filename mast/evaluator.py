from pathlib import Path
from typing import Any, Literal

from .contracts import FAILURE_MODE_CODES
from .models import EvaluationResult
from .providers import call_anthropic_vertex, call_openai
from .structured_output import parse_structured_response

_TAXONOMY_DIR = Path(__file__).parent.parent / "taxonomy_definitions_examples"

ModelName = Literal["vertex/claude-opus-4-6", "openai/gpt-5.2"]


def _load_definitions() -> str:
    return (_TAXONOMY_DIR / "definitions.txt").read_text()


def _load_examples() -> str:
    return (_TAXONOMY_DIR / "examples.txt").read_text()


def _build_prompt(trace: str, definitions: str, examples: str) -> str:
    return f"""You are evaluating a multi-agent system trace using the MAST taxonomy.
Return only structured output that matches the provided schema.

Evaluation rules:
- Use the 14 failure mode codes exactly: {", ".join(FAILURE_MODE_CODES)}.
- Set `present=true` only when you can point to concrete evidence in the trace.
- Use `evidence` as an object mapping each transcript ID to a short explanation.
- Transcript IDs look like: m_0007, tc_0002, tr_0002.
- If a failure mode is not present, set `evidence` to {{}}.
- `task_completed=true` only if the main user task is completed correctly.
- Keep `summary` to one sentence.

Trace:
{trace}

Definitions:
{definitions}

Examples:
{examples}
"""


def _call_openai(
    prompt: str,
    api_key: str | None = None,
) -> tuple[dict[str, Any], str]:
    return call_openai(prompt=prompt, api_key=api_key)


def _call_anthropic_vertex(
    prompt: str,
    project_id: str | None = None,
    region: str = "us-east5",
) -> tuple[dict[str, Any], str]:
    return call_anthropic_vertex(prompt=prompt, project_id=project_id, region=region)


def evaluate(
    traces: list[str],
    *,
    model_name: ModelName = "openai/gpt-5.2",
    api_key: str | None = None,
    project_id: str | None = None,
    region: str = "us-east5",
    max_trace_length: int = 1048570,
) -> list[EvaluationResult]:
    """
    Evaluate multi-agent system traces for failure modes.

    Args:
        traces: List of trace strings to evaluate
        model_name: Model to use - "openai/gpt-5.2" or "vertex/claude-opus-4-6"
        api_key: OpenAI API key (for openai model, defaults to OPENAI_API_KEY env var)
        project_id: Google Cloud project ID (for vertex model, defaults to GOOGLE_CLOUD_PROJECT env var)
        region: Google Cloud region for Vertex AI (default: us-east5)
        max_trace_length: Maximum trace length before truncation

    Returns:
        List of EvaluationResult objects
    """
    definitions = _load_definitions()
    examples = _load_examples()

    results: list[EvaluationResult] = []
    for trace in traces:
        if len(trace) + len(examples) > max_trace_length:
            trace = trace[: max_trace_length - len(examples)]

        prompt = _build_prompt(trace, definitions, examples)

        if model_name == "openai/gpt-5.2":
            structured_payload, raw_response = _call_openai(prompt, api_key)
        elif model_name == "vertex/claude-opus-4-6":
            structured_payload, raw_response = _call_anthropic_vertex(
                prompt,
                project_id,
                region,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        results.append(parse_structured_response(structured_payload, raw_response))

    return results
