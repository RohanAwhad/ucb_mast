import json
import os
import re
from pathlib import Path
from typing import Any, Literal, cast

from anthropic import AnthropicVertex
from openai import OpenAI

from .models import EvaluationResult, FailureModes

_TAXONOMY_DIR = Path(__file__).parent.parent / "taxonomy_definitions_examples"

ModelName = Literal["vertex/claude-opus-4-6", "openai/gpt-5.2"]

_FAILURE_MODE_CODES = (
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "2.6",
    "3.1",
    "3.2",
    "3.3",
)
_ANTHROPIC_TOOL_NAME = "submit_mast_evaluation"
_DEFAULT_EVIDENCE_REASON = "Judge cited this transcript ID as evidence."
_LEGACY_EVIDENCE_REASON = "Evidence cited in plain-text response."


def _build_structured_response_schema() -> dict[str, Any]:
    failure_mode_properties = {
        code: {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "evidence": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["present", "evidence"],
            "additionalProperties": False,
        }
        for code in _FAILURE_MODE_CODES
    }

    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "task_completed": {"type": "boolean"},
            "failure_modes": {
                "type": "object",
                "properties": failure_mode_properties,
                "required": list(_FAILURE_MODE_CODES),
                "additionalProperties": False,
            },
        },
        "required": ["summary", "task_completed", "failure_modes"],
        "additionalProperties": False,
    }


_STRUCTURED_RESPONSE_SCHEMA = _build_structured_response_schema()


def _load_definitions() -> str:
    return (_TAXONOMY_DIR / "definitions.txt").read_text()


def _load_examples() -> str:
    return (_TAXONOMY_DIR / "examples.txt").read_text()


def _build_prompt(trace: str, definitions: str, examples: str) -> str:
    return f"""You are evaluating a multi-agent system trace using the MAST taxonomy.
Return only structured output that matches the provided schema.

Evaluation rules:
- Use the 14 failure mode codes exactly: {", ".join(_FAILURE_MODE_CODES)}.
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


def _parse_yes_no(text: str, mode: str) -> bool:
    """Parse yes/no for a specific failure mode from a plain-text response."""
    mode_escaped = mode.replace(".", r"\.")
    patterns = [
        rf"^\s*{mode_escaped}\b[^\n]*?:\s*(yes|no)\s*$",
        rf"^\s*{mode_escaped}\s*[:\-]?\s*(yes|no)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).lower() == "yes"
    return False


def _parse_id_list(raw_ids: str) -> list[str]:
    parts = [part.strip().strip("'\"`") for part in raw_ids.split(",")]
    filtered = [
        part for part in parts if part and part.lower() not in {"none", "null", "n/a"}
    ]
    deduped: list[str] = []
    for item in filtered:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _ids_to_evidence_map(ids: list[str], reason: str) -> dict[str, str]:
    return {item: reason for item in ids}


def _parse_evidence_ids(text: str, mode: str) -> dict[str, str]:
    mode_escaped = mode.replace(".", r"\.")
    patterns = [
        rf"^\s*(?:\[\s*{mode_escaped}\s*\]|{mode_escaped})[^\n]*?evidence(?:_ids)?\s*[:=]\s*\[([^\]]*)\]",
        rf"^\s*(?:\[\s*{mode_escaped}\s*\]|{mode_escaped})[^\n]*?:?\s*\[([^\]]*)\]",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        for match in reversed(matches):
            ids = _parse_id_list(match.group(1).strip())
            return _ids_to_evidence_map(ids, _LEGACY_EVIDENCE_REASON)
    return {}


def _parse_response(response: str) -> EvaluationResult:
    """Legacy parser for plain-text model output."""
    cleaned = response.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]

    summary_match = re.search(
        r"A\.\s*(.+?)(?=B\.|$)", cleaned, re.DOTALL | re.IGNORECASE
    )
    summary = summary_match.group(1).strip() if summary_match else ""

    task_match = re.search(r"B\.[^\n]*?(yes|no)\b", cleaned, re.IGNORECASE)
    task_completed = task_match.group(1).lower() == "yes" if task_match else False

    failure_modes = FailureModes(
        disobey_task_specification=_parse_yes_no(cleaned, "1.1"),
        disobey_role_specification=_parse_yes_no(cleaned, "1.2"),
        step_repetition=_parse_yes_no(cleaned, "1.3"),
        loss_of_conversation_history=_parse_yes_no(cleaned, "1.4"),
        unaware_of_termination_conditions=_parse_yes_no(cleaned, "1.5"),
        conversation_reset=_parse_yes_no(cleaned, "2.1"),
        fail_to_ask_for_clarification=_parse_yes_no(cleaned, "2.2"),
        task_derailment=_parse_yes_no(cleaned, "2.3"),
        information_withholding=_parse_yes_no(cleaned, "2.4"),
        ignored_other_agent_input=_parse_yes_no(cleaned, "2.5"),
        action_reasoning_mismatch=_parse_yes_no(cleaned, "2.6"),
        premature_termination=_parse_yes_no(cleaned, "3.1"),
        no_or_incorrect_verification=_parse_yes_no(cleaned, "3.2"),
        weak_verification=_parse_yes_no(cleaned, "3.3"),
    )

    failure_mode_evidence: dict[str, dict[str, str]] = {
        code: _parse_evidence_ids(cleaned, code) for code in _FAILURE_MODE_CODES
    }

    return EvaluationResult(
        summary=summary,
        task_completed=task_completed,
        failure_modes=failure_modes,
        raw_response=response,
        failure_mode_evidence=failure_mode_evidence,
    )


def _normalize_evidence_map(raw_evidence: Any) -> dict[str, str]:
    if isinstance(raw_evidence, dict):
        cleaned_map: dict[str, str] = {}
        for raw_id, raw_reason in raw_evidence.items():
            evidence_id = str(raw_id).strip()
            if not evidence_id:
                continue
            if evidence_id.lower() in {"none", "null", "n/a"}:
                continue

            reason = str(raw_reason).strip() if raw_reason is not None else ""
            cleaned_map[evidence_id] = reason or _DEFAULT_EVIDENCE_REASON
        return cleaned_map

    if isinstance(raw_evidence, list):
        cleaned_ids: list[str] = []
        for item in raw_evidence:
            item_str = item if isinstance(item, str) else str(item)
            normalized = item_str.strip()
            if not normalized:
                continue
            if normalized.lower() in {"none", "null", "n/a"}:
                continue
            if normalized not in cleaned_ids:
                cleaned_ids.append(normalized)
        return _ids_to_evidence_map(cleaned_ids, _DEFAULT_EVIDENCE_REASON)

    return {}


def _parse_structured_response(
    payload: dict[str, Any],
    raw_response: str,
) -> EvaluationResult:
    summary_raw = payload.get("summary", "")
    summary = summary_raw if isinstance(summary_raw, str) else str(summary_raw)

    task_completed_raw = payload.get("task_completed", False)
    task_completed = (
        task_completed_raw
        if isinstance(task_completed_raw, bool)
        else bool(task_completed_raw)
    )

    failure_modes_payload = payload.get("failure_modes")
    if not isinstance(failure_modes_payload, dict):
        raise ValueError("Structured response missing failure_modes object")

    mode_presence: dict[str, bool] = {}
    mode_evidence: dict[str, dict[str, str]] = {}

    for code in _FAILURE_MODE_CODES:
        mode_payload = failure_modes_payload.get(code)
        if not isinstance(mode_payload, dict):
            raise ValueError(f"Structured response missing mode payload for {code}")

        present_raw = mode_payload.get("present", False)
        present = present_raw if isinstance(present_raw, bool) else bool(present_raw)
        evidence_map = _normalize_evidence_map(
            mode_payload.get("evidence", mode_payload.get("evidence_ids", []))
        )
        mode_presence[code] = present
        mode_evidence[code] = evidence_map if present else {}

    failure_modes = FailureModes(
        disobey_task_specification=mode_presence["1.1"],
        disobey_role_specification=mode_presence["1.2"],
        step_repetition=mode_presence["1.3"],
        loss_of_conversation_history=mode_presence["1.4"],
        unaware_of_termination_conditions=mode_presence["1.5"],
        conversation_reset=mode_presence["2.1"],
        fail_to_ask_for_clarification=mode_presence["2.2"],
        task_derailment=mode_presence["2.3"],
        information_withholding=mode_presence["2.4"],
        ignored_other_agent_input=mode_presence["2.5"],
        action_reasoning_mismatch=mode_presence["2.6"],
        premature_termination=mode_presence["3.1"],
        no_or_incorrect_verification=mode_presence["3.2"],
        weak_verification=mode_presence["3.3"],
    )

    return EvaluationResult(
        summary=summary.strip(),
        task_completed=task_completed,
        failure_modes=failure_modes,
        raw_response=raw_response,
        failure_mode_evidence=mode_evidence,
    )


def _call_openai(
    prompt: str,
    api_key: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Call OpenAI API with structured JSON schema output."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        reasoning={"effort": "high"},
        text={
            "format": {
                "type": "json_schema",
                "name": "mast_evaluation",
                "schema": _STRUCTURED_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    )

    response_any: Any = response
    output_text = getattr(response_any, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("OpenAI response missing structured output text")

    payload = json.loads(output_text)
    if not isinstance(payload, dict):
        raise ValueError("OpenAI structured output is not a JSON object")

    return payload, output_text


def _call_anthropic_vertex(
    prompt: str,
    project_id: str | None = None,
    region: str = "us-east5",
) -> tuple[dict[str, Any], str]:
    """Call Anthropic Claude via Vertex AI with a forced tool schema."""
    resolved_project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if resolved_project_id is None:
        raise ValueError("GOOGLE_CLOUD_PROJECT must be set for vertex model")

    client = AnthropicVertex(
        project_id=cast(str, resolved_project_id),
        region=region,
    )

    response = client.messages.create(
        model="claude-opus-4-6@default",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "name": _ANTHROPIC_TOOL_NAME,
                "description": "Submit MAST evaluation in structured JSON format.",
                "input_schema": _STRUCTURED_RESPONSE_SCHEMA,
            }
        ],
        tool_choice={"type": "tool", "name": _ANTHROPIC_TOOL_NAME},
    )

    response_any: Any = response
    content_blocks = getattr(response_any, "content", None)
    if not isinstance(content_blocks, list):
        raise ValueError("Anthropic response missing content blocks")

    for block in content_blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _ANTHROPIC_TOOL_NAME:
            continue
        tool_input = getattr(block, "input", None)
        if isinstance(tool_input, dict):
            return tool_input, json.dumps(tool_input)

    raise ValueError("Anthropic response missing structured tool output")


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

        results.append(_parse_structured_response(structured_payload, raw_response))

    return results
