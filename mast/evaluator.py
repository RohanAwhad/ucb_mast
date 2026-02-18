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


def _load_definitions() -> str:
    return (_TAXONOMY_DIR / "definitions.txt").read_text()


def _load_examples() -> str:
    return (_TAXONOMY_DIR / "examples.txt").read_text()


def _build_prompt(trace: str, definitions: str, examples: str) -> str:
    return f"""Below I will provide a multi-agent system trace.
Analyze the system behavior and detect failure modes from the taxonomy.
Only mark a failure mode if you can point to concrete evidence in the trace.

Important: transcript lines may include IDs like [m_0004], [tc_0002], [tr_0002].
When you mark a failure mode as yes, cite the relevant IDs.

Return your answer strictly between @@ and @@ with this exact structure:
@@
A. <one sentence summary>
B. <yes or no>
C.
1.1 Disobey Task Specification: <yes or no>
1.2 Disobey Role Specification: <yes or no>
1.3 Step Repetition: <yes or no>
1.4 Loss of Conversation History: <yes or no>
1.5 Unaware of Termination Conditions: <yes or no>
2.1 Conversation Reset: <yes or no>
2.2 Fail to Ask for Clarification: <yes or no>
2.3 Task Derailment: <yes or no>
2.4 Information Withholding: <yes or no>
2.5 Ignored Other Agent's Input: <yes or no>
2.6 Action-Reasoning Mismatch: <yes or no>
3.1 Premature Termination: <yes or no>
3.2 No or Incorrect Verification: <yes or no>
3.3 Weak Verification: <yes or no>
D.
1.1 Disobey Task Specification: [<id1>, <id2>]
1.2 Disobey Role Specification: [<id1>, <id2>]
1.3 Step Repetition: [<id1>, <id2>]
1.4 Loss of Conversation History: [<id1>, <id2>]
1.5 Unaware of Termination Conditions: [<id1>, <id2>]
2.1 Conversation Reset: [<id1>, <id2>]
2.2 Fail to Ask for Clarification: [<id1>, <id2>]
2.3 Task Derailment: [<id1>, <id2>]
2.4 Information Withholding: [<id1>, <id2>]
2.5 Ignored Other Agent's Input: [<id1>, <id2>]
2.6 Action-Reasoning Mismatch: [<id1>, <id2>]
3.1 Premature Termination: [<id1>, <id2>]
3.2 No or Incorrect Verification: [<id1>, <id2>]
3.3 Weak Verification: [<id1>, <id2>]
@@

Rules for section D:
- Always provide all 14 lines.
- Use [] when there is no evidence.
- IDs must be copied exactly from the trace.

Example answer:
@@
A. The task is not completed due to derailment and poor verification.
B. no
C.
1.1 no
1.2 no
1.3 no
1.4 no
1.5 no
2.1 no
2.2 no
2.3 yes
2.4 no
2.5 no
2.6 yes
3.1 no
3.2 yes
3.3 no
D.
1.1: []
1.2: []
1.3: []
1.4: []
1.5: []
2.1: []
2.2: []
2.3: [m_0011, m_0012]
2.4: []
2.5: []
2.6: [m_0014]
3.1: []
3.2: [tr_0003]
3.3: []
@@

Here is the trace:
{trace}

Also, here are the explanations (definitions) of the failure modes and inefficiencies:
{definitions}

Here are some examples of the failure modes and inefficiencies:
{examples}
"""


def _parse_yes_no(text: str, mode: str) -> bool:
    """Parse yes/no for a specific failure mode from response."""
    # Escape the dot in mode number for regex
    mode_escaped = mode.replace(".", r"\.")
    # Try multiple patterns to handle different response formats
    patterns = [
        # Format: "1.1 yes" or "1.1  yes" (just number and yes/no)
        rf"{mode_escaped}\s+(yes|no)",
        # Format: "1.1 Disobey Task Specification: yes"
        rf"{mode_escaped}[^0-9\n]*?:\s*(yes|no)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
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


def _parse_evidence_ids(text: str, mode: str) -> list[str]:
    mode_escaped = mode.replace(".", r"\.")
    patterns = [
        rf"^\s*(?:\[\s*{mode_escaped}\s*\]|{mode_escaped})[^\n]*?evidence(?:_ids)?\s*[:=]\s*\[([^\]]*)\]",
        rf"^\s*(?:\[\s*{mode_escaped}\s*\]|{mode_escaped})[^\n]*?:?\s*\[([^\]]*)\]",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        for match in reversed(matches):
            parsed = _parse_id_list(match.group(1).strip())
            return parsed
    return []


def _parse_response(response: str) -> EvaluationResult:
    """Parse the LLM response into an EvaluationResult."""
    cleaned = response.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]

    # Parse A: summary
    summary_match = re.search(
        r"A\.\s*(.+?)(?=B\.|$)", cleaned, re.DOTALL | re.IGNORECASE
    )
    summary = summary_match.group(1).strip() if summary_match else ""

    # Parse B: task completed
    task_match = re.search(r"B\.[^\n]*?(yes|no)\b", cleaned, re.IGNORECASE)
    task_completed = task_match.group(1).lower() == "yes" if task_match else False

    # Parse C: failure modes
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

    failure_mode_evidence = {
        code: _parse_evidence_ids(cleaned, code) for code in _FAILURE_MODE_CODES
    }

    return EvaluationResult(
        summary=summary,
        task_completed=task_completed,
        failure_modes=failure_modes,
        raw_response=response,
        failure_mode_evidence=failure_mode_evidence,
    )


def _call_openai(prompt: str, api_key: str | None = None) -> str:
    """Call OpenAI API with GPT-5.2 and thinking medium."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        reasoning={"effort": "high"},
    )

    response_any: Any = response

    output_text = getattr(response_any, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    output_blocks: Any = getattr(response_any, "output", None)
    if isinstance(output_blocks, list):
        for block in output_blocks:
            content_items = getattr(block, "content", None)
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                text_value = getattr(content_item, "text", None)
                if isinstance(text_value, str):
                    return text_value
    return ""


def _call_anthropic_vertex(
    prompt: str,
    project_id: str | None = None,
    region: str = "us-east5",
) -> str:
    """Call Anthropic Claude via Vertex AI with extended thinking."""
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
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text from response
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


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

    results = []
    for trace in traces:
        # Truncate if needed
        if len(trace) + len(examples) > max_trace_length:
            trace = trace[: max_trace_length - len(examples)]

        prompt = _build_prompt(trace, definitions, examples)

        if model_name == "openai/gpt-5.2":
            response = _call_openai(prompt, api_key)
        elif model_name == "vertex/claude-opus-4-6":
            response = _call_anthropic_vertex(prompt, project_id, region)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        result = _parse_response(response)
        results.append(result)

    return results
