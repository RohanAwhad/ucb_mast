import os
import re
from pathlib import Path

from openai import OpenAI

from .models import EvaluationResult, FailureModes

_TAXONOMY_DIR = Path(__file__).parent.parent / "taxonomy_definitions_examples"


def _load_definitions() -> str:
    return (_TAXONOMY_DIR / "definitions.txt").read_text()


def _load_examples() -> str:
    return (_TAXONOMY_DIR / "examples.txt").read_text()


def _build_prompt(trace: str, definitions: str, examples: str) -> str:
    return (
        "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
        "In the traces, analyze the system behaviour."
        "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
        "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
        "Also tell me whether the task is successfully completed or not, as a binary yes or no."
        "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
        "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
        "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
        "*** begin of things you should answer *** @@"
        "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
        "B. Whether the task is successfully completed or not: <yes or no>"
        "C. Whether you encounter any of the failure modes or inefficiencies:"
        "1.1 Disobey Task Specification: <yes or no>"
        "1.2 Disobey Role Specification: <yes or no>"
        "1.3 Step Repetition: <yes or no>"
        "1.4 Loss of Conversation History: <yes or no>"
        "1.5 Unaware of Termination Conditions: <yes or no>"
        "2.1 Conversation Reset: <yes or no>"
        "2.2 Fail to Ask for Clarification: <yes or no>"
        "2.3 Task Derailment: <yes or no>"
        "2.4 Information Withholding: <yes or no>"
        "2.5 Ignored Other Agent's Input: <yes or no>"
        "2.6 Action-Reasoning Mismatch: <yes or no>"
        "3.1 Premature Termination: <yes or no>"
        "3.2 No or Incorrect Verification: <yes or no>"
        "3.3 Weak Verification: <yes or no>"
        "@@*** end of your answer ***"
        "An example answer is: \n"
        "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
        "B. no \n"
        "C. \n"
        "1.1 no \n"
        "1.2 no \n"
        "1.3 no \n"
        "1.4 no \n"
        "1.5 no \n"
        "2.1 no \n"
        "2.2 no \n"
        "2.3 yes \n"
        "2.4 no \n"
        "2.5 no \n"
        "2.6 yes \n"
        "3.1 no \n"
        "3.2 yes \n"
        "3.3 no \n"
        "Here is the trace: \n"
        f"{trace}"
        "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
        f"{definitions} \n"
        "Here are some examples of the failure modes and inefficiencies: \n"
        f"{examples}"
    )


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


def _parse_response(response: str) -> EvaluationResult:
    """Parse the LLM response into an EvaluationResult."""
    cleaned = response.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]

    # Parse A: summary
    summary_match = re.search(r"A\.\s*(.+?)(?=B\.|$)", cleaned, re.DOTALL | re.IGNORECASE)
    summary = summary_match.group(1).strip() if summary_match else ""

    # Parse B: task completed
    task_match = re.search(r"B\.\s*(yes|no)", cleaned, re.IGNORECASE)
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

    return EvaluationResult(
        summary=summary,
        task_completed=task_completed,
        failure_modes=failure_modes,
        raw_response=response,
    )


def _call_openai(prompt: str, api_key: str | None = None) -> str:
    """Call OpenAI API with GPT-5.1 and thinking medium."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-5.1",
        input=prompt,
        reasoning={"effort": "medium"},
    )
    # Extract text from response - handle both possible response structures
    if hasattr(response, "output_text"):
        return response.output_text
    elif hasattr(response, "output") and len(response.output) > 0:
        # New API structure returns output as list of content blocks
        for block in response.output:
            if hasattr(block, "content") and len(block.content) > 0:
                for content in block.content:
                    if hasattr(content, "text"):
                        return content.text
    return ""


def evaluate(
    traces: list[str],
    *,
    api_key: str | None = None,
    max_trace_length: int = 1048570,
) -> list[EvaluationResult]:
    """
    Evaluate multi-agent system traces for failure modes.

    Args:
        traces: List of trace strings to evaluate
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
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
        response = _call_openai(prompt, api_key)
        result = _parse_response(response)
        results.append(result)

    return results
