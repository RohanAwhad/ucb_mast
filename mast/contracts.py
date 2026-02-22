from typing import Any, TypeAlias, TypedDict

FAILURE_MODE_CODES = (
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

FAILURE_MODE_FIELD_MAP = {
    "1.1": "disobey_task_specification",
    "1.2": "disobey_role_specification",
    "1.3": "step_repetition",
    "1.4": "loss_of_conversation_history",
    "1.5": "unaware_of_termination_conditions",
    "2.1": "conversation_reset",
    "2.2": "fail_to_ask_for_clarification",
    "2.3": "task_derailment",
    "2.4": "information_withholding",
    "2.5": "ignored_other_agent_input",
    "2.6": "action_reasoning_mismatch",
    "3.1": "premature_termination",
    "3.2": "no_or_incorrect_verification",
    "3.3": "weak_verification",
}

DEFAULT_EVIDENCE_REASON = "Judge cited this transcript ID as evidence."


class FailureModePayload(TypedDict):
    present: bool
    evidence: dict[str, str]


FailureModesPayload: TypeAlias = dict[str, FailureModePayload]


class StructuredResponsePayload(TypedDict):
    summary: str
    task_completed: bool
    failure_modes: FailureModesPayload


def build_structured_response_schema() -> dict[str, Any]:
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
        for code in FAILURE_MODE_CODES
    }

    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "task_completed": {"type": "boolean"},
            "failure_modes": {
                "type": "object",
                "properties": failure_mode_properties,
                "required": list(FAILURE_MODE_CODES),
                "additionalProperties": False,
            },
        },
        "required": ["summary", "task_completed", "failure_modes"],
        "additionalProperties": False,
    }


STRUCTURED_RESPONSE_SCHEMA = build_structured_response_schema()
