from typing import Any, Mapping

from .contracts import (
    DEFAULT_EVIDENCE_REASON,
    FailureModesPayload,
    FAILURE_MODE_CODES,
    FAILURE_MODE_FIELD_MAP,
    StructuredResponsePayload,
)
from .models import EvaluationResult, FailureModes


def _normalize_evidence_map(raw_evidence: dict[str, Any]) -> dict[str, str]:
    cleaned_map: dict[str, str] = {}
    for raw_id, raw_reason in raw_evidence.items():
        evidence_id = str(raw_id).strip()
        if not evidence_id:
            continue
        if evidence_id.lower() in {"none", "null", "n/a"}:
            continue

        reason = str(raw_reason).strip() if raw_reason is not None else ""
        cleaned_map[evidence_id] = reason or DEFAULT_EVIDENCE_REASON

    return cleaned_map


def _require_string(
    payload: Mapping[str, Any],
    key: str,
    error_key: str | None = None,
) -> str:
    value = payload.get(key)
    label = error_key or key
    if not isinstance(value, str):
        raise ValueError(f"Structured response field {label} must be a string")
    return value.strip()


def _require_bool(
    payload: Mapping[str, Any],
    key: str,
    error_key: str | None = None,
) -> bool:
    value = payload.get(key)
    label = error_key or key
    if not isinstance(value, bool):
        raise ValueError(f"Structured response field {label} must be a bool")
    return value


def _parse_failure_mode(
    failure_modes_payload: FailureModesPayload,
    code: str,
) -> tuple[bool, dict[str, str]]:
    mode_payload = failure_modes_payload.get(code)
    if not isinstance(mode_payload, dict):
        raise ValueError(f"Structured response missing mode payload for {code}")

    present = _require_bool(
        mode_payload,
        "present",
        f"failure_modes.{code}.present",
    )

    raw_evidence = mode_payload.get("evidence")
    if not isinstance(raw_evidence, dict):
        raise ValueError(
            f"Structured response field failure_modes.{code}.evidence must be an object"
        )

    evidence = _normalize_evidence_map(raw_evidence) if present else {}
    return present, evidence


def parse_structured_response(
    payload: StructuredResponsePayload | dict[str, Any],
    raw_response: str,
) -> EvaluationResult:
    summary = _require_string(payload, "summary")
    task_completed = _require_bool(payload, "task_completed")

    failure_modes_payload = payload.get("failure_modes")
    if not isinstance(failure_modes_payload, dict):
        raise ValueError("Structured response missing failure_modes object")

    failure_modes_payload_typed = failure_modes_payload

    mode_presence: dict[str, bool] = {}
    mode_evidence: dict[str, dict[str, str]] = {}

    for code in FAILURE_MODE_CODES:
        present, evidence = _parse_failure_mode(failure_modes_payload_typed, code)
        mode_presence[code] = present
        mode_evidence[code] = evidence

    failure_modes = FailureModes(
        **{
            field_name: mode_presence[code]
            for code, field_name in FAILURE_MODE_FIELD_MAP.items()
        }
    )

    return EvaluationResult(
        summary=summary,
        task_completed=task_completed,
        failure_modes=failure_modes,
        raw_response=raw_response,
        failure_mode_evidence=mode_evidence,
    )
