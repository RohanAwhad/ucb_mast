import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from mast import EvaluationResult, FailureModes, ModelName, evaluate
from mast.contracts import (
    FAILURE_MODE_CODES,
    FAILURE_MODE_FIELD_MAP,
    STRUCTURED_RESPONSE_SCHEMA,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_trace() -> str:
    """Load sample trace from fixtures."""
    trace_path = FIXTURES_DIR / "sample_trace.json"
    data = json.loads(trace_path.read_text())
    # Convert trajectory to string format
    trajectory = data.get("trajectory", [])
    return json.dumps(trajectory, indent=2)


@pytest.fixture
def mock_structured_payload() -> dict[str, object]:
    """Mock structured output for schema-based model calls."""
    codes = [
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
    ]
    modes: dict[str, dict[str, object]] = {
        code: {"present": False, "evidence": {}} for code in codes
    }
    modes["1.3"] = {
        "present": True,
        "evidence": {
            "m_0007": "Agent repeats an already completed step.",
            "m_0010": "Repeated follow-up with no new progress.",
        },
    }
    modes["1.5"] = {
        "present": True,
        "evidence": {
            "m_0010": "Agent continues despite missing stopping condition.",
            "m_0012": "Conversation should have terminated earlier.",
        },
    }
    modes["2.5"] = {
        "present": True,
        "evidence": {
            "m_0008": "Agent ignores the counterpart's clarification.",
        },
    }
    modes["3.2"] = {
        "present": True,
        "evidence": {
            "m_0017": "Agent claims verification without checking timestamps.",
            "tr_0007": "Tool output contradicts verification claim.",
        },
    }
    return {
        "summary": "The task failed due to repeated ignored clarifications.",
        "task_completed": False,
        "failure_modes": modes,
    }


def test_parse_structured_response(
    mock_structured_payload: dict[str, object],
) -> None:
    """Structured parser should return mode booleans and evidence map."""
    from mast.structured_output import parse_structured_response

    raw_payload = json.dumps(mock_structured_payload)
    result = parse_structured_response(mock_structured_payload, raw_payload)

    assert isinstance(result, EvaluationResult)
    assert result.task_completed is False
    assert result.failure_modes.step_repetition is True
    assert result.failure_modes.unaware_of_termination_conditions is True
    assert result.failure_modes.ignored_other_agent_input is True
    assert result.failure_modes.no_or_incorrect_verification is True
    assert result.failure_mode_evidence["3.2"]["m_0017"]
    assert result.failure_mode_evidence["3.2"]["tr_0007"]


def test_parse_structured_response_rejects_non_bool_task_completed(
    mock_structured_payload: dict[str, object],
) -> None:
    from mast.structured_output import parse_structured_response

    payload = dict(mock_structured_payload)
    payload["task_completed"] = "yes"

    with pytest.raises(ValueError, match="task_completed"):
        parse_structured_response(payload, json.dumps(payload))


def test_parse_structured_response_rejects_non_bool_mode_present(
    mock_structured_payload: dict[str, object],
) -> None:
    from mast.structured_output import parse_structured_response

    payload = dict(mock_structured_payload)
    failure_modes = cast(dict[str, object], payload["failure_modes"]).copy()
    mode_11 = cast(dict[str, object], failure_modes["1.1"]).copy()
    mode_11["present"] = "no"
    failure_modes["1.1"] = mode_11
    payload["failure_modes"] = failure_modes

    with pytest.raises(ValueError, match="present"):
        parse_structured_response(payload, json.dumps(payload))


def test_parse_structured_response_rejects_missing_failure_mode_payload(
    mock_structured_payload: dict[str, object],
) -> None:
    from mast.structured_output import parse_structured_response

    payload = dict(mock_structured_payload)
    failure_modes = cast(dict[str, object], payload["failure_modes"]).copy()
    del failure_modes["2.4"]
    payload["failure_modes"] = failure_modes

    with pytest.raises(ValueError, match="2.4"):
        parse_structured_response(payload, json.dumps(payload))


def test_parse_structured_response_rejects_non_object_evidence(
    mock_structured_payload: dict[str, object],
) -> None:
    from mast.structured_output import parse_structured_response

    payload = dict(mock_structured_payload)
    failure_modes = cast(dict[str, object], payload["failure_modes"]).copy()
    mode_33 = cast(dict[str, object], failure_modes["3.3"]).copy()
    mode_33["present"] = True
    mode_33["evidence"] = ["m_0001"]
    failure_modes["3.3"] = mode_33
    payload["failure_modes"] = failure_modes

    with pytest.raises(ValueError, match=r"3\.3\.evidence"):
        parse_structured_response(payload, json.dumps(payload))


def test_contract_schema_and_mapping_alignment() -> None:
    schema_required = STRUCTURED_RESPONSE_SCHEMA["properties"]["failure_modes"][
        "required"
    ]
    assert list(FAILURE_MODE_CODES) == schema_required
    assert set(FAILURE_MODE_CODES) == set(FAILURE_MODE_FIELD_MAP)


def test_failure_modes_to_dict() -> None:
    """Test FailureModes.to_dict() method."""
    fm = FailureModes(
        disobey_task_specification=True,
        disobey_role_specification=False,
        step_repetition=True,
        loss_of_conversation_history=False,
        unaware_of_termination_conditions=False,
        conversation_reset=False,
        fail_to_ask_for_clarification=False,
        task_derailment=False,
        information_withholding=False,
        ignored_other_agent_input=False,
        action_reasoning_mismatch=False,
        premature_termination=False,
        no_or_incorrect_verification=False,
        weak_verification=False,
    )

    d = fm.to_dict()
    assert d["1.1"] is True
    assert d["1.2"] is False
    assert d["1.3"] is True
    assert len(d) == 14


def test_failure_modes_get_detected() -> None:
    fm = FailureModes(
        disobey_task_specification=False,
        disobey_role_specification=False,
        step_repetition=True,
        loss_of_conversation_history=False,
        unaware_of_termination_conditions=False,
        conversation_reset=False,
        fail_to_ask_for_clarification=False,
        task_derailment=False,
        information_withholding=False,
        ignored_other_agent_input=True,
        action_reasoning_mismatch=False,
        premature_termination=False,
        no_or_incorrect_verification=False,
        weak_verification=False,
    )

    detected = fm.get_detected()
    assert [item["code"] for item in detected] == ["1.3", "2.5"]


@pytest.mark.parametrize(
    ("patch_target", "model_name"),
    [
        ("mast.evaluator._call_openai", "openai/gpt-5.2"),
        ("mast.evaluator._call_anthropic_vertex", "vertex/claude-opus-4-6"),
    ],
)
def test_evaluate_with_mocked_provider(
    sample_trace: str,
    mock_structured_payload: dict[str, object],
    patch_target: str,
    model_name: str,
) -> None:
    """Test evaluate function with mocked model provider calls."""
    raw_payload = json.dumps(mock_structured_payload)
    with patch(patch_target, return_value=(mock_structured_payload, raw_payload)):
        results = evaluate([sample_trace], model_name=cast(ModelName, model_name))

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert result.task_completed is False
    assert result.failure_modes.step_repetition is True
    assert result.failure_modes.ignored_other_agent_input is True
    assert result.failure_modes.no_or_incorrect_verification is True
    assert sorted(result.failure_mode_evidence["3.2"].keys()) == ["m_0017", "tr_0007"]


def test_evaluate_truncates_trace_before_model_call(
    mock_structured_payload: dict[str, object],
) -> None:
    raw_payload = json.dumps(mock_structured_payload)
    trace = "abcdefghij"

    with (
        patch("mast.evaluator._load_definitions", return_value=""),
        patch("mast.evaluator._load_examples", return_value="EXAMPLES"),
        patch(
            "mast.evaluator._call_openai",
            return_value=(mock_structured_payload, raw_payload),
        ) as mocked_call,
    ):
        evaluate([trace], model_name="openai/gpt-5.2", max_trace_length=10)

    prompt = mocked_call.call_args[0][0]
    assert "Trace:\nab\n" in prompt


def test_evaluate_rejects_unknown_model(sample_trace: str) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        evaluate([sample_trace], model_name=cast(ModelName, "unknown/model"))


@pytest.mark.integration
def test_evaluate_e2e(sample_trace: str) -> None:
    """
    End-to-end test with real OpenAI API call.

    Run with: pytest -m integration tests/
    Requires OPENAI_API_KEY environment variable.
    """
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    results = evaluate([sample_trace], model_name="openai/gpt-5.2")

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0
    assert isinstance(result.task_completed, bool)
    assert isinstance(result.failure_modes, FailureModes)


@pytest.mark.integration
def test_evaluate_e2e_vertex(sample_trace: str) -> None:
    """
    End-to-end test with real Anthropic Vertex API call.

    Run with: pytest -m integration tests/
    Requires GOOGLE_CLOUD_PROJECT environment variable and gcloud auth.
    """
    import os

    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        pytest.skip("GOOGLE_CLOUD_PROJECT not set")

    results = evaluate([sample_trace], model_name="vertex/claude-opus-4-6")

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0
    assert isinstance(result.task_completed, bool)
    assert isinstance(result.failure_modes, FailureModes)
