import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mast import EvaluationResult, FailureModes, evaluate

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
    from mast.evaluator import _parse_structured_response

    raw_payload = json.dumps(mock_structured_payload)
    result = _parse_structured_response(mock_structured_payload, raw_payload)

    assert isinstance(result, EvaluationResult)
    assert result.task_completed is False
    assert result.failure_modes.step_repetition is True
    assert result.failure_modes.unaware_of_termination_conditions is True
    assert result.failure_modes.ignored_other_agent_input is True
    assert result.failure_modes.no_or_incorrect_verification is True
    assert result.failure_mode_evidence["3.2"]["m_0017"]
    assert result.failure_mode_evidence["3.2"]["tr_0007"]


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


def test_evaluate_with_mock(
    sample_trace: str,
    mock_structured_payload: dict[str, object],
) -> None:
    """Test evaluate function with mocked OpenAI call."""
    raw_payload = json.dumps(mock_structured_payload)
    with patch(
        "mast.evaluator._call_openai",
        return_value=(mock_structured_payload, raw_payload),
    ):
        results = evaluate([sample_trace])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert result.task_completed is False
    assert result.failure_modes.step_repetition is True
    assert result.failure_modes.ignored_other_agent_input is True
    assert result.failure_modes.no_or_incorrect_verification is True
    assert sorted(result.failure_mode_evidence["3.2"].keys()) == ["m_0017", "tr_0007"]


def test_evaluate_with_mock_vertex(
    sample_trace: str,
    mock_structured_payload: dict[str, object],
) -> None:
    """Test evaluate function with mocked Anthropic Vertex call."""
    raw_payload = json.dumps(mock_structured_payload)
    with patch(
        "mast.evaluator._call_anthropic_vertex",
        return_value=(mock_structured_payload, raw_payload),
    ):
        results = evaluate([sample_trace], model_name="vertex/claude-opus-4-6")

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert result.task_completed is False
    assert result.failure_modes.ignored_other_agent_input is True
    assert result.failure_modes.no_or_incorrect_verification is True


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
