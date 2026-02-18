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
def mock_openai_response() -> str:
    """Mock OpenAI response for testing parsing."""
    return """@@
A. The task failed because the mathproxyagent repeatedly ignored the assistant's request for clarification about missing information. The assistant correctly identified that the problem lacked crucial data (total ribbon length), but the mathproxyagent kept responding with "Continue" instead of providing the needed information or acknowledging the issue.

B. no

C.
1.1 Disobey Task Specification: no
1.2 Disobey Role Specification: no
1.3 Step Repetition: yes
1.4 Loss of Conversation History: no
1.5 Unaware of Termination Conditions: yes
2.1 Conversation Reset: no
2.2 Fail to Ask for Clarification: no
2.3 Task Derailment: no
2.4 Information Withholding: no
2.5 Ignored Other Agent's Input: yes
2.6 Action-Reasoning Mismatch: no
3.1 Premature Termination: no
3.2 No or Incorrect Verification: no
3.3 Weak Verification: no

D.
1.1 Disobey Task Specification: []
1.2 Disobey Role Specification: []
1.3 Step Repetition: [m_0007, m_0010]
1.4 Loss of Conversation History: []
1.5 Unaware of Termination Conditions: [m_0010, m_0012]
2.1 Conversation Reset: []
2.2 Fail to Ask for Clarification: []
2.3 Task Derailment: []
2.4 Information Withholding: []
2.5 Ignored Other Agent's Input: [m_0008]
2.6 Action-Reasoning Mismatch: []
3.1 Premature Termination: []
3.2 No or Incorrect Verification: []
3.3 Weak Verification: []
@@"""


def test_parse_response(mock_openai_response: str) -> None:
    """Test that response parsing works correctly."""
    from mast.evaluator import _parse_response

    result = _parse_response(mock_openai_response)

    assert isinstance(result, EvaluationResult)
    assert result.task_completed is False
    assert (
        "mathproxyagent" in result.summary.lower()
        or "ignored" in result.summary.lower()
    )

    # Check specific failure modes
    assert result.failure_modes.step_repetition is True
    assert result.failure_modes.unaware_of_termination_conditions is True
    assert result.failure_modes.ignored_other_agent_input is True
    assert result.failure_modes.disobey_task_specification is False
    assert result.failure_modes.task_derailment is False
    assert result.failure_mode_evidence["1.3"] == ["m_0007", "m_0010"]
    assert result.failure_mode_evidence["1.5"] == ["m_0010", "m_0012"]
    assert result.failure_mode_evidence["2.5"] == ["m_0008"]


def test_parse_response_bracketed_mode_evidence() -> None:
    """Support bracketed mode format like [2.6] ...: [id1, id2]."""
    from mast.evaluator import _parse_response

    response = """@@
A. Action-reasoning mismatch was observed.
B. yes
C.
2.6 Action-Reasoning Mismatch: yes
D.
[2.6] Action-Reasoning Mismatch: [abc, def, xyz]
@@"""

    result = _parse_response(response)
    assert result.failure_modes.action_reasoning_mismatch is True
    assert result.failure_mode_evidence["2.6"] == ["abc", "def", "xyz"]


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


def test_evaluate_with_mock(sample_trace: str, mock_openai_response: str) -> None:
    """Test evaluate function with mocked OpenAI call."""
    with patch("mast.evaluator._call_openai", return_value=mock_openai_response):
        results = evaluate([sample_trace])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert result.task_completed is False
    assert result.failure_modes.ignored_other_agent_input is True


def test_evaluate_with_mock_vertex(
    sample_trace: str, mock_openai_response: str
) -> None:
    """Test evaluate function with mocked Anthropic Vertex call."""
    with patch(
        "mast.evaluator._call_anthropic_vertex", return_value=mock_openai_response
    ):
        results = evaluate([sample_trace], model_name="vertex/claude-opus-4-6")

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert result.task_completed is False
    assert result.failure_modes.ignored_other_agent_input is True


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

    results = evaluate([sample_trace], model_name="openai/gpt-5.1")

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
