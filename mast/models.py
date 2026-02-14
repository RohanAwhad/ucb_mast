from dataclasses import dataclass


@dataclass
class FailureModes:
    """Failure modes detected in multi-agent system traces."""

    disobey_task_specification: bool  # 1.1
    disobey_role_specification: bool  # 1.2
    step_repetition: bool  # 1.3
    loss_of_conversation_history: bool  # 1.4
    unaware_of_termination_conditions: bool  # 1.5
    conversation_reset: bool  # 2.1
    fail_to_ask_for_clarification: bool  # 2.2
    task_derailment: bool  # 2.3
    information_withholding: bool  # 2.4
    ignored_other_agent_input: bool  # 2.5
    action_reasoning_mismatch: bool  # 2.6
    premature_termination: bool  # 3.1
    no_or_incorrect_verification: bool  # 3.2
    weak_verification: bool  # 3.3

    def to_dict(self) -> dict[str, bool]:
        return {
            "1.1": self.disobey_task_specification,
            "1.2": self.disobey_role_specification,
            "1.3": self.step_repetition,
            "1.4": self.loss_of_conversation_history,
            "1.5": self.unaware_of_termination_conditions,
            "2.1": self.conversation_reset,
            "2.2": self.fail_to_ask_for_clarification,
            "2.3": self.task_derailment,
            "2.4": self.information_withholding,
            "2.5": self.ignored_other_agent_input,
            "2.6": self.action_reasoning_mismatch,
            "3.1": self.premature_termination,
            "3.2": self.no_or_incorrect_verification,
            "3.3": self.weak_verification,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a multi-agent system trace."""

    summary: str  # A: Freeform text summary
    task_completed: bool  # B: Whether task was successfully completed
    failure_modes: FailureModes  # C: Detected failure modes
    raw_response: str  # Original LLM response for debugging
