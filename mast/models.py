from dataclasses import dataclass


# Short definitions for each failure mode (from MAST taxonomy)
FAILURE_MODE_DEFINITIONS: dict[str, dict[str, str]] = {
    "1.1": {
        "name": "Disobey Task Specification",
        "definition": "Agent fails to adhere to specified constraints, guidelines, or requirements associated with a particular task.",
    },
    "1.2": {
        "name": "Disobey Role Specification",
        "definition": "Failure to adhere to the defined responsibilities and constraints of an assigned role, potentially leading to an agent behaving like another.",
    },
    "1.3": {
        "name": "Step Repetition",
        "definition": "Agent unnecessarily repeats a phase, task, or stage that has already been completed due to inadequate state tracking or workflow management.",
    },
    "1.4": {
        "name": "Loss of Conversation History",
        "definition": "Unexpected context truncation, disregarding recent interaction history and reverting to an antecedent conversational state.",
    },
    "1.5": {
        "name": "Unaware of Termination Conditions",
        "definition": "Agent fails to recognize criteria designed to trigger termination of an interaction, conversation, phase, or task.",
    },
    "2.1": {
        "name": "Conversation Reset",
        "definition": "Unexpected or unwarranted restarting of a dialogue, potentially losing context and progress made in the interaction.",
    },
    "2.2": {
        "name": "Fail to Ask for Clarification",
        "definition": "Inability to request additional information when faced with unclear or incomplete data, potentially resulting in incorrect actions.",
    },
    "2.3": {
        "name": "Task Derailment",
        "definition": "Deviation from the intended objective or focus of a given task, potentially resulting in irrelevant or unproductive actions.",
    },
    "2.4": {
        "name": "Information Withholding",
        "definition": "Agent possesses critical information but fails to share it with other agents that rely upon this information for their operations.",
    },
    "2.5": {
        "name": "Ignored Other Agent's Input",
        "definition": "Not properly considering input or recommendations provided by other agents, leading to bad decisions or stalled progress.",
    },
    "2.6": {
        "name": "Action-Reasoning Mismatch",
        "definition": "Discrepancy between agent's logical reasoning/conclusions and the actual actions or outputs the system produces.",
    },
    "3.1": {
        "name": "Premature Termination",
        "definition": "Ending a dialogue, interaction or task before all necessary information has been exchanged or objectives have been met.",
    },
    "3.2": {
        "name": "No or Incorrect Verification",
        "definition": "Omission of proper checking or confirmation of task outcomes, allowing errors or inconsistencies to propagate undetected.",
    },
    "3.3": {
        "name": "Weak Verification",
        "definition": "Verification mechanisms exist but fail to comprehensively cover all essential aspects necessary for robust outputs.",
    },
}


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

    @staticmethod
    def get_definition(code: str) -> dict[str, str] | None:
        """Get the name and definition for a failure mode code (e.g., '1.3')."""
        return FAILURE_MODE_DEFINITIONS.get(code)

    @staticmethod
    def get_all_definitions() -> dict[str, dict[str, str]]:
        """Get all failure mode definitions."""
        return FAILURE_MODE_DEFINITIONS

    def get_detected(self) -> list[dict[str, str]]:
        """Get list of detected failure modes with their definitions.

        Returns list of dicts with keys: code, name, definition
        """
        detected = []
        for code, is_detected in self.to_dict().items():
            if is_detected:
                info = FAILURE_MODE_DEFINITIONS[code]
                detected.append({
                    "code": code,
                    "name": info["name"],
                    "definition": info["definition"],
                })
        return detected

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
