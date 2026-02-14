"""Simple evaluation API for custom agents on tau2 with MAST failure analysis."""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel

# Set up tau2 data directory before importing tau2
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "tau2_data" / "data"
os.environ["TAU2_DATA_DIR"] = str(DATA_DIR)

# Add mast to path
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from tau2.agent.base import LocalAgent
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

from mast.evaluator import evaluate as mast_evaluate, ModelName as MastModelName
from mast.models import EvaluationResult as MastEvaluationResult


class TaskEvaluationResult(BaseModel):
    """Result of evaluating an agent on a single tau2 task."""

    task_id: str
    reward: float
    success: bool
    num_turns: int
    agent_cost: float
    user_cost: float
    termination_reason: str
    trace: str  # conversation trace for MAST
    mast_result: Optional[dict] = None  # MAST evaluation result


class EvaluationSummary(BaseModel):
    """Summary of evaluation results."""

    total_tasks: int
    successful_tasks: int
    average_reward: float
    total_cost: float
    results: list[TaskEvaluationResult]


class OpenEvolveResult(BaseModel):
    """OpenEvolve-compatible evaluation result."""

    metrics: dict[str, float]
    artifacts: dict[str, str]


def simulation_to_trace(simulation: SimulationRun) -> str:
    """Convert tau2 simulation messages to a trace string for MAST."""
    lines = []
    if not simulation.messages:
        return ""

    for msg in simulation.messages:
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")

        # Handle tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_strs = []
            for tc in msg.tool_calls:
                tool_strs.append(f"{tc.name}({json.dumps(tc.arguments)})")
            lines.append(f"[{role}] Tool calls: {', '.join(tool_strs)}")
            if content:
                lines.append(f"[{role}] {content}")
        elif role == "tool":
            # Tool response
            tool_id = getattr(msg, "id", "unknown")
            lines.append(f"[tool:{tool_id}] {content}")
        else:
            lines.append(f"[{role}] {content}")

    return "\n".join(lines)


def evaluate(
    agent_factory: Callable[[list[Tool], str], LocalAgent],
    domain: str = "airline",
    num_tasks: int = 5,
    num_trials: int = 1,
    user_model: str = "gpt-4.1-mini",
    task_ids: Optional[list[str]] = None,
    max_steps: int = 200,
    max_concurrency: int = 1,
    run_mast: bool = False,
    mast_model: MastModelName = "vertex/claude-opus-4-5",
) -> EvaluationSummary:
    """
    Evaluate a custom agent on tau2 with optional MAST failure analysis.

    Args:
        agent_factory: A callable that takes (tools, domain_policy) and returns a LocalAgent.
                       Example: lambda tools, policy: LangGraphAgent(tools, policy, model="gpt-4.1")
        domain: The domain to evaluate on (default: "airline")
        num_tasks: Number of tasks to run (default: 5)
        num_trials: Number of trials per task (default: 1)
        user_model: The LLM to use for the user simulator (default: "gpt-4.1-mini")
        task_ids: Specific task IDs to run (optional)
        max_steps: Maximum steps per task (default: 200)
        max_concurrency: Maximum concurrent task evaluations (default: 1)
        run_mast: Whether to run MAST failure analysis (default: False)
        mast_model: Model to use for MAST evaluation (default: vertex/claude-opus-4-5)

    Returns:
        EvaluationSummary with results for each task (includes MAST results if run_mast=True)
    """
    # Get environment and tasks
    env_factory = registry.get_env_constructor(domain)
    task_factory = registry.get_tasks_loader(domain)
    all_tasks = task_factory(None)

    # Filter tasks
    if task_ids:
        tasks = [t for t in all_tasks if t.id in task_ids]
    else:
        tasks = all_tasks[:num_tasks]

    def run_single_task(task: Task, trial: int) -> TaskEvaluationResult:
        """Run a single task evaluation (sync)."""
        print(f"Running task {task.id}, trial {trial + 1}/{num_trials}")

        # Create fresh environment for this task
        env: Environment = env_factory()

        # Create agent using the factory
        agent = agent_factory(env.get_tools(), env.get_policy())

        # Get user tools if available
        user_tools = None
        try:
            user_tools = env.get_user_tools()
        except ValueError:
            pass

        # Create user simulator
        user = UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=user_model,
            llm_args={"temperature": 0.0},
        )

        # Create orchestrator
        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=max_steps,
            max_errors=10,
        )

        # Run simulation
        simulation = orchestrator.run()

        # Evaluate the simulation to get reward
        reward_info = evaluate_simulation(
            domain=domain,
            task=task,
            simulation=simulation,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
        )
        reward = reward_info.reward if reward_info else 0.0

        # Convert simulation to trace for MAST
        trace = simulation_to_trace(simulation)

        # Extract result
        result = TaskEvaluationResult(
            task_id=task.id,
            reward=reward,
            success=reward == 1.0,
            num_turns=len(simulation.messages) if simulation.messages else 0,
            agent_cost=simulation.agent_cost or 0.0,
            user_cost=simulation.user_cost or 0.0,
            termination_reason=str(simulation.termination_reason.value) if simulation.termination_reason else "unknown",
            trace=trace,
        )

        # Print result
        status = "PASSED" if result.success else "FAILED"
        print(f"  Task {task.id}: {status} (reward={result.reward:.2f})")
        return result

    async def run_with_semaphore(
        sem: asyncio.Semaphore, task: Task, trial: int
    ) -> TaskEvaluationResult:
        """Run task with semaphore-controlled concurrency."""
        async with sem:
            return await asyncio.to_thread(run_single_task, task, trial)

    async def run_all_tasks() -> list[TaskEvaluationResult]:
        """Run all tasks concurrently with semaphore."""
        sem = asyncio.Semaphore(max_concurrency)
        coros = [
            run_with_semaphore(sem, task, trial)
            for task in tasks
            for trial in range(num_trials)
        ]
        return await asyncio.gather(*coros)

    # Run evaluations
    results = asyncio.run(run_all_tasks())

    # Run MAST evaluation on failed tasks if requested
    if run_mast:
        failed_results = [r for r in results if not r.success and r.trace]
        if failed_results:
            print(f"\nRunning MAST analysis on {len(failed_results)} failed tasks...")
            traces = [r.trace for r in failed_results]
            mast_results = mast_evaluate(traces, model_name=mast_model)

            for result, mast_result in zip(failed_results, mast_results):
                result.mast_result = {
                    "summary": mast_result.summary,
                    "task_completed": mast_result.task_completed,
                    "failure_modes": mast_result.failure_modes.to_dict(),
                    "detected": [
                        {"code": d["code"], "name": d["name"]}
                        for d in mast_result.failure_modes.get_detected()
                    ],
                }

    # Compute summary
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.success)
    average_reward = sum(r.reward for r in results) / total_tasks if total_tasks > 0 else 0.0
    total_cost = sum(r.agent_cost + r.user_cost for r in results)

    summary = EvaluationSummary(
        total_tasks=total_tasks,
        successful_tasks=successful_tasks,
        average_reward=average_reward,
        total_cost=total_cost,
        results=results,
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"  Tasks: {successful_tasks}/{total_tasks} passed")
    print(f"  Average Reward: {average_reward:.2%}")
    print(f"  Total Cost: ${total_cost:.4f}")
    print("=" * 60)

    return summary


def to_openevolve_result(summary: EvaluationSummary) -> OpenEvolveResult:
    """Convert EvaluationSummary to OpenEvolve-compatible format.

    Returns:
        OpenEvolveResult with metrics and artifacts for evolutionary feedback.
    """
    # Compute metrics
    num_failures = sum(1 for r in summary.results if not r.success)
    failure_score = 1.0 - (num_failures / max(summary.total_tasks, 1))

    metrics = {
        "combined_score": summary.average_reward,
        "success_rate": summary.successful_tasks / max(summary.total_tasks, 1),
        "failure_score": failure_score,
        "num_failures": float(num_failures),
        "total_tasks": float(summary.total_tasks),
    }

    # Build failure feedback artifact
    failure_feedback_lines = []
    mast_summary_lines = []

    for r in summary.results:
        if not r.success:
            failure_feedback_lines.append(
                f"- Task {r.task_id}: {r.termination_reason} (reward={r.reward:.2f})"
            )
            if r.mast_result:
                detected = r.mast_result.get("detected", [])
                if detected:
                    mast_summary_lines.append(f"Task {r.task_id}:")
                    mast_summary_lines.append(f"  Summary: {r.mast_result.get('summary', '')}")
                    for d in detected:
                        mast_summary_lines.append(f"  - [{d['code']}] {d['name']}")

    artifacts = {
        "failed_tasks": "\n".join(failure_feedback_lines) if failure_feedback_lines else "No failures",
        "mast_analysis": "\n".join(mast_summary_lines) if mast_summary_lines else "No MAST analysis available",
    }

    return OpenEvolveResult(metrics=metrics, artifacts=artifacts)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate agent on tau2 with MAST analysis")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks")
    parser.add_argument("--max-concurrency", type=int, default=5, help="Max concurrent tasks")
    parser.add_argument("--run-mast", action="store_true", help="Run MAST failure analysis")
    parser.add_argument("--mast-model", default="vertex/claude-opus-4-5",
                        choices=["vertex/claude-opus-4-5", "openai/gpt-5.1"],
                        help="Model for MAST evaluation")
    args = parser.parse_args()

    from tau2.agent.llm_agent import LLMAgent

    def create_agent(tools: list[Tool], policy: str) -> LocalAgent:
        return LLMAgent(
            tools=tools,
            domain_policy=policy,
            llm="gpt-4.1-mini",
            llm_args={"temperature": 0.0},
        )

    summary = evaluate(
        agent_factory=create_agent,
        domain="airline",
        num_tasks=args.num_tasks,
        num_trials=1,
        max_concurrency=args.max_concurrency,
        run_mast=args.run_mast,
        mast_model=args.mast_model,
    )

    print(f"\nResults: {summary.successful_tasks}/{summary.total_tasks} tasks passed")

    # Convert to OpenEvolve format
    openevolve_result = to_openevolve_result(summary)
    print("\n" + "=" * 60)
    print("OPENEVOLVE-COMPATIBLE OUTPUT")
    print("=" * 60)
    print(f"Metrics: {json.dumps(openevolve_result.metrics, indent=2)}")
    print(f"\nArtifacts:")
    for key, value in openevolve_result.artifacts.items():
        print(f"\n### {key}")
        print(value)

