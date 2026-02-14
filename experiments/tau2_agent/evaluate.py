"""Simple evaluation API for custom agents on tau2."""
import asyncio
import os
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

# Set up tau2 data directory before importing tau2
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "tau2_data" / "data"
os.environ["TAU2_DATA_DIR"] = str(DATA_DIR)

from tau2.agent.base import LocalAgent
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator


class EvaluationResult(BaseModel):
    """Result of evaluating an agent on tau2."""

    task_id: str
    reward: float
    success: bool
    num_turns: int
    agent_cost: float
    user_cost: float
    termination_reason: str


class EvaluationSummary(BaseModel):
    """Summary of evaluation results."""

    total_tasks: int
    successful_tasks: int
    average_reward: float
    total_cost: float
    results: list[EvaluationResult]


def evaluate(
    agent_factory: Callable[[list[Tool], str], LocalAgent],
    domain: str = "airline",
    num_tasks: int = 5,
    num_trials: int = 1,
    user_model: str = "gpt-4.1-mini",
    task_ids: Optional[list[str]] = None,
    max_steps: int = 200,
    max_concurrency: int = 1,
) -> EvaluationSummary:
    """
    Evaluate a custom agent on tau2.

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

    Returns:
        EvaluationSummary with results for each task
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

    def run_single_task(task: Task, trial: int) -> EvaluationResult:
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

        # Extract result
        result = EvaluationResult(
            task_id=task.id,
            reward=reward,
            success=reward == 1.0,
            num_turns=len(simulation.messages) if simulation.messages else 0,
            agent_cost=simulation.agent_cost or 0.0,
            user_cost=simulation.user_cost or 0.0,
            termination_reason=str(simulation.termination_reason.value) if simulation.termination_reason else "unknown",
        )

        # Print result
        status = "PASSED" if result.success else "FAILED"
        print(f"  Task {task.id}: {status} (reward={result.reward:.2f})")
        return result

    async def run_with_semaphore(
        sem: asyncio.Semaphore, task: Task, trial: int
    ) -> EvaluationResult:
        """Run task with semaphore-controlled concurrency."""
        async with sem:
            return await asyncio.to_thread(run_single_task, task, trial)

    async def run_all_tasks() -> list[EvaluationResult]:
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
    print(f"EVALUATION COMPLETE")
    print(f"  Tasks: {successful_tasks}/{total_tasks} passed")
    print(f"  Average Reward: {average_reward:.2%}")
    print(f"  Total Cost: ${total_cost:.4f}")
    print("=" * 60)

    return summary


if __name__ == '__main__':
    from tau2.agent.llm_agent import LLMAgent
    
    def create_agent(tools: list[Tool], policy: str) -> LocalAgent:
        return LLMAgent(
            tools=tools,
            instructions=policy,
            llm="gpt-4.1-mini",
            llm_args={"temperature": 0.0},
        )
    
    summary = evaluate(
        agent_factory=create_agent,
        domain="airline",
        num_tasks=50,
        num_trials=1,
        max_concurrency=20,
    )
    
    print(f"\nResults: {summary.successful_tasks}/{summary.total_tasks} tasks passed")

