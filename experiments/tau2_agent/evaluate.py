"""
OpenEvolve-compatible evaluator for tau2 agents with MAST failure analysis.

Takes a program_path to an agent file, runs tau2 evaluation, and returns
OpenEvolve EvaluationResult with metrics and artifacts.
"""
import asyncio
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

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

# Import OpenEvolve's EvaluationResult
try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:
    # Fallback if openevolve not installed - define compatible class
    from dataclasses import dataclass
    from typing import Dict, Any

    @dataclass
    class EvaluationResult:
        metrics: Dict[str, float]
        artifacts: Dict[str, str] = None

        def __post_init__(self):
            if self.artifacts is None:
                self.artifacts = {}


# Configuration
DEFAULT_NUM_TASKS = 10
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_DOMAIN = "airline"
DEFAULT_USER_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_STEPS = 200


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
            tool_id = getattr(msg, "id", "unknown")
            lines.append(f"[tool:{tool_id}] {content}")
        else:
            lines.append(f"[{role}] {content}")

    return "\n".join(lines)


def load_agent_factory(program_path: str) -> Callable[[list[Tool], str], LocalAgent]:
    """
    Load an agent factory from a program file.

    The program must define a `create_agent(tools, domain_policy) -> LocalAgent` function.
    """
    spec = importlib.util.spec_from_file_location("agent_program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    if not hasattr(program, "create_agent"):
        raise AttributeError("Program must define 'create_agent(tools, domain_policy) -> LocalAgent' function")

    return program.create_agent


def run_single_task(
    task: Task,
    agent_factory: Callable[[list[Tool], str], LocalAgent],
    env_factory: Callable[[], Environment],
    domain: str,
    user_model: str,
    max_steps: int,
) -> dict:
    """Run a single task evaluation."""
    # Create fresh environment
    env: Environment = env_factory()

    # Create agent
    agent = agent_factory(env.get_tools(), env.get_policy())

    # Get user tools
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

    # Evaluate
    reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
    )
    reward = reward_info.reward if reward_info else 0.0

    # Convert to trace
    trace = simulation_to_trace(simulation)

    return {
        "task_id": task.id,
        "reward": reward,
        "success": reward == 1.0,
        "num_turns": len(simulation.messages) if simulation.messages else 0,
        "termination_reason": str(simulation.termination_reason.value) if simulation.termination_reason else "unknown",
        "trace": trace,
    }


def evaluate(
    program_path: str,
    num_tasks: int = DEFAULT_NUM_TASKS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    domain: str = DEFAULT_DOMAIN,
    user_model: str = DEFAULT_USER_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    run_mast: bool = True,
    mast_model: MastModelName = "vertex/claude-opus-4-5",
) -> EvaluationResult:
    """
    Evaluate an agent program on tau2 with MAST failure analysis.

    OpenEvolve-compatible evaluator that takes a program_path and returns
    EvaluationResult with metrics and artifacts.

    Args:
        program_path: Path to agent program file (must define create_agent function)
        num_tasks: Number of tau2 tasks to run
        max_concurrency: Maximum concurrent task evaluations
        domain: tau2 domain (default: airline)
        user_model: LLM for user simulator
        max_steps: Max steps per task
        run_mast: Whether to run MAST analysis on failed tasks
        mast_model: Model for MAST evaluation

    Returns:
        EvaluationResult with metrics and artifacts
    """
    try:
        # Load agent factory from program
        agent_factory = load_agent_factory(program_path)
    except Exception as e:
        print(f"Error loading agent from {program_path}: {e}")
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "success_rate": 0.0,
                "num_failures": 1.0,
                "error": 1.0,
            },
            artifacts={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
                "suggestion": "Program must define 'create_agent(tools, domain_policy) -> LocalAgent' function",
            }
        )

    try:
        # Get environment and tasks
        env_factory = registry.get_env_constructor(domain)
        task_factory = registry.get_tasks_loader(domain)
        all_tasks = task_factory(None)
        tasks = all_tasks[:num_tasks]

        # Run tasks with async concurrency
        async def run_with_semaphore(sem, task):
            async with sem:
                return await asyncio.to_thread(
                    run_single_task,
                    task, agent_factory, env_factory, domain, user_model, max_steps
                )

        async def run_all():
            sem = asyncio.Semaphore(max_concurrency)
            coros = [run_with_semaphore(sem, task) for task in tasks]
            return await asyncio.gather(*coros)

        results = asyncio.run(run_all())

        # Compute metrics
        total_tasks = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total_tasks - successful
        avg_reward = sum(r["reward"] for r in results) / total_tasks if total_tasks > 0 else 0.0
        success_rate = successful / total_tasks if total_tasks > 0 else 0.0

        print(f"\nTau2 Results: {successful}/{total_tasks} passed ({success_rate:.1%})")

        # Run MAST on failed tasks
        mast_analysis_lines = []
        if run_mast and failed > 0:
            failed_results = [r for r in results if not r["success"] and r["trace"]]
            if failed_results:
                print(f"Running MAST analysis on {len(failed_results)} failed tasks...")
                traces = [r["trace"] for r in failed_results]

                try:
                    mast_results = mast_evaluate(traces, model_name=mast_model)

                    for result, mast_result in zip(failed_results, mast_results):
                        detected = mast_result.failure_modes.get_detected()
                        if detected:
                            mast_analysis_lines.append(f"Task {result['task_id']}:")
                            mast_analysis_lines.append(f"  Summary: {mast_result.summary}")
                            for d in detected:
                                mast_analysis_lines.append(f"  - [{d['code']}] {d['name']}: {d['definition']}")
                except Exception as e:
                    print(f"MAST evaluation failed: {e}")
                    mast_analysis_lines.append(f"MAST evaluation error: {str(e)}")

        # Build artifacts
        failed_tasks_lines = []
        for r in results:
            if not r["success"]:
                failed_tasks_lines.append(
                    f"- Task {r['task_id']}: {r['termination_reason']} (reward={r['reward']:.2f})"
                )

        artifacts = {
            "failed_tasks": "\n".join(failed_tasks_lines) if failed_tasks_lines else "No failures",
            "mast_analysis": "\n".join(mast_analysis_lines) if mast_analysis_lines else "No MAST failures detected",
            "evaluation_summary": f"Evaluated {total_tasks} tasks: {successful} passed, {failed} failed",
        }

        return EvaluationResult(
            metrics={
                "combined_score": avg_reward,
                "success_rate": success_rate,
                "num_failures": float(failed),
                "total_tasks": float(total_tasks),
            },
            artifacts=artifacts
        )

    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "success_rate": 0.0,
                "num_failures": 1.0,
                "error": 1.0,
            },
            artifacts={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
                "suggestion": "Check agent implementation and tau2 environment setup",
            }
        )


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path: str) -> EvaluationResult:
    """
    First stage: Quick validation with 2 tasks.
    Checks if agent runs without errors.
    """
    return evaluate(
        program_path,
        num_tasks=2,
        max_concurrency=2,
        run_mast=False,  # Skip MAST in stage 1 for speed
    )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """
    Second stage: Full evaluation with MAST analysis.
    """
    return evaluate(
        program_path,
        num_tasks=DEFAULT_NUM_TASKS,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
        run_mast=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate agent on tau2 with MAST analysis")
    parser.add_argument("program_path", nargs="?", default=None,
                        help="Path to agent program file")
    parser.add_argument("--num-tasks", type=int, default=DEFAULT_NUM_TASKS)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--run-mast", action="store_true", default=True)
    parser.add_argument("--no-mast", action="store_true", help="Disable MAST analysis")
    parser.add_argument("--mast-model", default="vertex/claude-opus-4-5",
                        choices=["vertex/claude-opus-4-5", "openai/gpt-5.1"])
    args = parser.parse_args()

    # If no program path, use default tau2 LLMAgent
    if args.program_path is None:
        # Create a temporary agent file for testing
        import tempfile
        agent_code = '''
from tau2.agent.llm_agent import LLMAgent

def create_agent(tools, domain_policy):
    return LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm="gpt-4.1-mini",
        llm_args={"temperature": 0.0},
    )
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(agent_code)
            program_path = f.name
        print(f"Using default LLMAgent (temp file: {program_path})")
    else:
        program_path = args.program_path

    result = evaluate(
        program_path,
        num_tasks=args.num_tasks,
        max_concurrency=args.max_concurrency,
        run_mast=not args.no_mast,
        mast_model=args.mast_model,
    )

    print("\n" + "=" * 60)
    print("OPENEVOLVE-COMPATIBLE OUTPUT")
    print("=" * 60)
    print(f"Metrics: {json.dumps(result.metrics, indent=2)}")
    print(f"\nArtifacts:")
    for key, value in result.artifacts.items():
        print(f"\n### {key}")
        print(value)
