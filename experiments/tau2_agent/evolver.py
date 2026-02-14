"""
OpenEvolve-based agent evolution for tau2-airline benchmark.

Uses evolutionary algorithms with LLM mutations to discover better agent implementations.
Evaluates agents using tau2 benchmark with MAST failure analysis for feedback.
"""
import argparse
import os
from pathlib import Path

# Ensure we're in the right directory for imports
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

from openevolve import run_evolution
from evaluate import evaluate


def create_evaluator(num_tasks: int, max_concurrency: int, run_mast: bool):
    """Create an evaluator function with the given parameters."""
    def evaluator(program_path: str):
        """Evaluate an evolved agent program."""
        return evaluate(
            program_path,
            num_tasks=num_tasks,
            max_concurrency=max_concurrency,
            run_mast=run_mast,
        )
    return evaluator


def main():
    parser = argparse.ArgumentParser(description="Evolve tau2 agents with OpenEvolve")
    parser.add_argument("--iterations", type=int, default=10, help="Number of evolution iterations")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tau2 tasks per evaluation")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Max concurrent task evaluations")
    parser.add_argument("--output-dir", type=str, default="openevolve_output", help="Output directory")
    parser.add_argument("--run-mast", action="store_true", help="Run MAST analysis on failures")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Read initial program
    initial_program_path = SCRIPT_DIR / "initial_agent.py"

    # Create evaluator function
    evaluator = create_evaluator(
        num_tasks=args.num_tasks,
        max_concurrency=args.max_concurrency,
        run_mast=args.run_mast,
    )

    # Config file path
    config_path = SCRIPT_DIR / args.config

    print("=" * 60)
    print("OPENEVOLVE TAU2 AGENT EVOLUTION")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Tasks per evaluation: {args.num_tasks}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"MAST analysis: {args.run_mast}")
    print(f"Config: {config_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Run evolution
    result = run_evolution(
        initial_program=str(initial_program_path),
        evaluator=evaluator,
        config=str(config_path),
        iterations=args.iterations,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best score: {result.best_score:.4f}")
    print(f"Best metrics: {result.metrics}")
    print(f"Output saved to: {result.output_dir or args.output_dir}")

    # Save best program
    best_program_path = Path(args.output_dir) / "best_agent.py"
    best_program_path.parent.mkdir(parents=True, exist_ok=True)
    best_program_path.write_text(result.best_code)
    print(f"Best agent saved to: {best_program_path}")

    return result


if __name__ == "__main__":
    main()
