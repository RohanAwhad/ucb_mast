"""Evaluate custom LangGraph agent on tau2-airline benchmark."""
import os
from pathlib import Path

# IMPORTANT: Set TAU2_DATA_DIR before any tau2 imports
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "tau2_data" / "data"
os.environ["TAU2_DATA_DIR"] = str(DATA_DIR)

import argparse
from langgraph_agent import LangGraphAgent
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate LangGraph agent on tau2-airline")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Agent LLM model")
    parser.add_argument("--user-model", default="gpt-4.1-mini", help="User simulator LLM model")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks to run")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of trials per task")
    args = parser.parse_args()

    print("=" * 60)
    print("TAU2-AIRLINE EVALUATION (LangGraph Agent)")
    print(f"  Agent Model: {args.model}")
    print(f"  User Model:  {args.user_model}")
    print(f"  Tasks: {args.num_tasks}, Trials: {args.num_trials}")
    print("=" * 60)

    # Create agent factory
    def agent_factory(tools, domain_policy):
        return LangGraphAgent(
            tools=tools,
            domain_policy=domain_policy,
            model=args.model,
            temperature=0.0,
        )

    # Run evaluation
    summary = evaluate(
        agent_factory=agent_factory,
        domain="airline",
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        user_model=args.user_model,
    )

    return summary


if __name__ == "__main__":
    main()
