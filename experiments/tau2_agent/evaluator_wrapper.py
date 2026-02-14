"""
OpenEvolve evaluator wrapper for tau2 agent evaluation.

This file is used as the evaluator for OpenEvolve (file-based evaluator).
OpenEvolve will call the evaluate() function with a program_path.
"""
import os
import sys
from pathlib import Path

# Set up paths before any imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

# Import our evaluator
from evaluate import evaluate as tau2_evaluate

# Configuration (can be overridden via environment variables)
NUM_TASKS = int(os.environ.get("TAU2_NUM_TASKS", "50"))
MAX_CONCURRENCY = int(os.environ.get("TAU2_MAX_CONCURRENCY", "25"))
RUN_MAST = os.environ.get("TAU2_RUN_MAST", "false").lower() == "true"


def evaluate(program_path: str):
    """
    Evaluate an agent program on tau2.

    This is the entry point called by OpenEvolve.
    """
    return tau2_evaluate(
        program_path,
        num_tasks=NUM_TASKS,
        max_concurrency=MAX_CONCURRENCY,
        run_mast=RUN_MAST,
    )


# For cascade evaluation
def evaluate_stage1(program_path: str):
    """Quick validation with fewer tasks."""
    return tau2_evaluate(
        program_path,
        num_tasks=5,
        max_concurrency=5,
        run_mast=False,
    )


def evaluate_stage2(program_path: str):
    """Full evaluation."""
    return evaluate(program_path)
