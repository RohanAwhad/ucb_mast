"""Run mast-evaluator on one transcript file or a directory of transcripts."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

from mast import ModelName, evaluate
from mast.models import EvaluationResult
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAST evaluator on a transcript file or directory."
    )
    parser.add_argument(
        "transcript_path",
        type=Path,
        help="Path to transcript file or directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "vertex"],
        default="openai",
        help="Analysis model to use (default: openai)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum concurrent model calls for directory mode (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mast_analysis"),
        help="Output directory for per-file analyses in directory mode",
    )
    return parser.parse_args()


def resolve_model(model_name: str) -> ModelName:
    if model_name == "vertex":
        return cast(ModelName, "vertex/claude-opus-4-6")
    return cast(ModelName, "openai/gpt-5.2")


def load_trace(path: Path) -> str:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            payload = data.get("trajectory", data)
        else:
            payload = data
        return json.dumps(payload, indent=2)
    return path.read_text()


def format_result(result: EvaluationResult) -> str:
    lines: list[str] = []
    lines.append("--- A. Summary ---")
    lines.append(result.summary)
    lines.append("")
    lines.append("--- B. Task Completed ---")
    lines.append(str(result.task_completed))
    lines.append("")
    lines.append("--- C. Detected Failure Modes (with definitions) ---")
    detected = result.failure_modes.get_detected()
    if not detected:
        lines.append("None")
    else:
        for fm in detected:
            evidence_map = result.failure_mode_evidence.get(fm["code"], {})
            if evidence_map:
                evidence_parts = [
                    f"{evidence_id}: {reason}"
                    for evidence_id, reason in evidence_map.items()
                ]
                evidence_text = "; ".join(evidence_parts)
            else:
                evidence_text = ""
            lines.append(f"[{fm['code']}] {fm['name']}: {{{evidence_text}}}")
            lines.append(f"  {fm['definition']}")
    lines.append("")
    lines.append("--- Raw Model Response ---")
    lines.append(result.raw_response.strip())
    return "\n".join(lines) + "\n"


def evaluate_file(path: Path, model: ModelName) -> EvaluationResult:
    trace = load_trace(path)
    return evaluate([trace], model_name=model)[0]


async def evaluate_file_with_semaphore(
    path: Path,
    model: ModelName,
    semaphore: asyncio.Semaphore,
) -> EvaluationResult:
    async with semaphore:
        return await asyncio.to_thread(evaluate_file, path, model)


async def analyze_directory(
    directory: Path,
    model: ModelName,
    output_dir: Path,
    max_concurrency: int,
) -> None:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_and_write(path: Path) -> Path:
        result = await evaluate_file_with_semaphore(path, model, semaphore)
        output_path = output_dir / path.name
        output_path.write_text(format_result(result))
        return output_path

    tasks = [asyncio.create_task(analyze_and_write(path)) for path in files]
    with tqdm(total=len(tasks), desc="Analyzing transcripts", unit="file") as progress:
        for task in asyncio.as_completed(tasks):
            await task
            progress.update(1)


async def main() -> None:
    args = parse_args()
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")

    input_path: Path = args.transcript_path
    model = resolve_model(args.model)

    print("=" * 70)
    print(f"MAST Evaluator (model: {model})")
    print("=" * 70)

    if input_path.is_dir():
        await analyze_directory(
            directory=input_path,
            model=model,
            output_dir=args.output_dir,
            max_concurrency=args.max_concurrency,
        )
        print(f"\nSaved analyses to: {args.output_dir}")
        return

    result = await evaluate_file_with_semaphore(
        path=input_path,
        model=model,
        semaphore=asyncio.Semaphore(1),
    )
    print(format_result(result))


if __name__ == "__main__":
    asyncio.run(main())
