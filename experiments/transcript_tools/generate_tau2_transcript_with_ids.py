"""Generate tau2 transcript text with stable IDs for messages, tool calls, and tool results."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ID-annotated transcript for one tau2 task.",
    )
    parser.add_argument(
        "sim_file", type=Path, help="Path to tau2 simulations JSON file"
    )
    parser.add_argument("task_id", type=str, help="Task id to extract")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output transcript path"
    )
    return parser.parse_args()


def clean_user_content(content: str) -> str:
    return (
        content.replace("###STOP###", "")
        .replace("###TRANSFER###", "")
        .replace("###OUT-OF-SCOPE###", "")
        .strip()
    )


def format_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        return ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return str(arguments)


def format_tool_result_content(content: Any) -> str:
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2)
    if content is None:
        return ""
    return str(content)


def build_transcript(messages: list[dict[str, Any]], task_id: str) -> str:
    lines: list[str] = []
    lines.append(f"# Conversation Transcript -- Task {task_id}")
    lines.append("")

    message_counter = 0
    tool_call_counter = 0
    tool_result_counter = 0
    source_call_id_to_local_id: dict[str, str] = {}

    for msg in messages:
        message_counter += 1
        message_id = f"m_{message_counter:04d}"

        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        if role == "user":
            lines.append(f"[{message_id}] **User:**")
            lines.append(clean_user_content(str(content or "")))
            lines.append("")
            continue

        if role == "assistant":
            lines.append(f"[{message_id}] **Agent:**")
            if content:
                lines.append(str(content))
                lines.append("")
            if tool_calls:
                for tool_call in tool_calls:
                    tool_call_counter += 1
                    tool_call_id = f"tc_{tool_call_counter:04d}"
                    source_tool_call_id = tool_call.get("id")
                    if source_tool_call_id:
                        source_call_id_to_local_id[str(source_tool_call_id)] = (
                            tool_call_id
                        )

                    args_str = format_tool_arguments(tool_call.get("arguments", {}))
                    lines.append(
                        f"[{tool_call_id}] > Tool call: `{tool_call['name']}({args_str})`"
                    )
                    if source_tool_call_id:
                        lines.append(
                            f"[{tool_call_id}] > Source tool_call_id: `{source_tool_call_id}`"
                        )
                lines.append("")
            continue

        if role == "tool":
            tool_result_counter += 1
            tool_result_id = f"tr_{tool_result_counter:04d}"
            source_tool_call_id = msg.get("id")
            linked_tool_call_id = ""
            if source_tool_call_id:
                linked_tool_call_id = source_call_id_to_local_id.get(
                    str(source_tool_call_id), ""
                )

            header = f"[{message_id}] [{tool_result_id}] > Tool result"
            if linked_tool_call_id:
                header += f" for [{linked_tool_call_id}]"
            if source_tool_call_id:
                header += f" (source tool_call_id=`{source_tool_call_id}`)"
            header += ":"
            lines.append(header)

            formatted_result = format_tool_result_content(content)
            lines.append("> ```json")
            for line in formatted_result.split("\n"):
                lines.append(f"> {line}")
            lines.append("> ```")
            lines.append("")
            continue

        lines.append(f"[{message_id}] **{role}:**")
        if content:
            lines.append(str(content))
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data = json.loads(args.sim_file.read_text())
    simulations = data.get("simulations", [])
    simulation = next(sim for sim in simulations if str(sim["task_id"]) == args.task_id)
    transcript = build_transcript(simulation["messages"], args.task_id)

    output_path = args.output or Path(f"task_{args.task_id}_transcript_with_ids.txt")
    output_path.write_text(transcript)
    print(f"Wrote transcript: {output_path}")


if __name__ == "__main__":
    main()
