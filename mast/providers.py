import json
import os
from typing import Any, cast

from anthropic import AnthropicVertex
from openai import OpenAI

from .contracts import STRUCTURED_RESPONSE_SCHEMA

_ANTHROPIC_TOOL_NAME = "submit_mast_evaluation"


def call_openai(
    prompt: str,
    api_key: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Call OpenAI API with structured JSON schema output."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        reasoning={"effort": "high"},
        text={
            "format": {
                "type": "json_schema",
                "name": "mast_evaluation",
                "schema": STRUCTURED_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    )

    response_any: Any = response
    output_text = getattr(response_any, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("OpenAI response missing structured output text")

    payload = json.loads(output_text)
    if not isinstance(payload, dict):
        raise ValueError("OpenAI structured output is not a JSON object")

    return payload, output_text


def call_anthropic_vertex(
    prompt: str,
    project_id: str | None = None,
    region: str = "us-east5",
) -> tuple[dict[str, Any], str]:
    """Call Anthropic Claude via Vertex AI with a forced tool schema."""
    resolved_project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if resolved_project_id is None:
        raise ValueError("GOOGLE_CLOUD_PROJECT must be set for vertex model")

    client = AnthropicVertex(
        project_id=cast(str, resolved_project_id),
        region=region,
    )

    response = client.messages.create(
        model="claude-opus-4-6@default",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "name": _ANTHROPIC_TOOL_NAME,
                "description": "Submit MAST evaluation in structured JSON format.",
                "input_schema": STRUCTURED_RESPONSE_SCHEMA,
            }
        ],
        tool_choice={"type": "tool", "name": _ANTHROPIC_TOOL_NAME},
    )

    response_any: Any = response
    content_blocks = getattr(response_any, "content", None)
    if not isinstance(content_blocks, list):
        raise ValueError("Anthropic response missing content blocks")

    for block in content_blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _ANTHROPIC_TOOL_NAME:
            continue
        tool_input = getattr(block, "input", None)
        if isinstance(tool_input, dict):
            return tool_input, json.dumps(tool_input)

    raise ValueError("Anthropic response missing structured tool output")
