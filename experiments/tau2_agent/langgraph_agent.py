"""LangGraph-based agent for tau2 evaluation."""
import json
import uuid
from typing import Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
)
from tau2.environment.tool import Tool


class LangGraphAgentState(BaseModel):
    """State for the LangGraph agent."""

    messages: list[dict]  # OpenAI-format messages
    system_prompt: str


def convert_tau2_tools_to_openai(tools: list[Tool]) -> list[dict]:
    """Convert tau2 Tool objects to OpenAI tool format."""
    return [tool.openai_schema for tool in tools]


def convert_tau2_message_to_openai(message: ValidAgentInputMessage) -> list[dict]:
    """Convert tau2 message to OpenAI message format."""
    from tau2.data_model.message import ToolMessage

    if isinstance(message, MultiToolMessage):
        # Multiple tool results
        return [
            {
                "role": "tool",
                "tool_call_id": tm.id,  # ToolMessage uses 'id' not 'tool_call_id'
                "content": tm.content or "",
            }
            for tm in message.tool_messages
        ]
    elif isinstance(message, ToolMessage):
        # Single tool result
        return [{
            "role": "tool",
            "tool_call_id": message.id,  # ToolMessage uses 'id' not 'tool_call_id'
            "content": message.content or "",
        }]
    else:
        # User message
        return [{
            "role": "user",
            "content": message.content or "",
        }]


class LangGraphAgent(LocalAgent[LangGraphAgentState]):
    """
    A LangGraph-based agent for tau2 evaluation.

    This wraps a simple ReAct-style agent loop using LangGraph.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model_name = model
        self.temperature = temperature
        self.openai_tools = convert_tau2_tools_to_openai(tools)

        # Build the LangGraph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for the agent."""

        # Simple single-node graph that calls the LLM
        def call_llm(state: dict) -> dict:
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
            )

            messages = [{"role": "system", "content": state["system_prompt"]}]
            messages.extend(state["messages"])

            # Bind tools to the LLM
            if self.openai_tools:
                llm_with_tools = llm.bind_tools(self.openai_tools)
            else:
                llm_with_tools = llm

            response = llm_with_tools.invoke(messages)

            # Convert response to dict format
            if response.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            else:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content,
                }

            return {
                "messages": state["messages"] + [assistant_msg],
                "system_prompt": state["system_prompt"],
                "last_response": assistant_msg,
            }

        # Build graph
        workflow = StateGraph(dict)
        workflow.add_node("agent", call_llm)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)

        return workflow.compile()

    @property
    def system_prompt(self) -> str:
        # Use tau2's exact system prompt
        return SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=self.domain_policy,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LangGraphAgentState:
        """Get the initial state of the agent."""
        messages = []
        if message_history:
            for msg in message_history:
                if hasattr(msg, "role"):
                    if msg.role == "assistant":
                        if msg.tool_calls:
                            messages.append({
                                "role": "assistant",
                                "content": msg.content,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.name,
                                            "arguments": json.dumps(tc.arguments),
                                        },
                                    }
                                    for tc in msg.tool_calls
                                ],
                            })
                        else:
                            messages.append({
                                "role": "assistant",
                                "content": msg.content,
                            })
                    elif msg.role == "user":
                        messages.append({
                            "role": "user",
                            "content": msg.content,
                        })
                    elif msg.role == "tool":
                        messages.append({
                            "role": "tool",
                            "tool_call_id": msg.id,  # ToolMessage uses 'id' not 'tool_call_id'
                            "content": msg.content,
                        })

        return LangGraphAgentState(
            messages=messages,
            system_prompt=self.system_prompt,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LangGraphAgentState
    ) -> tuple[AssistantMessage, LangGraphAgentState]:
        """Generate the next message using LangGraph."""

        # Convert incoming message to OpenAI format and add to state
        new_messages = convert_tau2_message_to_openai(message)
        current_messages = list(state.messages) + new_messages

        # Run the graph
        graph_state = {
            "messages": current_messages,
            "system_prompt": state.system_prompt,
        }
        result = self._graph.invoke(graph_state)

        # Extract the last response
        last_response = result.get("last_response", result["messages"][-1])

        # Convert to tau2 AssistantMessage
        tool_calls = None
        content = last_response.get("content")

        if "tool_calls" in last_response and last_response["tool_calls"]:
            tool_calls = []
            for tc in last_response["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(ToolCall(
                    id=tc.get("id", str(uuid.uuid4())),
                    name=func.get("name", ""),
                    arguments=args,
                    requestor="assistant",
                ))
            # When making tool calls, content should be None for tau2
            content = None

        assistant_message = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

        # Update state
        new_state = LangGraphAgentState(
            messages=result["messages"],
            system_prompt=state.system_prompt,
        )

        return assistant_message, new_state
