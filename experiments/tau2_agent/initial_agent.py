"""
Initial agent program for OpenEvolve evolution.

This agent will be evolved to improve performance on tau2-airline benchmark.
Must define: create_agent(tools, domain_policy) -> LocalAgent
"""
from tau2.agent.llm_agent import LLMAgent
from tau2.environment.tool import Tool


def create_agent(tools: list[Tool], domain_policy: str):
    """
    Create an agent for tau2 evaluation.

    Args:
        tools: List of available tools from the environment
        domain_policy: The policy/instructions for this domain

    Returns:
        A LocalAgent instance
    """
    return LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm="gpt-4.1-mini",
        llm_args={"temperature": 0.0},
    )
