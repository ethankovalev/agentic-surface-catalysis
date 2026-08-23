"""
Worker agent construction.

A tool-bound LLM behind a system message.
"""

from functools import partial

from langchain_core.messages import SystemMessage


def _describe_tools(tools) -> str:
    return ", ".join(t.name for t in tools)


def _invoke_worker(llm_with_tools, system_text, messages):
    return llm_with_tools.invoke([SystemMessage(content=system_text)] + list(messages))


def create_agent(llm, tools, instructions: str):
    """Return a callable: pass it a message list, get back the model's
    response, tools already bound and a system message already attached.
    """
    system_text = f"{instructions}\n\nAvailable tools: {_describe_tools(tools)}."
    llm_with_tools = llm.bind_tools(tools)
    return partial(_invoke_worker, llm_with_tools, system_text)
