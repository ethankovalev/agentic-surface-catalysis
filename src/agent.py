"""
Agent construction helper.

Binds a set of tools to an LLM behind a system prompt. The tool names go
into the prompt so the model knows what it can reach for; everything
else it learns from the tool docstrings.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_agent(llm, tools, prompt_content: str, system_message: str = ""):
    """Return a prompt-and-tools chain ready to invoke."""
    tools_name = [t.name for t in tools]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{prompt_content}\n\nTools available: {tools_name}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join(tools_name))
    return prompt | llm.bind_tools(tools)
