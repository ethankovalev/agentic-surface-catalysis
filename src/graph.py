"""
Supervisor graph.

Three workers, a supervisor that picks who acts next, and one deliberate
departure from the standard pattern: the supervisor cannot end the run.

In a plain supervisor graph the LLM decides FINISH. Here `exit_gate`
reads the validation record in Python and refuses to exit until every
check has actually run and passed. A model that talks itself into
"looks good to me" is precisely the failure this project exists to
catch, so the guarantee lives in code rather than in a prompt.
"""

import functools
import operator
import sys
from pathlib import Path
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import store
from src.prompt import (
    simulation_agent_prompt,
    structure_agent_prompt,
    validation_agent_prompt,
)
from src.tools import SIMULATION_TOOLS, STRUCTURE_TOOLS, VALIDATION_TOOLS


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    reaction_id: str
    attempts: int


members = ["Structure_Agent", "Simulation_Agent", "Validation_Agent"]
options = ["FINISH"] + members


system_prompt = f"""
<Role>
    You supervise a surface catalysis barrier calculation, delegating to
    these workers: {members}.

<Objective>
    Name the worker that should act next. Respond FINISH only when the
    barrier has been computed AND every validation check has passed.

<Workers>
    <Structure_Agent>
        Builds the metal slab, places the adsorbate, and constructs the
        dissociated final state.

    <Simulation_Agent>
        Relaxes the endpoints and runs the nudged elastic band.
        Always uses dispersion correction unless told otherwise.

    <Validation_Agent>
        Runs the physical checks and reports which passed.

<Order>
    Structures first, then simulation, then validation. If validation
    fails, route back to whoever can fix it:
      - dispersion or convergence failure -> Simulation_Agent
      - geometry or endpoint failure -> Structure_Agent

<Rules>
    Never respond FINISH before Validation_Agent has run.
    Never respond FINISH while any check is failing.
"""


class routeResponse(BaseModel):
    next: Literal[*options]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
            + "\nGiven the conversation above, who should act next? "
              "Or should we FINISH? Select one of: {options}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(options=str(options), members=", ".join(members))


def print_stream(s):
    if "messages" not in s:
        return   # a node-keyed chunk, not a full-state chunk - skip it
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()


def agent_node(state, agent, name):
    """Run one worker to completion, hand its last message back."""
    last_messages = None
    for s in agent.stream(state, {"recursion_limit": config.RECURSION_LIMIT},
                          stream_mode="values"):
        print_stream(s)
        if "messages" in s:
            last_messages = s["messages"]

    if not last_messages:
        return {"messages": [HumanMessage(content="(no output produced)",
                                          name=name)]}
    content = last_messages[-1].content
    return {"messages": [HumanMessage(content=content, name=name)]}


def exit_gate(state) -> str:
    """Where control actually goes. The supervisor only proposes.

    FINISH is honoured only when the store shows every check run and
    passed. Otherwise the run is sent to validation, or stopped once it
    has burned through its attempts.
    """
    choice = state.get("next", "FINISH")
    attempts = state.get("attempts", 0)

    if attempts >= config.MAX_ATTEMPTS:
        print(f"\n[gate] attempt limit ({config.MAX_ATTEMPTS}) reached, stopping")
        return END

    if choice != "FINISH":
        return choice

    if store.all_checks_passed():
        print("\n[gate] all checks passed, finishing")
        return END

    checks = store.validation()
    if not checks:
        print("\n[gate] supervisor said FINISH but nothing is validated "
              "-> Validation_Agent")
    else:
        failed = [k for k, ok in checks.items() if not ok]
        print(f"\n[gate] supervisor said FINISH but these failed: {failed}")
    return "Validation_Agent"


def create_graph(cfg: dict = None):
    cfg = cfg or config.as_dict()

    if "claude" not in cfg["LANGSIM_MODEL"]:
        raise ValueError("Only Anthropic models are wired up in this template.")

    llm = ChatAnthropic(
        model=cfg["LANGSIM_MODEL"],
        api_key=cfg["ANTHROPIC_API_KEY"],
        temperature=0.0,
    )

    def supervisor_agent(state):
        chain = prompt | llm.with_structured_output(routeResponse)
        result = chain.invoke(state)
        return {"next": result.next, "attempts": state.get("attempts", 0) + 1}

    structure_agent = create_react_agent(
        llm, tools=STRUCTURE_TOOLS, prompt=structure_agent_prompt
    )
    simulation_agent = create_react_agent(
        llm, tools=SIMULATION_TOOLS, prompt=simulation_agent_prompt
    )
    validation_agent = create_react_agent(
        llm, tools=VALIDATION_TOOLS, prompt=validation_agent_prompt
    )

    graph = StateGraph(AgentState)
    graph.add_node("Structure_Agent",
                   functools.partial(agent_node, agent=structure_agent,
                                     name="Structure_Agent"))
    graph.add_node("Simulation_Agent",
                   functools.partial(agent_node, agent=simulation_agent,
                                     name="Simulation_Agent"))
    graph.add_node("Validation_Agent",
                   functools.partial(agent_node, agent=validation_agent,
                                     name="Validation_Agent"))
    graph.add_node("Supervisor", supervisor_agent)

    for member in members:
        graph.add_edge(member, "Supervisor")

    conditional_map = {k: k for k in members}
    conditional_map[END] = END
    graph.add_conditional_edges("Supervisor", exit_gate, conditional_map)
    graph.add_edge(START, "Supervisor")

    return graph.compile(checkpointer=MemorySaver())
