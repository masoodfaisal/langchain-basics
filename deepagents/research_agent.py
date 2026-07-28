"""Shared research agent used by Module 2 (Deep Agents) and Module 4 (LangSmith).

This is the minimal useful Deep Agent: a research subagent + Tavily search + a
checkpointer. It deliberately omits HITL and FilesystemBackend so that
evaluation runs in Module 4 don't pause or leak files to disk.

Module 2 builds up to this agent step-by-step in the notebook. This file
packages the same pattern so Module 4 can import it directly:

    from agents.research_agent import build_research_agent
    agent = build_research_agent()
"""

## Load in langgrpah.json via     // "agent": "./deepagents/research_agent.py:build_runtime_research_agent"
#
from datetime import datetime
from pathlib import Path
import os
import sys

# LangGraph loads this module directly from its file path, so Python does not
# automatically add this file's directory to the import path. Make the sibling
# ``utils`` package importable regardless of the directory LangGraph starts in.
_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from deepagents import FilesystemPermission, RubricMiddleware, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.context_hub import ContextHubBackend
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from utils.models import model
from utils.search import resilient_tavily_search


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: Search query to execute.
    """
    # `resilient_tavily_search` retries on Tavily failure with linear backoff,
    # then falls back to a topic-matched canned response so a flaky network
    # doesn't break the workshop. See utils/search.py for details.
    return resilient_tavily_search(query, max_retries=2)


research_model: BaseChatModel = init_chat_model(
    "openai:gpt-4.1-mini",
    temperature=0,
)

# This app shares a LangSmith project with other agents, so the root run needs
# a stable name plus bounded-cardinality metadata for run rules to target it.
APP_NAME = "asx_financial_analyst"
ENVIRONMENT = os.getenv("APP_ENVIRONMENT", "development")
TRACING_CONFIG = {
    "run_name": APP_NAME,
    "metadata": {"app": APP_NAME, "environment": ENVIRONMENT},
}


def build_research_agent():
    """Return a fresh research deep agent.

    Each call returns a new agent with a fresh checkpointer — useful so eval
    runs don't share state with each other.
    """
    research_subagent = {
        "name": "research-agent",
        "description": "Delegate research tasks. Give one topic at a time.",
        # "model": research_model,
        "system_prompt": (
            f"You are a research assistant. Today is "
            f"{datetime.now().strftime('%Y-%m-%d')}.\n"
            "Use tools to gather information. Limit to 3 search calls."
        ),
        "tools": [tavily_search],
    }

    return create_deep_agent(
        model=model,
        tools=[tavily_search],
        system_prompt=(
            "You are a helpful research assistant. "
            "Delegate research to the research-agent."
        ),
        subagents=[research_subagent],
        checkpointer=MemorySaver(),
        name=APP_NAME,
    ).with_config(TRACING_CONFIG)


def build_research_agent_with_context(
    context_repo: str = "-/research-agent-context",
):
    """Return a research agent that loads instructions from Context Hub.

    ``context_repo`` identifies a LangSmith Context Hub agent repository, for
    example ``"my-org/research-agent-context"``. Its ``AGENTS.md`` file is
    mounted at ``/context/AGENTS.md`` and loaded into the agent's memory.

    All other files use the ephemeral state backend. Context Hub is exposed as
    read-only to the agent's filesystem tools so reports and scratch work
    cannot accidentally create commits in the context repository.

    Rubric-based revision is opt-in for each run. Pass a top-level ``rubric``
    alongside ``messages`` when invoking the graph, for example::

        agent.invoke({
            "messages": [{"role": "user", "content": "Research topic X"}],
            "rubric": (
                "The answer directly addresses the topic; "
                "important claims cite reliable sources; "
                "uncertainties and conflicting evidence are explicit."
            ),
        })

    When a rubric is supplied, the grader can request revisions up to three
    times. Without one, the middleware is a no-op.
    """
    research_subagent = {
        "name": "research-agent",
        "description": "Delegate research tasks. Give one topic at a time.",
        "system_prompt": (
            f"You are a research assistant. Today is "
            f"{datetime.now().strftime('%Y-%m-%d')}.\n"
            "Use tools to gather information. Limit to 3 search calls."
        ),
        "tools": [tavily_search],
    }

    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/context/": ContextHubBackend(context_repo)},
    )

    RUBRIC_PROMPT = """
                The answer directly addresses the topic;
                important claims cite reliable sources;
                uncertainties and conflicting evidence are explicit.
    """
    return create_deep_agent(
        model=model,
        tools=[tavily_search],
        system_prompt=(
            "You are a helpful research assistant. "
            "Delegate research to the research-agent."
        ),
        memory=["/context/AGENTS.md"],
        subagents=[research_subagent],
        backend=backend,
        middleware=[
            RubricMiddleware(
                model=research_model,
                max_iterations=3,
                system_prompt= RUBRIC_PROMPT
            )
        ],
        permissions=[
            FilesystemPermission(
                operations=["write"],
                paths=["/context/**"],
                mode="deny",
            )
        ],
        checkpointer=MemorySaver(),
        name=APP_NAME,
    ).with_config(TRACING_CONFIG)


def build_runtime_research_agent():
    """Return the context-enabled graph from a zero-argument runtime factory.

    LangGraph treats a one-argument graph factory as accepting a
    ``RunnableConfig``. Keeping this wrapper parameterless prevents the
    runtime config from being mistaken for ``context_repo``.
    """
    return build_research_agent_with_context()
