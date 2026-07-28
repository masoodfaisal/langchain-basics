"""LangSmith run metadata for the research deep agent.

The research agent is multi-turn and pauses for human-in-the-loop approval, so a
single conversation spans several root runs. LangSmith groups those runs into one
thread only when every root run carries a ``thread_id`` in its metadata, and
filtering per end-user / per deployment needs ``user_id`` and ``environment``
there too.

Usage:

    from utils.tracing import TracingMetadataMiddleware

    create_deep_agent(..., middleware=[TracingMetadataMiddleware()])

Callers still pass the conversation id the usual way — it is read from
``configurable``, the same place the checkpointer reads it from:

    agent.invoke(
        {"messages": [...]},
        config={"configurable": {"thread_id": "conv-42", "user_id": "u-7"}},
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.config import get_config
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)

DEFAULT_ENVIRONMENT = "development"


def tracing_environment() -> str:
    """Return the environment label to stamp on root runs."""
    return (
        os.getenv("LANGSMITH_ENVIRONMENT")
        or os.getenv("ENVIRONMENT")
        or DEFAULT_ENVIRONMENT
    )


def _resolve_metadata(config: RunnableConfig) -> dict[str, Any]:
    """Collect the thread / user / environment metadata for the current run."""
    configurable = config.get("configurable") or {}
    existing = config.get("metadata") or {}

    metadata: dict[str, Any] = {"environment": tracing_environment()}
    for key in ("thread_id", "user_id"):
        value = configurable.get(key) or existing.get(key)
        if value is not None:
            metadata[key] = str(value)

    if "thread_id" not in metadata:
        logger.warning(
            "No thread_id in config; this run will not group into a LangSmith "
            "thread. Pass configurable.thread_id when invoking the agent."
        )
    return metadata


def _annotate_root_run() -> None:
    """Stamp thread / user / environment metadata onto the trace's root run."""
    run_tree = get_current_run_tree()
    if run_tree is None:
        return

    config = get_config()
    metadata = _resolve_metadata(config)
    run_tree.add_metadata(metadata)

    # The root run belongs to the caller (LangGraph server, notebook, eval), so
    # the only handle on it from inside the graph is the tracer's in-flight run
    # map. Mutating it there means the metadata ships with the root run's patch.
    handlers = getattr(config.get("callbacks"), "handlers", None) or []
    for handler in handlers:
        if not isinstance(handler, LangChainTracer):
            continue
        root_run = handler.run_map.get(str(run_tree.trace_id))
        if root_run is None:
            continue
        root_run.extra.setdefault("metadata", {}).update(metadata)


class TracingMetadataMiddleware(AgentMiddleware):
    """Add thread_id / user_id / environment metadata to every root run."""

    def before_agent(self, state, runtime) -> None:
        self._annotate()

    async def abefore_agent(self, state, runtime) -> None:
        self._annotate()

    def _annotate(self) -> None:
        try:
            _annotate_root_run()
        except Exception:  # tracing metadata must never fail a run
            logger.debug("Could not attach tracing metadata", exc_info=True)
