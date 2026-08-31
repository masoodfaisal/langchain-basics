"""A thin LangGraph wrapper around the existing remote ``agent`` graph."""

import os

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.pregel.remote import RemoteGraph


remote_agent = RemoteGraph(
    "agent",
    url=os.environ["REMOTE_GRAPH_URL"],
    api_key=os.environ["LANGSMITH_API_KEY"],
)

builder = StateGraph(MessagesState)
builder.add_node("remote_agent", remote_agent)
builder.add_edge(START, "remote_agent")
builder.add_edge("remote_agent", END)

graph = builder.compile()
