"""Verify short-term memory via LangGraph checkpointing.

Uses a deterministic ``GenericFakeChatModel`` so the test is offline,
free, and stable. The model is scripted to answer turn 1 with a fact and
turn 2 with a summary that quotes whatever came before it.

With a checkpointer and the same ``thread_id``:
    - turn 1 stores the exchange
    - turn 2 starts from the saved state and sees turn 1's messages

With a different ``thread_id``:
    - turn 2 starts from a fresh state and cannot see turn 1

The agent is driven via ``ainvoke`` (the async path) to match the rest of
the project. ``pytest-asyncio`` runs every ``async def test_*`` here
automatically (``asyncio_mode = "auto"`` in ``pyproject.toml``).
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver


def _build_agent():
    """Agent with no tools, scripted two-turn replies, and a checkpointer."""
    scripted = iter([
        AIMessage(content="My favourite genre is Jazz."),
        AIMessage(content="You told me your favourite genre is Jazz."),
        AIMessage(content="I do not know your favourite genre."),
    ])
    model = GenericFakeChatModel(messages=scripted)
    return create_agent(
        model,
        tools=[],
        system_prompt="You are a test model.",
        checkpointer=InMemorySaver(),
    )


async def test_same_thread_id_preserves_prior_turn():
    agent = _build_agent()
    config = {"configurable": {"thread_id": "demo-thread"}}

    # Turn 1: seed the fact.
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "My favourite genre is Jazz."}]},
        config,
    )

    # Turn 2: same thread, the model's scripted reply should be surfaced.
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is my favourite genre?"}]},
        config,
    )
    state = await agent.aget_state(config)

    final = result["messages"][-1].content
    assert "Jazz" in final, f"expected memory to surface Jazz, got: {final!r}"

    # Sanity: the saved state accumulates across turns.
    roles = [m.type for m in state.values["messages"]]
    # Expect both user turns + both assistant replies.
    assert roles.count("human") == 2
    assert roles.count("ai") == 2


async def test_different_thread_id_starts_fresh():
    agent = _build_agent()

    # Turn 1 on thread A.
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "My favourite genre is Jazz."}]},
        {"configurable": {"thread_id": "thread-a"}},
    )

    # Turn 2 on thread B: must NOT see the prior fact.
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is my favourite genre?"}]},
        {"configurable": {"thread_id": "thread-b"}},
    )
    state_b = await agent.aget_state({"configurable": {"thread_id": "thread-b"}})

    final = result["messages"][-1].content
    # The GenericFakeChatModel keeps advancing through its message list;
    # regardless of which scripted response we get, thread-b's state
    # must not contain anything from thread-a.
    prior_human_contents = [
        m.content for m in state_b.values["messages"] if m.type == "human"
    ]
    assert "My favourite genre is Jazz." not in prior_human_contents, (
        f"thread-b leaked thread-a's state: {prior_human_contents!r}"
    )
    # And thread-b should have exactly the one question we just sent.
    assert prior_human_contents == ["What is my favourite genre?"], (
        f"expected one user message in thread-b, got {prior_human_contents!r}"
    )
    # The final reply is whatever the scripted model returned; we just
    # assert the agent produced an AI message and did not crash.
    assert isinstance(final, str) and final
