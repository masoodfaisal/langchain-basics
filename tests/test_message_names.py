"""Regression tests for message ``name`` sanitization.

OpenAI rejects any ``messages[N].name`` that does not match
``^[^\\s<|\\\\/>]+$``. Because ``create_agent(name=...)`` stamps that name onto
every AIMessage, a spaced display name only fails on the *second* turn, when
the checkpointed AIMessage is replayed to the model. These tests drive a fake
model that applies the same validation OpenAI does, so the failure mode is
reproduced without network access or tokens.

Run with::

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_message_names.py -v
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from agent import AGENT_NAME
from middleware import sanitize_message_names, sanitize_name

# The pattern OpenAI enforces on messages[N].name.
OPENAI_NAME_PATTERN = re.compile(r"^[^\s<|\\/>]+$")
SPACED_DISPLAY_NAME = "Chinook Support Agent"


class NameValidatingModel(BaseChatModel):
    """Fake chat model that 400s on an invalid message ``name``, like OpenAI does."""

    replies: list[str]
    seen_names: list[Any] = Field(default_factory=list)
    turns: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "name-validating-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        for index, message in enumerate(messages):
            name = getattr(message, "name", None)
            self.seen_names.append(name)
            if name is not None and not OPENAI_NAME_PATTERN.match(name):
                raise ValueError(
                    f"400 invalid_request_error: messages[{index}].name "
                    f"{name!r} does not match '^[^\\s<|\\\\/>]+$'"
                )
        reply = self.replies[min(len(self.turns), len(self.replies) - 1)]
        self.turns.append(1)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=reply))]
        )


def _build_agent(model: NameValidatingModel, *, middleware):
    return create_agent(
        model,
        [],
        name=SPACED_DISPLAY_NAME,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


async def _two_turns(agent) -> str:
    config = {"configurable": {"thread_id": "name-regression"}}
    await agent.ainvoke({"messages": [HumanMessage("I want a jazz album")]}, config)
    second = await agent.ainvoke(
        {"messages": [HumanMessage("Order the first one")]}, config
    )
    return str(second["messages"][-1].content)


# ---------------------------------------------------------------------------
# 1. sanitize_name coerces (or drops) names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Chinook Support Agent", "Chinook_Support_Agent"),
        ("chinook_support_agent", "chinook_support_agent"),
        ("a/b\\c|d<e>f", "a_b_c_d_e_f"),
        ("   ", None),
        ("<<>>", None),
        (None, None),
    ],
)
def test_sanitize_name(raw, expected):
    assert sanitize_name(raw) == expected
    if expected is not None:
        assert OPENAI_NAME_PATTERN.match(expected)


# ---------------------------------------------------------------------------
# 2. The canonical agent identifier is wire-safe as-is
# ---------------------------------------------------------------------------
def test_agent_name_is_whitespace_free():
    assert OPENAI_NAME_PATTERN.match(AGENT_NAME)


# ---------------------------------------------------------------------------
# 3. Two turns with a spaced display name still answer the user
# ---------------------------------------------------------------------------
async def test_second_turn_survives_spaced_display_name():
    model = NameValidatingModel(replies=["Try Kind of Blue.", "Order confirmed."])
    agent = _build_agent(model, middleware=[sanitize_message_names])

    answer = await _two_turns(agent)

    assert answer == "Order confirmed."
    assert all(
        name is None or OPENAI_NAME_PATTERN.match(name)
        for name in model.seen_names
    ), f"invalid name reached the model: {model.seen_names!r}"


# ---------------------------------------------------------------------------
# 4. Without the middleware the same conversation 400s (guards the guard)
# ---------------------------------------------------------------------------
async def test_second_turn_without_sanitizer_reproduces_the_400():
    model = NameValidatingModel(replies=["Try Kind of Blue.", "Order confirmed."])
    agent = _build_agent(model, middleware=[])

    with pytest.raises(ValueError, match="invalid_request_error"):
        await _two_turns(agent)
