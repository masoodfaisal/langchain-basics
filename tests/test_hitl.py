"""Tests for the human-in-the-loop resume payload helpers in ``hitl.py``.

The regression case drives a real ``HumanInTheLoopMiddleware`` interrupt over a
``write_file`` tool with a scripted model (no LLM tokens), then resumes with
``hitl.build_resume_command``. Before the fix, resuming with the bare string a
UI button produces raised ``TypeError: string indices must be integers`` inside
the middleware and the run never produced a final AI message.

Run with::

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_hitl.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import hitl


# ---------------------------------------------------------------------------
# Choice -> decision mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("choice", ["yes", "Y", "approve", "0", " ok "])
def test_approval_choices_map_to_approve_decision(choice):
    assert hitl.build_hitl_response(choice) == {"decisions": [{"type": "approve"}]}


@pytest.mark.parametrize("choice", ["no", "reject", "1"])
def test_rejection_choices_map_to_reject_decision(choice):
    assert hitl.build_hitl_response(choice) == {"decisions": [{"type": "reject"}]}


def test_reject_carries_optional_message():
    assert hitl.build_hitl_response("reject", message="not allowed") == {
        "decisions": [{"type": "reject", "message": "not allowed"}]
    }


def test_edit_requires_edited_action():
    with pytest.raises(ValueError, match="edited_action"):
        hitl.build_hitl_response("edit")


def test_unknown_choice_is_rejected():
    with pytest.raises(ValueError, match="Unknown HITL choice"):
        hitl.build_hitl_response("maybe")


def test_multiple_gated_calls_get_one_decision_each():
    response = hitl.build_hitl_response(["yes", "no"])
    assert response == {"decisions": [{"type": "approve"}, {"type": "reject"}]}


# ---------------------------------------------------------------------------
# Guard against the bare-string resume value that caused the crash
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["approve", "yes", "0", 0, None, ["approve"]])
def test_non_mapping_resume_values_are_rejected(value):
    with pytest.raises(TypeError, match="must be a mapping"):
        hitl.ensure_hitl_response(value)


def test_resume_value_without_decisions_is_rejected():
    with pytest.raises(TypeError, match="'decisions' list"):
        hitl.ensure_hitl_response({"type": "approve"})


def test_resume_command_is_keyed_by_interrupt_id():
    command = hitl.build_resume_command("int-1", "yes")
    assert command.resume == {"int-1": {"decisions": [{"type": "approve"}]}}


# ---------------------------------------------------------------------------
# End-to-end interrupt + resume over a HITL-gated write_file
# ---------------------------------------------------------------------------
FINAL_TEXT = "Saved the notes to /notes.txt."


class _ScriptedModel(BaseChatModel):
    """Model that calls ``write_file`` once, then answers with plain text."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "notes.txt", "content": "hello"},
                        "id": "call_write_file_1",
                    }
                ],
            )
        else:
            message = AIMessage(content=FINAL_TEXT)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _build_agent(tmp_path: Path):
    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file under the workspace."""
        target = tmp_path / path
        target.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    return create_agent(
        model=_ScriptedModel(),
        tools=[write_file],
        checkpointer=MemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_file": {"allowed_decisions": ["approve", "edit", "reject"]}
                }
            )
        ],
    )


def test_write_file_approval_resumes_and_finishes(tmp_path):
    agent = _build_agent(tmp_path)
    config = {"configurable": {"thread_id": "hitl-approve"}}

    paused = agent.invoke(
        {"messages": [{"role": "user", "content": "Save my notes"}]}, config=config
    )
    interrupts = paused["__interrupt__"]
    assert len(interrupts) == 1

    resumed = agent.invoke(
        hitl.build_resume_command(interrupts[0].id, "yes"), config=config
    )

    assert "__interrupt__" not in resumed
    final = resumed["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content == FINAL_TEXT
    assert (tmp_path / "notes.txt").read_text() == "hello"


def test_bare_string_resume_still_crashes_the_middleware(tmp_path):
    """Documents the original failure mode the helpers exist to prevent."""
    from langgraph.types import Command

    agent = _build_agent(tmp_path)
    config = {"configurable": {"thread_id": "hitl-bare-string"}}

    agent.invoke(
        {"messages": [{"role": "user", "content": "Save my notes"}]}, config=config
    )

    with pytest.raises(TypeError):
        agent.invoke(Command(resume="approve"), config=config)
