"""Deterministic offline evaluators for ``simple-agent.py``.

These functions follow LangSmith's ``(run, example)`` evaluator interface.
They check the exact tool trajectory and simple answer requirements without
using another model. ``run_simple_agent_eval_offline.py`` uploads them as
reusable code evaluators and binds them to the simple-agent dataset.
"""

from __future__ import annotations

from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _run_outputs(run: Any) -> dict[str, Any]:
    outputs = _field(run, "outputs", {}) or {}
    return outputs if isinstance(outputs, dict) else {"messages": outputs}


def _reference_outputs(example: Any) -> dict[str, Any]:
    outputs = _field(example, "outputs", {}) or {}
    return outputs if isinstance(outputs, dict) else {}


def _final_text(outputs: dict[str, Any]) -> str:
    messages = outputs.get("messages", [])
    for message in reversed(messages):
        role = str(_field(message, "type", _field(message, "role", ""))).lower()
        content = _field(message, "content")
        if role in {"ai", "assistant"} and content:
            return str(content)
    answer = outputs.get("answer")
    return answer if isinstance(answer, str) else ""


def _tool_calls(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in outputs.get("messages", []):
        message_calls = _field(message, "tool_calls", [])
        if isinstance(message_calls, list):
            calls.extend(call for call in message_calls if isinstance(call, dict))
    return calls


def tool_trajectory(run: Any, example: Any) -> dict[str, Any]:
    """Pass when the agent makes exactly the expected tool calls."""
    reference = _reference_outputs(example)
    expected_tools = reference.get("expected_tools", [])
    expected_args = reference.get("expected_tool_args", [])
    actual_calls = _tool_calls(_run_outputs(run))
    actual_names = [call.get("name") for call in actual_calls]

    if actual_names != expected_tools:
        return {
            "key": "simple_agent_tool_trajectory",
            "score": 0,
            "comment": f"expected {expected_tools!r}, got {actual_names!r}",
        }

    for index, specification in enumerate(expected_args):
        actual = actual_calls[index]
        if actual.get("name") != specification.get("name"):
            return {
                "key": "simple_agent_tool_trajectory",
                "score": 0,
                "comment": f"tool name mismatch in call #{index + 1}",
            }
        actual_args = actual.get("args", {})
        for name, expected_value in specification.get("args", {}).items():
            if actual_args.get(name) != expected_value:
                return {
                    "key": "simple_agent_tool_trajectory",
                    "score": 0,
                    "comment": (
                        f"call #{index + 1} argument {name!r}: expected "
                        f"{expected_value!r}, got {actual_args.get(name)!r}"
                    ),
                }

    return {
        "key": "simple_agent_tool_trajectory",
        "score": 1,
        "comment": f"tool trajectory matched: {actual_names!r}",
    }


def output_contains(run: Any, example: Any) -> dict[str, Any]:
    """Pass when every required phrase appears in the final answer."""
    required = _reference_outputs(example).get("expected_output_substrings", [])
    text = _final_text(_run_outputs(run)).lower()
    missing = [value for value in required if value.lower() not in text]
    return {
        "key": "simple_agent_output_contains",
        "score": 0 if missing else 1,
        "comment": (
            f"missing required phrases: {missing}"
            if missing
            else "all required phrases are present"
        ),
    }


def output_excludes(run: Any, example: Any) -> dict[str, Any]:
    """Pass when no forbidden phrase appears in the final answer."""
    forbidden = _reference_outputs(example).get("expected_output_forbidden", [])
    text = _final_text(_run_outputs(run)).lower()
    found = [value for value in forbidden if value.lower() in text]
    return {
        "key": "simple_agent_output_excludes",
        "score": 0 if found else 1,
        "comment": (
            f"found forbidden phrases: {found}"
            if found
            else "no forbidden phrases are present"
        ),
    }


OFFLINE_EVALUATORS = [
    tool_trajectory,
    output_contains,
    output_excludes,
]
