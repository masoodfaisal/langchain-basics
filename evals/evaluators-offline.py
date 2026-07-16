"""Server-safe offline evaluators uploaded and bound to the Chinook dataset.

LangSmith invokes an offline code evaluator as ``perform_eval(run, example)``.
The upload script appends an alias selecting one function below as that entry
point. This file intentionally uses only the Python standard library because
uploaded code evaluators run in LangSmith's restricted environment.
"""

from __future__ import annotations

from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _final_text(outputs: Any) -> str:
    if not isinstance(outputs, (dict, list)):
        return ""
    messages = outputs.get("messages", []) if isinstance(outputs, dict) else outputs
    for message in reversed(messages):
        content = _field(message, "content")
        role = _field(message, "type", _field(message, "role", ""))
        if str(role).lower() in {"ai", "assistant"} and content:
            return str(content)
    if isinstance(outputs, dict) and isinstance(outputs.get("answer"), str):
        return outputs["answer"]
    return ""


def _tool_calls(outputs: Any) -> list[dict[str, Any]]:
    if not isinstance(outputs, (dict, list)):
        return []
    messages = outputs.get("messages", []) if isinstance(outputs, dict) else outputs
    calls: list[dict[str, Any]] = []
    for message in messages:
        message_calls = _field(message, "tool_calls", [])
        if isinstance(message_calls, list):
            calls.extend(call for call in message_calls if isinstance(call, dict))
    return calls


def _run_outputs(run: Any) -> dict[str, Any]:
    outputs = _field(run, "outputs", {}) or {}
    return outputs if isinstance(outputs, dict) else {"messages": outputs}


def _reference_outputs(example: Any) -> dict[str, Any]:
    outputs = _field(example, "outputs", {}) or {}
    return outputs if isinstance(outputs, dict) else {}


def tool_trajectory(run: Any, example: Any) -> dict[str, Any]:
    """Pass when expected tools form a prefix and specified args match."""
    reference = _reference_outputs(example)
    expected_tools = reference.get("expected_tools", [])
    expected_args = reference.get("expected_tool_args", [])
    actual_calls = _tool_calls(_run_outputs(run))
    actual_names = [call.get("name") for call in actual_calls]

    if actual_names[: len(expected_tools)] != expected_tools:
        return {
            "key": "tool_trajectory",
            "score": 0,
            "comment": f"expected prefix {expected_tools!r}, got {actual_names!r}",
        }

    for index, specification in enumerate(expected_args):
        actual = actual_calls[index] if index < len(actual_calls) else {}
        expected_name = specification.get("name")
        if actual.get("name") != expected_name:
            return {
                "key": "tool_trajectory",
                "score": 0,
                "comment": (
                    f"call #{index} name mismatch: expected {expected_name!r}, "
                    f"got {actual.get('name')!r}"
                ),
            }
        for key, value in specification.get("args", {}).items():
            actual_value = actual.get("args", {}).get(key)
            if actual_value != value:
                return {
                    "key": "tool_trajectory",
                    "score": 0,
                    "comment": (
                        f"call #{index} arg {key!r} mismatch: expected "
                        f"{value!r}, got {actual_value!r}"
                    ),
                }

    return {
        "key": "tool_trajectory",
        "score": 1,
        "comment": f"trajectory matched: {actual_names}",
    }


def output_contains(run: Any, example: Any) -> dict[str, Any]:
    """Pass when all required substrings occur in the final answer."""
    required = _reference_outputs(example).get("expected_output_substrings", [])
    text = _final_text(_run_outputs(run)).lower()
    missing = [value for value in required if value.lower() not in text]
    return {
        "key": "output_contains",
        "score": 0 if missing else 1,
        "comment": (
            f"missing substrings: {missing}"
            if missing
            else "all required substrings present"
        ),
    }


def no_data_leak(run: Any, example: Any) -> dict[str, Any]:
    """Pass when no dataset-declared forbidden substring occurs."""
    forbidden = _reference_outputs(example).get("expected_output_forbidden", [])
    text = _final_text(_run_outputs(run)).lower()
    leaked = [value for value in forbidden if value.lower() in text]
    return {
        "key": "no_data_leak",
        "score": 0 if leaked else 1,
        "comment": (
            f"leaked forbidden substrings: {leaked}"
            if leaked
            else "no forbidden substrings present"
        ),
    }


def no_pii(run: Any, example: Any) -> dict[str, Any]:
    """Pass when the target's LangChain PIIMiddleware detectors found no PII."""
    del example
    raw_counts = _run_outputs(run).get("pii_counts", {})
    detected = {
        str(pii_type): int(count)
        for pii_type, count in raw_counts.items()
        if isinstance(count, int) and count > 0
    } if isinstance(raw_counts, dict) else {}
    return {
        "key": "no_pii",
        "score": 0 if detected else 1,
        "comment": (
            f"detected PII types and counts: {detected}"
            if detected
            else "LangChain PIIMiddleware detected no PII"
        ),
    }
