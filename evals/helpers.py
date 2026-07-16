"""Shared helpers for preparing Chinook evaluation inputs and outputs."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import PIIMiddleware


PII_TYPES = ("email", "credit_card", "ip", "mac_address", "url")
PII_DETECTORS = {
    pii_type: PIIMiddleware(
        pii_type,
        apply_to_input=False,
        apply_to_output=True,
    ).detector
    for pii_type in PII_TYPES
}


def final_text(outputs: dict[str, Any]) -> str:
    """Extract the final assistant text from an agent output."""
    messages = outputs.get("messages", [])
    if not messages and isinstance(outputs, list):
        messages = outputs
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        role = getattr(message, "type", None) or (
            message.get("type", message.get("role"))
            if isinstance(message, dict)
            else None
        )
        if role in {"ai", "assistant"} and content:
            return str(content)
    return ""


def pii_counts(text: str) -> dict[str, int]:
    """Count PII using the detectors built into LangChain PIIMiddleware."""
    return {
        pii_type: len(matches)
        for pii_type, detector in PII_DETECTORS.items()
        if (matches := detector(text))
    }
