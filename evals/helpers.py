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
            return _content_text(message, content)
    return ""


def _content_text(message: Any, content: Any) -> str:
    """Return plain text from string or content-block message content.

    The Responses API returns ``content`` as a list of blocks (reasoning,
    text, ...). Prefer the message's ``text`` property, which joins only
    the text blocks, and fall back to joining text blocks by hand for
    plain dict messages.
    """
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") == "text")
        )
    return str(content)


def pii_counts(text: str) -> dict[str, int]:
    """Count PII using the detectors built into LangChain PIIMiddleware."""
    return {
        pii_type: len(matches)
        for pii_type, detector in PII_DETECTORS.items()
        if (matches := detector(text))
    }
