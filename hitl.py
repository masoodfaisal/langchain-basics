"""Resume payloads for human-in-the-loop interrupts.

``HumanInTheLoopMiddleware`` pauses the graph with a ``HITLRequest`` and reads
the resume value as ``interrupt(hitl_request)["decisions"]``. The resume value
must therefore be a mapping shaped like::

    {"decisions": [{"type": "approve"}]}

Resuming with a bare string such as ``"approve"``, ``"yes"`` or ``"0"`` — the
raw value a UI button or CLI prompt produces — crashes the middleware with
``TypeError: string indices must be integers``. Callers should build their
``Command(resume=...)`` payload with :func:`build_resume_command` so the UI
choice is translated into a structured decision keyed by interrupt id, and
guard values they build themselves with :func:`ensure_hitl_response`.

Decision schema follows
https://docs.langchain.com/oss/python/langchain/middleware (see the
human-in-the-loop section).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from langgraph.types import Command

APPROVE = "approve"
EDIT = "edit"
REJECT = "reject"
RESPOND = "respond"

DECISION_TYPES: frozenset[str] = frozenset({APPROVE, EDIT, REJECT, RESPOND})

# UI buttons, keyboard shortcuts and numbered CLI prompts all hand us free-form
# strings; map them onto the four decision types the middleware accepts.
CHOICE_ALIASES: dict[str, str] = {
    "0": APPROVE,
    "y": APPROVE,
    "yes": APPROVE,
    "ok": APPROVE,
    "accept": APPROVE,
    "approve": APPROVE,
    "approved": APPROVE,
    "1": REJECT,
    "n": REJECT,
    "no": REJECT,
    "deny": REJECT,
    "reject": REJECT,
    "rejected": REJECT,
    "2": EDIT,
    "edit": EDIT,
    "3": RESPOND,
    "reply": RESPOND,
    "respond": RESPOND,
}


def decision_type(choice: str) -> str:
    """Map a UI choice such as ``"yes"``, ``"approve"`` or ``"0"`` to a decision type."""
    if not isinstance(choice, str):
        msg = f"HITL choice must be a string, got {type(choice).__name__}: {choice!r}"
        raise TypeError(msg)

    resolved = CHOICE_ALIASES.get(choice.strip().lower())
    if resolved is None:
        msg = (
            f"Unknown HITL choice {choice!r}. Expected one of "
            f"{sorted(CHOICE_ALIASES)} or a decision type in {sorted(DECISION_TYPES)}."
        )
        raise ValueError(msg)
    return resolved


def build_decision(
    choice: str,
    *,
    message: str | None = None,
    edited_action: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one structured decision object from a UI choice."""
    resolved = decision_type(choice)
    decision: dict[str, Any] = {"type": resolved}

    if resolved == EDIT:
        if edited_action is None:
            msg = "An 'edit' decision requires edited_action={'name': ..., 'args': ...}."
            raise ValueError(msg)
        decision["edited_action"] = dict(edited_action)
    elif resolved == RESPOND:
        if message is None:
            msg = "A 'respond' decision requires message=<text returned to the model>."
            raise ValueError(msg)
        decision["message"] = message
    elif resolved == REJECT and message is not None:
        decision["message"] = message

    return decision


def build_hitl_response(
    choices: str | Iterable[str],
    *,
    message: str | None = None,
    edited_action: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``{"decisions": [...]}`` value the middleware indexes on resume."""
    if isinstance(choices, str):
        choices = [choices]

    decisions = [
        build_decision(choice, message=message, edited_action=edited_action)
        for choice in choices
    ]
    if not decisions:
        msg = (
            "At least one decision is required; the middleware matches one "
            "decision per gated tool call."
        )
        raise ValueError(msg)
    return {"decisions": decisions}


def ensure_hitl_response(value: Any) -> Mapping[str, Any]:
    """Reject resume values the middleware cannot index, with an actionable message."""
    if not isinstance(value, Mapping):
        msg = (
            f"HITL resume value must be a mapping like "
            f"{{'decisions': [{{'type': 'approve'}}]}}, got "
            f"{type(value).__name__}: {value!r}. Map the UI choice with "
            f"hitl.build_hitl_response(choice) before resuming."
        )
        raise TypeError(msg)

    decisions = value.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        msg = (
            f"HITL resume value must carry a non-empty 'decisions' list, got "
            f"{value!r}."
        )
        raise TypeError(msg)

    for decision in decisions:
        if (
            not isinstance(decision, Mapping)
            or decision.get("type") not in DECISION_TYPES
        ):
            msg = (
                f"Each HITL decision must be a mapping with a 'type' in "
                f"{sorted(DECISION_TYPES)}, got {decision!r}."
            )
            raise TypeError(msg)

    return value


def build_resume_command(
    interrupt_id: str,
    choices: str | Iterable[str],
    *,
    message: str | None = None,
    edited_action: Mapping[str, Any] | None = None,
) -> Command:
    """Build ``Command(resume={interrupt_id: {"decisions": [...]}})`` for one interrupt."""
    response = build_hitl_response(
        choices, message=message, edited_action=edited_action
    )
    return Command(resume={interrupt_id: ensure_hitl_response(response)})
