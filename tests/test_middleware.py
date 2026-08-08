"""Tests for the agent middleware.

Drives ``customer_scoping.awrap_tool_call`` directly with a fake request and
a fake async handler so each gate path is exercised without spinning up the
agent or spending LLM tokens. The middleware is async (so it can use the
async DB layer); each test is an ``async def`` driven by ``pytest-asyncio``
in auto mode.

Coverage:

1. Public tools pass through for any caller (authed or not).
2. Account tools deny anonymous callers (authn gate).
3. Account tools allow authenticated callers when no per-resource check
   is required.
4. ``get_invoice_details`` denies cross-customer access (authz gate)
   *before* the SQL fans out.
5. ``get_invoice_details`` allows the rightful owner.
6. ``get_invoice_details`` denies when ``invoice_id`` is missing.
7. Unknown invoice IDs pass through to the tool for normal not-found handling.
8. ``bounded_tool_retry`` lets a first attempt through, refuses a re-issued
   call whose only difference is list-argument ordering, and refuses any call
   to a tool that already burned its failure budget.
9. ``demo_feedback`` records root-trace feedback after agent completion and is
   a no-op when tracing is disabled.
10. The graph selected by ``langgraph.json`` registers ``demo_feedback``.

Run with::

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_middleware.py -v
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from langchain.messages import AIMessage, ToolMessage

from context import UserContext
from middleware import (
    DEMO_FEEDBACK_KEY,
    DEMO_FEEDBACK_SCORE,
    DEMO_FEEDBACK_VALUE,
    bounded_tool_retry,
    customer_scoping,
    demo_feedback,
)

# Chinook ground truth used in the ownership cases.
OWNER_OF_INVOICE_1 = 2
NON_OWNER_OF_INVOICE_1 = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _request(
    tool_name: str,
    args: dict[str, Any],
    customer_id: int | None,
) -> SimpleNamespace:
    """Fake ToolCallRequest. The middleware only reads ``tool_call`` and ``runtime``."""
    return SimpleNamespace(
        tool_call={"name": tool_name, "args": args, "id": "call_test"},
        tool=None,
        state={},
        runtime=SimpleNamespace(context=UserContext(customer_id=customer_id)),
    )


_ALLOWED_SENTINEL = "TOOL_RAN"


async def _fake_handler(_request) -> str:
    """Stand-in for the real (async) tool node. If the middleware lets the
    call through, the test sees this sentinel back."""
    return _ALLOWED_SENTINEL


async def _drive(tool_name: str, args: dict[str, Any], customer_id: int | None):
    """Drive the (async) middleware with a synthesized request + handler."""
    return await customer_scoping.awrap_tool_call(
        _request(tool_name, args, customer_id),
        _fake_handler,
    )


def _assert_denied(result, *, expected_substrings: list[str]) -> None:
    assert isinstance(result, ToolMessage), (
        f"expected ToolMessage refusal, got {type(result).__name__}: {result!r}"
    )
    assert result.status == "error", f"expected status='error', got {result.status!r}"
    text = str(result.content).lower()
    for needle in expected_substrings:
        assert needle.lower() in text, (
            f"refusal message missing {needle!r}; got: {result.content!r}"
        )


def _assert_allowed(result) -> None:
    assert result == _ALLOWED_SENTINEL, (
        f"expected handler to run and return {_ALLOWED_SENTINEL!r}, got {result!r}"
    )


# ---------------------------------------------------------------------------
# 1. Public tools always pass through
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("customer_id", [None, 1, 2])
async def test_public_tool_passes_through_for_any_caller(customer_id):
    result = await _drive("popular_in_genre", {"genre": "Jazz"}, customer_id=customer_id)
    _assert_allowed(result)


async def test_public_tool_passes_through_with_unrelated_args():
    # Whatever shape the public tool wants, the middleware should stay out of it.
    result = await _drive(
        "find_similar_albums",
        {"album_name": "Let There Be Rock", "limit": 3},
        customer_id=None,
    )
    _assert_allowed(result)


# ---------------------------------------------------------------------------
# 2-3. list_my_orders: authn gate
# ---------------------------------------------------------------------------
async def test_list_my_orders_denied_for_anonymous_caller():
    result = await _drive("list_my_orders", {"limit": 5}, customer_id=None)
    _assert_denied(result, expected_substrings=["signed in"])


async def test_list_my_orders_allowed_for_authenticated_caller():
    result = await _drive("list_my_orders", {"limit": 5}, customer_id=1)
    _assert_allowed(result)


# ---------------------------------------------------------------------------
# 4-5. get_invoice_details: ownership gate
# ---------------------------------------------------------------------------
async def test_get_invoice_details_denied_for_non_owner_before_sql():
    """Cross-customer access must be refused before the underlying SQL runs.

    The fake handler returns a sentinel; if it ever ran, the assertion
    below would fail because the result would be ``_ALLOWED_SENTINEL``
    instead of a ToolMessage refusal.
    """
    result = await _drive(
        "get_invoice_details",
        {"invoice_id": 1},
        customer_id=NON_OWNER_OF_INVOICE_1,
    )
    _assert_denied(result, expected_substrings=["does not belong"])


async def test_get_invoice_details_allowed_for_owner():
    result = await _drive(
        "get_invoice_details",
        {"invoice_id": 1},
        customer_id=OWNER_OF_INVOICE_1,
    )
    _assert_allowed(result)


async def test_get_invoice_details_denied_when_anonymous():
    """Anonymous callers should be blocked at the authn gate, before the
    ownership lookup even runs."""
    result = await _drive("get_invoice_details", {"invoice_id": 1}, customer_id=None)
    _assert_denied(result, expected_substrings=["signed in"])


# ---------------------------------------------------------------------------
# 6. get_invoice_details: arg validation
# ---------------------------------------------------------------------------
async def test_get_invoice_details_denied_when_invoice_id_missing():
    result = await _drive("get_invoice_details", {}, customer_id=1)
    _assert_denied(result, expected_substrings=["invoice_id"])


# ---------------------------------------------------------------------------
# 7. Unknown invoice id: middleware does not fabricate ownership
# ---------------------------------------------------------------------------
async def test_get_invoice_details_unknown_invoice_passes_through():
    """If the invoice doesn't exist at all, ownership cannot be evaluated;
    the middleware lets the call through and the tool itself surfaces the
    "no invoice found" message. This documents the current contract."""
    result = await _drive(
        "get_invoice_details",
        {"invoice_id": 999_999_999},
        customer_id=1,
    )
    _assert_allowed(result)


# ---------------------------------------------------------------------------
# 8. bounded_tool_retry: failed calls are not re-issued
# ---------------------------------------------------------------------------
def _turn(*attempts: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """Message history where every listed tool call was requested and failed."""
    messages: list[Any] = []
    for index, (tool_name, args) in enumerate(attempts):
        call_id = f"call_{index}"
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": args, "id": call_id}],
            )
        )
        messages.append(
            ToolMessage(
                content="boom",
                tool_call_id=call_id,
                name=tool_name,
                status="error",
            )
        )
    return {"messages": messages}


async def _drive_retry(
    tool_name: str,
    args: dict[str, Any],
    state: dict[str, Any],
):
    request = _request(tool_name, args, customer_id=1)
    request.state = state
    return await bounded_tool_retry.awrap_tool_call(request, _fake_handler)


async def test_first_attempt_passes_through():
    result = await _drive_retry("popular_in_genre", {"genre": "Jazz"}, {"messages": []})
    _assert_allowed(result)


async def test_reordered_list_argument_is_the_same_failed_call():
    """A permuted list argument must canonicalize to the signature that failed."""
    state = _turn(("find_similar_albums", {"genres": ["Jazz", "Rock"]}))

    result = await _drive_retry(
        "find_similar_albums",
        {"genres": ["Rock", "Jazz"]},
        state,
    )

    _assert_denied(result, expected_substrings=["retry refused", "blocked"])


async def test_different_arguments_are_allowed_until_the_failure_cap():
    state = _turn(("find_similar_albums", {"album_name": "Ride the Lightning"}))

    result = await _drive_retry(
        "find_similar_albums",
        {"album_name": "Let There Be Rock"},
        state,
    )

    _assert_allowed(result)


async def test_tool_refused_once_failure_budget_is_exhausted():
    state = _turn(
        ("find_similar_albums", {"album_name": "one"}),
        ("find_similar_albums", {"album_name": "two"}),
    )

    result = await _drive_retry("find_similar_albums", {"album_name": "three"}, state)

    _assert_denied(result, expected_substrings=["retry budget exhausted", "deliverables"])


async def test_other_tools_keep_their_own_budget():
    state = _turn(
        ("find_similar_albums", {"album_name": "one"}),
        ("find_similar_albums", {"album_name": "two"}),
    )

    result = await _drive_retry("popular_in_genre", {"genre": "Jazz"}, state)

    _assert_allowed(result)


# ---------------------------------------------------------------------------
# 9. In-source demo feedback
# ---------------------------------------------------------------------------
async def test_demo_feedback_attaches_score_to_root_trace(monkeypatch):
    trace_id = uuid4()
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def create_feedback(self, **kwargs):
            calls.append(kwargs)

    current_run = SimpleNamespace(
        id=uuid4(),
        trace_id=trace_id,
        ls_client=FakeClient(),
    )
    monkeypatch.setattr("middleware.get_current_run_tree", lambda: current_run)

    result = await demo_feedback.aafter_agent({}, SimpleNamespace())

    assert result is None
    assert calls == [
        {
            "key": DEMO_FEEDBACK_KEY,
            "score": DEMO_FEEDBACK_SCORE,
            "value": DEMO_FEEDBACK_VALUE,
            "trace_id": trace_id,
            "comment": "Created in source by the after-agent demo middleware.",
        }
    ]


async def test_demo_feedback_is_noop_without_active_trace(monkeypatch):
    monkeypatch.setattr("middleware.get_current_run_tree", lambda: None)

    result = await demo_feedback.aafter_agent({}, SimpleNamespace())

    assert result is None


def test_configured_graph_registers_demo_feedback():
    """Protect the LangGraph entrypoint from silently omitting the middleware."""
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads((project_root / "langgraph.json").read_text())
    module_path, _, _attribute = config["graphs"]["agent"].partition(":")
    graph_source = (project_root / module_path).read_text()
    tree = ast.parse(graph_source)

    create_agent_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_agent"
    ]
    assert len(create_agent_calls) == 1

    middleware_keyword = next(
        (
            keyword
            for keyword in create_agent_calls[0].keywords
            if keyword.arg == "middleware"
        ),
        None,
    )
    assert middleware_keyword is not None
    assert isinstance(middleware_keyword.value, ast.List)
    assert any(
        isinstance(item, ast.Name) and item.id == "demo_feedback"
        for item in middleware_keyword.value.elts
    )
