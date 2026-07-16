"""Tests for the ``customer_scoping`` middleware.

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

Run with::

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_middleware.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain.messages import ToolMessage

from context import UserContext
from middleware import customer_scoping

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
