"""Input/output tests for every tool in ``tools.py``.

These tests exercise the tools directly (bypassing the LLM) against the
local ``chinook.db``. They cover:

* Happy paths for all four tools.
* Both inline refusal paths (anonymous + wrong-owner).
* Edge cases: unknown album, unknown genre, missing invoice.

Tools are async, so we ``await`` them directly via ``.ainvoke`` for the
music tools (validates the schema layer too) and via ``.coroutine`` for
the account tools (so we can pass the synthesized ``runtime`` directly).

``pytest-asyncio`` runs every ``async def test_*`` here automatically
(``asyncio_mode = "auto"`` in ``pyproject.toml``); no per-test decorator
needed.

Run with::

    PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from context import UserContext
from tools import (
    find_similar_albums,
    get_invoice_details,
    list_my_orders,
    popular_in_genre,
    recall,
    remember,
)

# Chinook ground truth.
OWNER_OF_INVOICE_1 = 2
NON_OWNER_OF_INVOICE_1 = 1


def _runtime(customer_id: int | None) -> SimpleNamespace:
    """Minimal ToolRuntime stand-in: our tools only read .context."""
    return SimpleNamespace(context=UserContext(customer_id=customer_id))


# ---------------------------------------------------------------------------
# find_similar_albums
# ---------------------------------------------------------------------------
async def test_find_similar_albums_returns_recommendations():
    out = await find_similar_albums.ainvoke(
        {"album_name": "Let There Be Rock", "limit": 3}
    )
    assert "Let There Be Rock" in out
    # Exactly `limit` hits, each with an artist and a score.
    assert out.count("\n  - ") == 3
    assert "genre-matching tracks" in out


async def test_find_similar_albums_handles_unknown_album():
    out = await find_similar_albums.ainvoke(
        {"album_name": "__no such album__", "limit": 5}
    )
    assert "No albums found" in out


@pytest.mark.parametrize("album_name", ["%", "_"])
async def test_find_similar_albums_escapes_wildcards(album_name):
    out = await find_similar_albums.ainvoke(
        {"album_name": album_name, "limit": 5}
    )
    assert "No albums found" in out


async def test_find_similar_albums_still_supports_partial_title_matching():
    out = await find_similar_albums.ainvoke(
        {"album_name": "Let There", "limit": 3}
    )
    assert "Let There" in out
    assert out.count("\n  - ") == 3


@pytest.mark.parametrize("limit", [0, -1])
async def test_find_similar_albums_rejects_non_positive_limit(limit):
    with pytest.raises(ValidationError, match="greater_than_equal"):
        await find_similar_albums.ainvoke(
            {"album_name": "Let There Be Rock", "limit": limit}
        )


async def test_find_similar_albums_rejects_oversized_limit():
    with pytest.raises(ValidationError, match="less_than_equal"):
        await find_similar_albums.ainvoke(
            {"album_name": "Let There Be Rock", "limit": 999_999}
        )


# ---------------------------------------------------------------------------
# popular_in_genre
# ---------------------------------------------------------------------------
async def test_popular_in_genre_returns_tracks_for_jazz():
    out = await popular_in_genre.ainvoke({"genre": "Jazz", "limit": 5})
    assert "Jazz" in out
    assert out.count("\n  - ") == 5
    assert "sold" in out  # ".. sold Nx"


async def test_popular_in_genre_handles_unknown_genre():
    out = await popular_in_genre.ainvoke({"genre": "__bogus__", "limit": 5})
    assert "No tracks found" in out


@pytest.mark.parametrize("genre", ["%", "_"])
async def test_popular_in_genre_does_not_treat_wildcards_as_patterns(genre):
    out = await popular_in_genre.ainvoke({"genre": genre, "limit": 5})
    assert "No tracks found" in out


@pytest.mark.parametrize("limit", [0, -1])
async def test_popular_in_genre_rejects_non_positive_limit(limit):
    with pytest.raises(ValidationError, match="greater_than_equal"):
        await popular_in_genre.ainvoke({"genre": "Jazz", "limit": limit})


async def test_popular_in_genre_rejects_oversized_limit():
    with pytest.raises(ValidationError, match="less_than_equal"):
        await popular_in_genre.ainvoke({"genre": "Jazz", "limit": 999_999})


# ---------------------------------------------------------------------------
# list_my_orders
# ---------------------------------------------------------------------------
async def test_list_my_orders_returns_invoices_for_authenticated_customer():
    out = await list_my_orders.coroutine(runtime=_runtime(1), limit=3)
    assert "Invoice #" in out
    assert out.count("\n  - ") == 3


async def test_list_my_orders_refuses_when_no_customer_in_context():
    out = await list_my_orders.coroutine(runtime=_runtime(None), limit=3)
    assert "sign in" in out.lower()
    assert "Invoice #" not in out


@pytest.mark.parametrize("limit", [0, -1])
def test_list_my_orders_rejects_non_positive_limit(limit):
    # Validate at the schema layer: ``.coroutine`` skips Pydantic, but the
    # agent's tool node always runs args through ``args_schema`` first.
    # Pure schema validation is sync, so this test stays sync.
    with pytest.raises(ValidationError, match="greater_than_equal"):
        list_my_orders.args_schema.model_validate({"limit": limit})


def test_list_my_orders_rejects_oversized_limit():
    with pytest.raises(ValidationError, match="less_than_equal"):
        list_my_orders.args_schema.model_validate({"limit": 999_999})


# ---------------------------------------------------------------------------
# get_invoice_details
# ---------------------------------------------------------------------------
async def test_get_invoice_details_returns_line_items_for_owner():
    out = await get_invoice_details.coroutine(
        invoice_id=1, runtime=_runtime(OWNER_OF_INVOICE_1)
    )
    assert "Invoice #1" in out
    # Invoice 1 line items are Accept's "Balls to the Wall" + "Restless and Wild".
    assert "Accept" in out
    assert "Balls to the Wall" in out


async def test_get_invoice_details_refuses_for_non_owner():
    out = await get_invoice_details.coroutine(
        invoice_id=1, runtime=_runtime(NON_OWNER_OF_INVOICE_1)
    )
    assert "not associated with your account" in out
    # No line-item data should leak.
    assert "Accept" not in out
    assert "Balls to the Wall" not in out


async def test_get_invoice_details_refuses_when_anonymous():
    out = await get_invoice_details.coroutine(
        invoice_id=1, runtime=_runtime(None)
    )
    assert "sign in" in out.lower()


async def test_get_invoice_details_handles_missing_invoice():
    out = await get_invoice_details.coroutine(
        invoice_id=999_999, runtime=_runtime(OWNER_OF_INVOICE_1)
    )
    assert "No invoice found" in out


# ---------------------------------------------------------------------------
# Tool schemas (input validation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tool, required_arg",
    [
        (find_similar_albums, "album_name"),
        (popular_in_genre, "genre"),
    ],
)
def test_tool_schema_declares_required_arg(tool, required_arg):
    schema = tool.args_schema.model_json_schema()
    assert required_arg in schema["properties"]
    assert required_arg in schema.get("required", [])
