"""Tests for long-term memory: ``memory.Memo`` and the ``remember`` /
``recall`` tools.

We use a fresh :class:`InMemoryStore` per test (no embedding index) so the
tests are deterministic and offline. The store still supports namespace
listing and ``query`` search; ``InMemoryStore.asearch`` falls back to
substring matching when no index is configured, which is enough to cover
the tools' contracts here. ``customer_scoping`` is exercised separately in
``test_middleware.py`` -- here we just drive the tools directly with a
synthesized runtime, the same way ``test_tools.py`` does.

``pytest-asyncio`` runs every ``async def test_*`` here automatically
(``asyncio_mode = "auto"`` in ``pyproject.toml``); no per-test decorator
needed.

Run with::

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_memory.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError

from context import UserContext
from memory import Memo
from tools import recall, remember


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _runtime(customer_id: int | None, store) -> SimpleNamespace:
    """Minimal ToolRuntime stand-in: tools read ``.context`` and ``.store``."""
    return SimpleNamespace(
        context=UserContext(customer_id=customer_id),
        store=store,
    )


@pytest.fixture
def store():
    """A fresh in-memory store per test (no embedding index)."""
    return InMemoryStore()


# ---------------------------------------------------------------------------
# Memo facade
# ---------------------------------------------------------------------------
def test_memo_namespace_is_per_customer():
    assert Memo.namespace(1) == ("1", "memories")
    assert Memo.namespace(42) == ("42", "memories")
    assert Memo.namespace(1) != Memo.namespace(2)


async def test_memo_write_then_search_returns_saved_text(store):
    memo = Memo(store)
    key = await memo.write(1, "User loves jazz")
    assert key, "write should return a non-empty key"

    hits = await memo.search(1, query="jazz", limit=5)
    assert len(hits) == 1
    assert hits[0].value["text"] == "User loves jazz"
    assert tuple(hits[0].namespace) == Memo.namespace(1)


async def test_memo_search_isolates_customers(store):
    """Customer A's memory must not surface in customer B's search."""
    memo = Memo(store)
    await memo.write(1, "Customer A likes jazz")
    await memo.write(2, "Customer B likes metal")

    a_hits = await memo.search(1, query="genre", limit=5)
    b_hits = await memo.search(2, query="genre", limit=5)

    a_texts = [h.value["text"] for h in a_hits]
    b_texts = [h.value["text"] for h in b_hits]

    assert a_texts == ["Customer A likes jazz"]
    assert b_texts == ["Customer B likes metal"]


async def test_memo_delete_removes_memory(store):
    memo = Memo(store)
    key = await memo.write(1, "Forget me")
    await memo.delete(1, key)

    hits = await memo.search(1, query="forget", limit=5)
    assert hits == []


async def test_memo_search_empty_namespace_returns_empty_list(store):
    memo = Memo(store)
    hits = await memo.search(99, query="anything", limit=5)
    assert hits == []


# ---------------------------------------------------------------------------
# remember tool
# ---------------------------------------------------------------------------
async def test_remember_saves_for_authenticated_customer(store):
    out = await remember.coroutine(
        fact="Prefers vinyl over CDs",
        runtime=_runtime(customer_id=1, store=store),
    )
    assert out.lower().startswith("saved")

    # Confirm via the underlying store, not the tool, to keep this test
    # honest if the tool's success message ever changes.
    items = await store.asearch(Memo.namespace(1), query="vinyl", limit=5)
    assert len(items) == 1
    assert items[0].value["text"] == "Prefers vinyl over CDs"


async def test_remember_refuses_anonymous_caller(store):
    out = await remember.coroutine(
        fact="should not be saved",
        runtime=_runtime(customer_id=None, store=store),
    )
    assert "sign in" in out.lower()
    # And nothing was written for anyone.
    assert await store.asearch(("1", "memories"), query="should", limit=5) == []


async def test_remember_refuses_when_store_is_unconfigured():
    out = await remember.coroutine(
        fact="hello",
        runtime=_runtime(customer_id=1, store=None),
    )
    assert "memory is not configured" in out.lower()


# ---------------------------------------------------------------------------
# recall tool
# ---------------------------------------------------------------------------
async def test_recall_returns_saved_memories_for_owner(store):
    # Pre-seed via the same path the tool would use.
    await Memo(store).write(1, "Prefers vinyl over CDs")
    await Memo(store).write(1, "Email invoices as PDF")

    out = await recall.coroutine(
        query="format",
        runtime=_runtime(customer_id=1, store=store),
        limit=5,
    )
    assert "Prefers vinyl over CDs" in out or "Email invoices as PDF" in out
    # Each line is bullet-prefixed.
    assert out.lstrip().startswith("- ")


async def test_recall_does_not_leak_other_customers(store):
    await Memo(store).write(2, "Customer 2's secret preference")

    out = await recall.coroutine(
        query="secret preference",
        runtime=_runtime(customer_id=1, store=store),
        limit=5,
    )
    assert "secret preference" not in out.lower()
    assert "no memories" in out.lower()


async def test_recall_returns_no_memories_message_when_empty(store):
    out = await recall.coroutine(
        query="anything",
        runtime=_runtime(customer_id=1, store=store),
        limit=5,
    )
    assert "no memories" in out.lower()


@pytest.mark.parametrize("limit", [0, -1])
def test_recall_rejects_non_positive_limit(limit):
    # Validate at the schema layer: ``.coroutine`` skips Pydantic, but the
    # agent's tool node always runs args through ``args_schema`` first.
    # Pure schema validation is sync, so this test stays sync.
    with pytest.raises(ValidationError, match="greater_than_equal"):
        recall.args_schema.model_validate({"query": "anything", "limit": limit})


def test_recall_rejects_oversized_limit():
    with pytest.raises(ValidationError, match="less_than_equal"):
        recall.args_schema.model_validate({"query": "anything", "limit": 999_999})


async def test_recall_refuses_anonymous_caller(store):
    await Memo(store).write(1, "Prefers vinyl over CDs")

    out = await recall.coroutine(
        query="vinyl",
        runtime=_runtime(customer_id=None, store=store),
        limit=5,
    )
    assert "sign in" in out.lower()
    assert "vinyl" not in out.lower()


async def test_recall_refuses_when_store_is_unconfigured():
    out = await recall.coroutine(
        query="anything",
        runtime=_runtime(customer_id=1, store=None),
        limit=5,
    )
    assert "memory is not configured" in out.lower()
