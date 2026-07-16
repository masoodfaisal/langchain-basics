"""Long-term memory facade for the Chinook support bot.

Backed by a LangGraph ``Store`` (see
https://docs.langchain.com/oss/python/langchain/long-term-memory and
https://docs.langchain.com/oss/python/langgraph/persistence#memory-store).
A store is independent of the checkpointer (which holds short-term, thread
scoped messages) and is shared across threads, namespaced by tuple. Here we
namespace memories by authenticated ``customer_id`` so the same trust
boundary as ``middleware.customer_scoping`` keeps tenant data isolated.

The store itself is provided by the runtime, not by this module:

* Local development with ``langgraph dev`` -- the LangGraph CLI provisions
  an in-memory store (optionally with semantic search, configured via the
  ``store`` block in ``langgraph.json``).
* CI -- tests inject a fresh :class:`InMemoryStore` directly into the
  tools' runtime; they never go through ``agent.graph``.
* LangSmith Deployment / self-hosted -- the platform provisions a managed
  Postgres-backed store, configured via the same ``store`` block. Passing
  ``store=`` to ``create_agent`` is rejected by the API loader, so we do
  not.

What's left here is just :class:`Memo`, a thin async facade so callers
(tools, middleware, tests) do not couple to LangGraph store internals.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from langgraph.store.base import BaseStore


@dataclass
class Memo:
    """Tiny async facade over :class:`BaseStore` scoped to a customer.

    Construct with the store from ``runtime.store`` so the same instance
    the agent was compiled with (or the platform injected) is used.
    """

    store: BaseStore

    @staticmethod
    def namespace(customer_id: int) -> tuple[str, str]:
        """Per-customer namespace. Mirrors ``customer_scoping`` boundary."""
        return (str(customer_id), "memories")

    async def write(
        self,
        customer_id: int,
        text: str,
        **meta: Any,
    ) -> str:
        """Store a single memory and return its key."""
        key = str(uuid.uuid4())
        await self.store.aput(
            self.namespace(customer_id),
            key,
            {"text": text, **meta},
        )
        return key

    async def search(
        self,
        customer_id: int,
        query: str,
        limit: int = 3,
    ) -> Sequence[Any]:
        """Search within a customer's namespace.

        Returns top matches by semantic similarity when the store has an
        embedding index (configured via ``langgraph.json``), otherwise
        falls back to insertion-order results within the namespace.
        """
        return await self.store.asearch(
            self.namespace(customer_id),
            query=query,
            limit=limit,
        )

    async def delete(self, customer_id: int, key: str) -> None:
        """Remove a single memory by key."""
        await self.store.adelete(self.namespace(customer_id), key)
