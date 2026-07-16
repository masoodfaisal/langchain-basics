"""Agent middleware for the Chinook support bot.

Currently provides a single ``customer_scoping`` middleware that enforces
per-customer data access *outside* the tool implementations. This is the
primary security boundary:

* Unauthenticated callers cannot invoke any account tool.
* A caller authenticated as customer A cannot read customer B's invoice,
  even if the model is tricked into passing a foreign ``invoice_id``.

Because the middleware short-circuits with a ``ToolMessage`` and never
calls ``handler(request)``, the underlying tool never runs and never reads
from the database in those cases. The model sees the refusal and must
explain it to the user.

The middleware itself is async (``@wrap_tool_call`` over an ``async def``)
so the ownership check uses the async DB layer and never blocks the event
loop. The agent must therefore be invoked via ``ainvoke`` / ``astream``.

Pattern follows https://docs.langchain.com/oss/python/langchain/middleware/custom
(see the "Tool call monitoring" and wrap_tool_call sections).
"""


## TODO -  add a check that compares runtime.context.customer_id against the customer_id stored in state
## from the first turn of this thread_id. If they don't match, refuse.



from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

from context import UserContext
from db import aconnect

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest
    from langgraph.types import Command

logger = logging.getLogger(__name__)

# Tools that operate on authenticated-customer-scoped data. Non-listed tools
# (catalog lookups, recommendations) are allowed for anyone.
ACCOUNT_TOOLS: frozenset[str] = frozenset(
    {"list_my_orders", "get_invoice_details", "remember", "recall"}
)


def _deny(request: "ToolCallRequest", message: str) -> ToolMessage:
    """Build a ToolMessage refusal bound to the pending tool_call id."""
    return ToolMessage(
        content=message,
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error",
    )


def _customer_id_from(request: "ToolCallRequest") -> int | None:
    ctx: UserContext | None = getattr(request.runtime, "context", None)
    return ctx.customer_id if ctx is not None else None


@wrap_tool_call
async def customer_scoping(
    request: "ToolCallRequest",
    handler: Callable[
        ["ToolCallRequest"], Awaitable["ToolMessage | Command"]
    ],
) -> "ToolMessage | Command":
    """Gate account tools on the authenticated customer in ``runtime.context``."""
    tool_name = request.tool_call["name"]

    # Catalog / discovery tools are public; let them through unchanged.
    if tool_name not in ACCOUNT_TOOLS:
        return await handler(request)

    customer_id = _customer_id_from(request)
    if customer_id is None:
        logger.info("auth-deny: %s called without a customer in context", tool_name)
        return _deny(
            request,
            "Access denied: you must be signed in as a customer to use this tool.",
        )

    # Ownership check for per-invoice lookups. We verify ownership here so
    # that a buggy, replaced, or prompt-injected tool still cannot expose
    # another customer's data.
    if tool_name == "get_invoice_details":
        invoice_id = request.tool_call["args"].get("invoice_id")
        if invoice_id is None:
            return _deny(request, "Access denied: invoice_id is required.")

        async with aconnect() as conn:
            cur = await conn.execute(
                "SELECT CustomerId FROM Invoice WHERE InvoiceId = :iid",
                {"iid": invoice_id},
            )
            row = await cur.fetchone()

        if row is not None and row["CustomerId"] != customer_id:
            logger.info(
                "auth-deny: customer %s tried to read invoice %s (owner=%s)",
                customer_id, invoice_id, row["CustomerId"],
            )
            return _deny(
                request,
                (
                    f"Access denied: invoice {invoice_id} does not belong to "
                    "your account. I can only show you invoices that belong to you."
                ),
            )


    # All checks passed - let the tool run.
    return await handler(request)
