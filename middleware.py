"""Agent middleware for the Chinook support bot.

Provides ``customer_scoping``, which enforces per-customer data access
*outside* the tool implementations, and ``sanitize_message_names``, which
keeps message ``name`` fields within the shape the Chat Completions API
accepts. ``customer_scoping`` is the primary security boundary:

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
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain.messages import ToolMessage

from context import UserContext
from db import aconnect

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest, ModelResponse
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


# OpenAI requires every ``messages[N].name`` to match ``^[^\s<|\\/>]+$``, so a
# display name with spaces ("Chinook Support Agent") stamped onto an AIMessage
# 400s the *next* turn when that message is replayed from the checkpoint.
_INVALID_NAME_CHARS = re.compile(r"[\s<|\\/>]+")


def sanitize_name(name: object) -> str | None:
    """Coerce a message name into ``^[^\\s<|\\\\/>]+$``, or ``None`` if nothing is left."""
    if not isinstance(name, str):
        return None
    return _INVALID_NAME_CHARS.sub("_", name).strip("_") or None


@wrap_model_call
async def sanitize_message_names(
    request: "ModelRequest",
    handler: Callable[["ModelRequest"], Awaitable["ModelResponse"]],
) -> "ModelResponse":
    """Rewrite invalid message ``name`` values before the request reaches the model."""
    messages = list(request.messages)
    rewritten = False

    for index, message in enumerate(messages):
        name = getattr(message, "name", None)
        if name is None:
            continue
        sanitized = sanitize_name(name)
        if sanitized == name:
            continue
        logger.debug("sanitizing message name %r -> %r", name, sanitized)
        messages[index] = message.model_copy(update={"name": sanitized})
        rewritten = True

    if rewritten:
        request = request.override(messages=messages)

    return await handler(request)
