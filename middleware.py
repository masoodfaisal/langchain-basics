"""Agent middleware for the Chinook support bot.

Currently provides a single ``customer_scoping`` middleware that enforces
per-customer data access *outside* the tool implementations. This is the
primary security boundary:

* Unauthenticated callers cannot invoke any account tool.
* A caller authenticated as customer A cannot read customer B's invoice,
  even if the model is tricked into passing a foreign ``invoice_id``.
* The authenticated customer is resolved once per thread and pinned into
  agent state, so a later turn whose context carries a different customer
  id is refused instead of silently returning another account's data.

Because the middleware short-circuits with a ``ToolMessage`` and never
calls ``handler(request)``, the underlying tool never runs and never reads
from the database in those cases. The model sees the refusal and must
explain it to the user.

The middleware hooks are async (``abefore_agent`` / ``awrap_tool_call``)
so the ownership check uses the async DB layer and never blocks the event
loop. The agent must therefore be invoked via ``ainvoke`` / ``astream``.

Pattern follows https://docs.langchain.com/oss/python/langchain/middleware/custom
(see the "Tool call monitoring" and wrap_tool_call sections).
"""


from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import ToolMessage
from typing_extensions import NotRequired

from context import UserContext
from db import aconnect

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime
    from langgraph.types import Command

logger = logging.getLogger(__name__)

# State key holding the customer id this thread was authenticated as.
SESSION_CUSTOMER_KEY = "session_customer_id"

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


def _state_value(state: Any, key: str) -> Any:
    """Read ``key`` from agent state, which may be a dict or a model."""
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


class CustomerScopingState(AgentState):
    """Agent state plus the customer id pinned to this thread."""

    session_customer_id: NotRequired[int | None]


class CustomerScoping(AgentMiddleware[CustomerScopingState, UserContext]):
    """Bind account tools to the customer authenticated on the thread's first turn."""

    state_schema = CustomerScopingState

    async def abefore_agent(
        self,
        state: CustomerScopingState,
        runtime: "Runtime[UserContext]",
    ) -> dict[str, Any] | None:
        """Resolve the authenticated customer once per thread and pin it into state."""
        if _state_value(state, SESSION_CUSTOMER_KEY) is not None:
            return None

        ctx: UserContext | None = getattr(runtime, "context", None)
        customer_id = ctx.customer_id if ctx is not None else None
        if customer_id is None:
            return None

        logger.info("session-bind: thread pinned to customer_id=%s", customer_id)
        return {SESSION_CUSTOMER_KEY: customer_id}

    async def awrap_tool_call(
        self,
        request: "ToolCallRequest",
        handler: Callable[
            ["ToolCallRequest"], Awaitable["ToolMessage | Command"]
        ],
    ) -> "ToolMessage | Command":
        """Gate account tools on the customer pinned to this session."""
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

        # The context can be re-supplied on every invocation of a thread, so a
        # later turn may carry a different customer id than the one this
        # session authenticated as. Refuse rather than serve the other account.
        session_customer_id = _state_value(request.state, SESSION_CUSTOMER_KEY)
        if session_customer_id is not None and session_customer_id != customer_id:
            logger.warning(
                "auth-deny: %s called with customer_id=%s but session is bound "
                "to customer_id=%s",
                tool_name, customer_id, session_customer_id,
            )
            return _deny(
                request,
                (
                    "Access denied: this conversation is bound to a different "
                    "signed-in account. Please start a new session to switch "
                    "accounts."
                ),
            )

        logger.info(
            "account-tool-call: tool=%s customer_id=%s session_customer_id=%s",
            tool_name, customer_id, session_customer_id,
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


customer_scoping = CustomerScoping()
