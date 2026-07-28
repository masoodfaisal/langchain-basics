"""Tools exposed to the Chinook customer support agent.

Four areas, seven tools:

Music discovery
    * ``find_similar_albums`` - albums that share genres with a given album.
    * ``popular_in_genre``    - best-selling tracks within a genre.

Account & order support
    * ``list_my_orders``      - the authenticated customer's recent invoices.
    * ``get_invoice_details`` - line items for one of their invoices.

Long-term memory (per-customer, cross-thread)
    * ``remember``            - save a durable fact about the customer.
    * ``recall``              - search the customer's saved memories.

LLM delegation
    * ``ask_music_expert``    - ask a second LLM for music expertise.

The account and memory tools read ``runtime.context.customer_id`` (see
``context.py``) so they never trust a customer id passed in by the model.
The memory tools also read ``runtime.store`` (the LangGraph store the
agent was compiled with via ``create_agent(..., store=...)``).

All tools are async (``async def``) so SQLite I/O does not block the event
loop. Sync callers must use ``await`` / ``ainvoke`` (see tests for examples).
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

import httpx
from langchain.tools import ToolRuntime, tool
from langchain_openai import ChatOpenAI
from pydantic import Field

from context import UserContext
from db import aconnect
from memory import Memo

logger = logging.getLogger(__name__)

MAX_LIMIT = 50

# This model is deliberately separate from the agent model in ``agent.py``.
# Its settings fall back to the primary model settings, while TOOL_LLM_*
# variables allow the tool to call a different model or gateway.
_tool_llm_verify_env = os.getenv(
    "TOOL_LLM_VERIFY_SSL", os.getenv("OPENAI_VERIFY_SSL", "true")
).strip().lower()
_tool_llm_http_async_client = httpx.AsyncClient(
    verify=_tool_llm_verify_env not in {"false", "0", "no"}
)
music_expert_model = ChatOpenAI(
    openai_api_key=os.getenv(
        "TOOL_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "sk-123456")
    ),
    openai_api_base=os.getenv(
        "TOOL_LLM_BASE_URL",
        os.getenv("OPENAI_BASE_URL", "http://ai-gateway:4000"),
    ),
    model_name=os.getenv(
        "TOOL_LLM_MODEL", os.getenv("MODEL_NAME", "llama-distributed")
    ),
    temperature=0.01,
    http_async_client=_tool_llm_http_async_client,
)

# Reusable annotated type for paginated tool args. Pydantic enforces the
# bounds before our function body runs, and ``@tool`` surfaces validation
# errors back to the model as a ToolMessage so it can retry with a sane
# value -- no need for our own branchy ``_normalize_limit`` helper.
Limit = Annotated[
    int,
    Field(
        ge=1,
        le=MAX_LIMIT,
        description=(
            f"Maximum number of results to return (1-{MAX_LIMIT}). "
            "Values outside this range are rejected."
        ),
    ),
]


def _escape_like(value: str) -> str:
    """Escape SQLite LIKE metacharacters in user-controlled input."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ---------------------------------------------------------------------------
# Music discovery
# ---------------------------------------------------------------------------
@tool
async def find_similar_albums(album_name: str, limit: Limit = 5) -> str:
    """Recommend albums similar to a given album.

    Similarity is genre-based: we take the distinct genres of tracks on the
    source album, then return other albums whose tracks share those genres,
    ranked by how many tracks overlap.

    Args:
        album_name: Title (or partial title) of the album to match against.
        limit: Maximum number of recommendations to return.
    """
    needle = _escape_like(album_name)

    sql = """
        WITH source_album AS (
            SELECT AlbumId
            FROM Album
            WHERE Title LIKE :needle ESCAPE '\\'
            ORDER BY Title
            LIMIT 1
        ),
        source_genres AS (
            SELECT DISTINCT t.GenreId
            FROM Track t
            JOIN source_album sa ON t.AlbumId = sa.AlbumId
        )
        SELECT
            a.Title         AS album,
            ar.Name         AS artist,
            COUNT(*)        AS genre_match_tracks
        FROM Track t
        JOIN Album  a  ON a.AlbumId  = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = a.ArtistId
        WHERE t.GenreId IN (SELECT GenreId FROM source_genres)
          AND a.AlbumId NOT IN (SELECT AlbumId FROM source_album)
        GROUP BY a.AlbumId
        ORDER BY genre_match_tracks DESC, a.Title
        LIMIT :limit
    """
    async with aconnect() as conn:
        cur = await conn.execute(
            sql, {"needle": f"%{needle}%", "limit": limit}
        )
        rows = await cur.fetchall()

    if not rows:
        return f"No albums found matching '{album_name}'."
    lines = [f"Albums similar to '{album_name}':"]
    lines += [
        f"  - {r['album']} by {r['artist']} "
        f"({r['genre_match_tracks']} genre-matching tracks)"
        for r in rows
    ]
    return "\n".join(lines)


@tool
async def popular_in_genre(genre: str, limit: Limit = 10) -> str:
    """Return the best-selling tracks in a genre.

    Popularity is measured by the number of ``InvoiceLine`` rows referencing
    each track (i.e., how many times it has been sold).

    Args:
        genre: Genre name, e.g. "Jazz", "Rock", "Classical".
        limit: Maximum number of tracks to return.
    """
    sql = """
        SELECT
            t.Name              AS track,
            ar.Name             AS artist,
            a.Title             AS album,
            COUNT(il.InvoiceLineId) AS times_sold
        FROM Track t
        JOIN Genre  g  ON g.GenreId   = t.GenreId
        JOIN Album  a  ON a.AlbumId   = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = a.ArtistId
        LEFT JOIN InvoiceLine il ON il.TrackId = t.TrackId
        WHERE g.Name = :genre
        GROUP BY t.TrackId
        ORDER BY times_sold DESC, t.Name
        LIMIT :limit
    """
    async with aconnect() as conn:
        cur = await conn.execute(sql, {"genre": genre, "limit": limit})
        rows = await cur.fetchall()

    if not rows:
        return f"No tracks found in genre '{genre}'."
    lines = [f"Top {len(rows)} tracks in {genre}:"]
    lines += [
        f"  - '{r['track']}' by {r['artist']} "
        f"(album: {r['album']}, sold {r['times_sold']}x)"
        for r in rows
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Account & order support
# ---------------------------------------------------------------------------
def _require_customer_id(runtime: ToolRuntime[UserContext]) -> int | str:
    """Return the authenticated customer id, or a refusal string."""
    ctx = runtime.context
    if ctx is None or ctx.customer_id is None:
        return (
            "I can't look that up without an authenticated customer. "
            "Please sign in and try again."
        )
    return ctx.customer_id


@tool
async def list_my_orders(
    runtime: ToolRuntime[UserContext],
    limit: Limit = 10,
) -> str:
    """List the authenticated customer's most recent invoices.

    Args:
        limit: Maximum number of invoices to return (most recent first).
    """
    customer_id = _require_customer_id(runtime)
    if isinstance(customer_id, str):
        return customer_id

    logger.info(
        "account-tool: list_my_orders customer_id=%s limit=%s", customer_id, limit
    )

    sql = """
        SELECT
            InvoiceId,
            CustomerId,
            InvoiceDate,
            BillingCity,
            BillingCountry,
            Total
        FROM Invoice
        WHERE CustomerId = :cid
        ORDER BY InvoiceDate DESC
        LIMIT :limit
    """
    async with aconnect() as conn:
        cur = await conn.execute(sql, {"cid": customer_id, "limit": limit})
        rows = await cur.fetchall()

    foreign = [r["InvoiceId"] for r in rows if r["CustomerId"] != customer_id]
    if foreign:
        logger.error(
            "scope-violation: list_my_orders for customer_id=%s fetched invoices "
            "%s owned by another customer; refusing to return them",
            customer_id, foreign,
        )
        return (
            "I couldn't confirm those invoices belong to your account, so I "
            "haven't shown them. Please try again."
        )

    if not rows:
        return "You have no invoices on file."
    lines = [f"Your {len(rows)} most recent invoice(s):"]
    lines += [
        f"  - Invoice #{r['InvoiceId']} on {r['InvoiceDate'][:10]} "
        f"to {r['BillingCity']}, {r['BillingCountry']} - ${r['Total']:.2f}"
        for r in rows
    ]
    return "\n".join(lines)


@tool
async def get_invoice_details(
    invoice_id: int,
    runtime: ToolRuntime[UserContext],
) -> str:
    """Return line items for one of the authenticated customer's invoices.

    Refuses if the invoice belongs to a different customer.

    Args:
        invoice_id: ``Invoice.InvoiceId`` to fetch.
    """
    customer_id = _require_customer_id(runtime)
    if isinstance(customer_id, str):
        return customer_id

    logger.info(
        "account-tool: get_invoice_details customer_id=%s invoice_id=%s",
        customer_id, invoice_id,
    )

    async with aconnect() as conn:
        cur = await conn.execute(
            "SELECT InvoiceId, CustomerId, InvoiceDate, Total "
            "FROM Invoice WHERE InvoiceId = :iid",
            {"iid": invoice_id},
        )
        invoice = await cur.fetchone()
        if invoice is None:
            return f"No invoice found with id {invoice_id}."
        if invoice["CustomerId"] != customer_id:
            # Deliberately do not disclose whose invoice it is.
            return (
                f"Invoice {invoice_id} is not associated with your account. "
                "I can only show you invoices that belong to you."
            )

        cur = await conn.execute(
            """
            SELECT
                t.Name      AS track,
                ar.Name     AS artist,
                il.UnitPrice,
                il.Quantity
            FROM InvoiceLine il
            JOIN Invoice i  ON i.InvoiceId = il.InvoiceId
                           AND i.CustomerId = :cid
            JOIN Track  t  ON t.TrackId  = il.TrackId
            JOIN Album  a  ON a.AlbumId  = t.AlbumId
            JOIN Artist ar ON ar.ArtistId = a.ArtistId
            WHERE il.InvoiceId = :iid
            ORDER BY il.InvoiceLineId
            """,
            {"iid": invoice_id, "cid": customer_id},
        )
        lines = await cur.fetchall()

    header = (
        f"Invoice #{invoice['InvoiceId']} "
        f"({invoice['InvoiceDate'][:10]}) - total ${invoice['Total']:.2f}"
    )
    body = [
        f"  - '{r['track']}' by {r['artist']} "
        f"x{r['Quantity']} @ ${r['UnitPrice']:.2f}"
        for r in lines
    ]
    return "\n".join([header, *body]) if body else header


# ---------------------------------------------------------------------------
# lt memory (per-customer, cross-thread)
# ---------------------------------------------------------------------------
@tool
async def remember(
    fact: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """Save a durable fact about the authenticated customer.

    Use this when the customer states a preference, a recurring need, or
    any other detail that should persist across conversations (e.g.
    "I prefer jazz", "always email me invoices as PDF"). Do not use it
    for one-off task state; that belongs in short-term thread memory.

    Args:
        fact: A short, self-contained statement to remember.
    """
    customer_id = _require_customer_id(runtime)
    if isinstance(customer_id, str):
        return customer_id
    if runtime.store is None:
        return "Long-term memory is not configured for this deployment."

    key = await Memo(runtime.store).write(customer_id, fact)
    return f"Saved (id={key[:8]})."


@tool
async def recall(
    query: str,
    runtime: ToolRuntime[UserContext],
    limit: Limit = 3,
) -> str:
    """Search the authenticated customer's saved memories.

    Returns the top matches by semantic similarity when the store has an
    embedding index, otherwise the most recent items in the namespace.

    Args:
        query: What to look for (e.g. "favourite genre", "shipping
            preferences").
        limit: Maximum number of memories to return.
    """
    customer_id = _require_customer_id(runtime)
    if isinstance(customer_id, str):
        return customer_id
    if runtime.store is None:
        return "Long-term memory is not configured for this deployment."

    hits = await Memo(runtime.store).search(customer_id, query, limit=limit)
    if not hits:
        return "No memories on file for this customer."
    return "\n".join(f"  - {h.value.get('text', h.value)}" for h in hits)


# ---------------------------------------------------------------------------
# LLM delegation
# ---------------------------------------------------------------------------
@tool
async def ask_music_expert(question: str) -> str:
    """Ask a second LLM to explain a music-related topic.

    Use this for music concepts, comparisons, and recommendation rationale
    that benefit from specialist reasoning. Do not use it for Chinook orders,
    customer data, or claims about what is currently in the Chinook catalog.

    Args:
        question: A self-contained, music-related question for the specialist.
    """
    response = await music_expert_model.ainvoke(
        [
            (
                "system",
                "You are a concise music expert supporting a digital music "
                "store agent. Answer only the music-related question. Do not "
                "claim access to the store catalog, customer data, current "
                "events, or external tools.",
            ),
            ("human", question),
        ]
    )
    return response.text


ALL_TOOLS = [
    find_similar_albums,
    popular_in_genre,
    list_my_orders,
    get_invoice_details,
    remember,
    recall,
    ask_music_expert,
]
