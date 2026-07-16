"""Lightweight SQLite access layer for the Chinook sample DB.

Async-only: ``aconnect()`` is an async context manager yielding an
``aiosqlite`` connection whose ``row_factory`` returns rows as ``sqlite3.Row``
(so callers keep dict-style access). Tools and middleware use it via
``async with`` so connections are always closed.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

CHINOOK_DB_PATH = Path(os.getenv("CHINOOK_DB_PATH", "chinook.db"))


@asynccontextmanager
async def aconnect() -> AsyncIterator[aiosqlite.Connection]:
    """Yield an async SQLite connection with ``sqlite3.Row`` row factory.

    Raises a clear error if the Chinook DB has not been bootstrapped yet.
    """
    if not CHINOOK_DB_PATH.exists():
        raise FileNotFoundError(
            f"Chinook DB not found at {CHINOOK_DB_PATH}. "
            "Run `python scripts/bootstrap_chinook.py` first."
        )
    conn = await aiosqlite.connect(CHINOOK_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        await conn.close()
