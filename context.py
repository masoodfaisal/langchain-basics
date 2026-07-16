"""Invocation-time context schema.

LangChain v1 agents accept a ``context_schema`` dataclass that is supplied at
``invoke``/``stream`` time and is available to tools via ``runtime.context``.
We use it to carry the authenticated customer's id, so catalog tools can
personalise results and account tools can enforce ownership checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class UserContext:
    """Per-request context for the support bot.

    Attributes:
        customer_id: Chinook ``Customer.CustomerId`` of the authenticated
            caller. ``None`` means the caller is anonymous; account tools
            must refuse to answer in that case.
    """

    customer_id: Optional[int] = None
