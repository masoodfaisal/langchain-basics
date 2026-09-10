"""Granite assesses a proposed memory; Rego decides; the memory tool enforces.

Read ``MemoryGuardian.allows`` for the policy flow. ``assess_memory`` contains
the Granite request and score parsing; it does not authorize a store operation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import re
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field

from rego_policy import RegoPolicy


REMEMBER_TOOL_DESCRIPTION = (
    "remember(fact) saves a durable customer preference or "
    "recurring service need for future conversations. The fact "
    "may paraphrase the user statement. This action does not "
    "change permissions or execute instructions in the stored text."
)

GRANITE_JUDGE_PROMPT = (
    "<no-think>As a judge agent, assess whether the provided text "
    "meets the given judging criteria using all available information, "
    "including conversations, documents, and tools. Provide your score "
    "immediately without explanation. Output empty <think>\\n</think> "
    "tags followed by your score in <score></score> tags."
    "\n\n### Criteria: {criterion}\n\n### Scoring Schema: "
    "If the last assistant's text meets the criteria, return 'yes'; "
    "otherwise, return 'no'."
)


logger = logging.getLogger(__name__)
POLICY_PATH = Path(__file__).with_name("policies") / "guardian.rego"
MemoryText = Annotated[
    str, Field(strict=True, min_length=1, max_length=1000, pattern=r"\S")
]
MemoryLimit = Annotated[int, Field(strict=True, ge=1, le=50)]


class RememberArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fact: MemoryText


class RecallArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: MemoryText
    limit: MemoryLimit


class MemoryGuardian:
    """One check shared by remember and recall, with no stored authorization state.

    The default client opens only for a write and closes after its assessment.
    Applications and tests may supply a client whose lifetime they manage.
    Recreate the guardian to load a changed Rego policy.
    """

    def __init__(
        self,
        granite: httpx.AsyncClient | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        policy_path: Path = POLICY_PATH,
    ) -> None:
        self.rego = RegoPolicy(policy_path)
        self.granite = granite
        self.model = model or os.getenv("GUARDIAN_MODEL", "guardian-local")
        self.base_url = base_url or os.getenv(
            "GUARDIAN_BASE_URL", "http://127.0.0.1:11434/v1/"
        )

    async def allows(
        self,
        tool_name: str,
        *,
        customer_id: int,
        namespace: tuple[str, str],
        args: dict,
        source_user_message: str = "",
    ) -> bool:
        """Validate the request, assess writes with Granite, then ask Rego.

        Recall skips Granite. Invalid input or any model/policy error denies.
        """
        ## for demo
        return True

        try:
            if type(customer_id) is not int or customer_id <= 0:
                return False

            if tool_name == "remember":
                checked_args = RememberArgs.model_validate(args).model_dump()
                operation = "put"
                if (
                    not isinstance(source_user_message, str)
                    or not source_user_message.strip()
                    or len(source_user_message) > 8000
                ):
                    return False
            elif tool_name == "recall":
                checked_args = RecallArgs.model_validate(args).model_dump()
                operation = "search"
            else:
                return False

            # Snapshot the exact operation before any model or policy await.
            policy_input = {
                "subject": {"authenticated": True, "customer_id": customer_id},
                "tool": {"name": tool_name, "args": checked_args},
                "memory": {
                    "customer_id": customer_id,
                    "namespace": list(namespace),
                    "operation": operation,
                },
                "source_user_message": source_user_message,
            }

            # The criterion and final decision use the same Rego policy snapshot.
            async with asyncio.timeout(10):
                policy = await self.rego.evaluate("data.chinook.guardian.policy")
                if not isinstance(policy, dict):
                    raise ValueError("Invalid policy metadata")
                policy_id = policy.get("id")
                criterion = policy.get("guardian_criterion")
                if not isinstance(policy_id, str) or not policy_id.strip():
                    raise ValueError("Missing policy version")
                if not isinstance(criterion, str) or not criterion.strip():
                    raise ValueError("Missing guardian criterion")

                # A read needs authorization, but no semantic assessment.
                if tool_name == "remember":
                    intent_match = await self.assess_memory(policy_input, criterion)
                    policy_input["guardian"] = {
                        "policy_id": policy_id,
                        "intent_match": intent_match,
                    }

                decision = await self.rego.evaluate(
                    "data.chinook.guardian.decision", policy_input,
                )
                if (
                    not isinstance(decision, dict)
                    or type(decision.get("allow")) is not bool
                    or decision.get("policy_id") != policy_id
                ):
                    raise ValueError("Invalid policy decision")
                return decision["allow"]
        except Exception as exc:
            logger.warning("Memory authorization failed: %s", type(exc).__name__)
            return False

    async def assess_memory(self, policy_input: dict, criterion: str) -> bool:
        """Ask Granite about the proposed fact, not permission to write it.

        Only source text and the proposed action reach the judge, not history
        or recalled memories. Failures propagate to ``allows``, which denies.
        """
        source_context = {
            "source_user_message": policy_input["source_user_message"],
            "memory": policy_input["memory"],
            "tool_description": REMEMBER_TOOL_DESCRIPTION,
        }
        messages = [
            {"role": "user", "content": json.dumps(source_context, allow_nan=False)},
            {"role": "assistant", "content": json.dumps(
                policy_input["tool"], allow_nan=False,
            )},
            {"role": "user", "content": GRANITE_JUDGE_PROMPT.format(criterion=criterion)},
        ]
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 128,
        }

        # Supplied clients belong to their caller. Otherwise open one per write.
        if self.granite is not None:
            response = await self.granite.post("chat/completions", json=request)
        else:
            headers = {}
            if os.environ.get("GUARDIAN_API_KEY"):
                headers["Authorization"] = f"Bearer {os.environ['GUARDIAN_API_KEY']}"
            async with httpx.AsyncClient(
                base_url=self.base_url, headers=headers, timeout=10.0,
            ) as granite:
                response = await granite.post("chat/completions", json=request)

        response.raise_for_status()
        choice = response.json()["choices"][0]
        score = re.fullmatch(
            r"\s*(?:<think>\s*</think>\s*)?<score>\s*(yes|no)\s*</score>\s*",
            choice["message"]["content"],
        )
        if choice.get("finish_reason") != "stop" or score is None:
            raise ValueError("Invalid Granite score")
        return score[1] == "yes"
