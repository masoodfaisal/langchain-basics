"""Check invoice messages with Granite Guardian; optionally query Rego via OPA."""

import json
import os
from pathlib import Path
import re

import httpx

from guardian import GRANITE_JUDGE_PROMPT


BILLING_POLICY_PATH = Path(__file__).with_name("policies") / "billing_guardian.rego"
CRITERION_QUERY = "data.chinook.billing.criterion"
ALLOW_QUERY = "data.chinook.billing.allow"
SEND_DESCRIPTION = (
    "send_invoice_explanation(invoice_id, body) sends the complete message "
    "to the customer. It explains an invoice and cannot change account terms."
)


class OpaPolicy:
    """Use OPA's HTTP API instead of the local RegoPolicy interpreter."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def evaluate(self, query: str, input_data: dict | None = None):
        if query not in (CRITERION_QUERY, ALLOW_QUERY):
            raise ValueError("Unsupported billing policy query")
        path = "/v1/data/" + query.removeprefix("data.").replace(".", "/")
        if input_data is None:
            response = await self.client.get(path)
        else:
            response = await self.client.post(path, json={"input": input_data})
        response.raise_for_status()
        return response.json()["result"]


class BillingAssessor:
    """Ask Granite whether the proposed message meets the criterion."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.client = client
        self.model = model or os.getenv("GUARDIAN_MODEL", "guardian-local")
        self.base_url = base_url or os.getenv(
            "GUARDIAN_BASE_URL", "http://127.0.0.1:11434/v1/",
        )

    async def assess(self, packet: dict, criterion: str) -> bool:
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError("A message criterion is required")
        context = {
            "source_user_message": packet["source_user_message"],
            "invoice": packet["invoice"],
            "tool_description": SEND_DESCRIPTION,
        }
        proposed_tool = packet["tool"]

        messages = [
            {"role": "user", "content": json.dumps(context)},
            {"role": "assistant", "content": json.dumps(proposed_tool)},
            {"role": "user", "content": GRANITE_JUDGE_PROMPT.format(criterion=criterion)},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 128,
        }
        if self.client is not None:
            response = await self.client.post("chat/completions", json=payload)
        else:
            headers = {}
            if os.environ.get("GUARDIAN_API_KEY"):
                headers["Authorization"] = f"Bearer {os.environ['GUARDIAN_API_KEY']}"
            async with httpx.AsyncClient(
                base_url=self.base_url, headers=headers, timeout=10.0,
            ) as client:
                response = await client.post("chat/completions", json=payload)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        message = choice["message"]
        if (
            choice["finish_reason"] != "stop"
            or message.get("tool_calls")
            or message.get("refusal")
        ):
            raise ValueError("The assessment did not return a complete score")
        content = message["content"]

        score = re.fullmatch(
            r"\s*(?:<think>\s*</think>\s*)?<score>\s*(yes|no)\s*</score>\s*",
            content,
        )
        if score is None:
            raise ValueError("Expected a yes or no Granite score")
        return score[1] == "yes"
