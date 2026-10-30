"""Link an existing LangSmith dataset to an existing Agent.

Run from the project root:

    uv run python -m evals.agent_associations \
      --agent fm-demo-app --dataset-name chinook-support-bot

This only adds the Agent's resource tag to the dataset. It does not change
examples, evaluator definitions, or run rules, and does not run an experiment.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import UUID

from dotenv import load_dotenv
from langsmith import Client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAGS_PATH = "/api/v1/workspaces/current/tags/resource"
TAGGINGS_PATH = "/api/v1/workspaces/current/taggings"


def _uuid_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a UUID string.")
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be a UUID string.") from exc


def get_agent_tag_value_id(client: Client, agent_id: str) -> str:
    """Read the resource tag belonging to the exact existing Agent ID."""
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("Agent ID must not be empty.")
    response = client.request_with_retries(
        "GET",
        f"/api/v1/agents/{quote(agent_id, safe='')}",
        stop_after_attempt=3,
    )
    agent = response.json()
    if not isinstance(agent, dict) or agent.get("id") != agent_id:
        raise RuntimeError("LangSmith did not return the requested Agent.")
    try:
        return _uuid_string(agent.get("tag_value_id"), "Agent tag value ID")
    except ValueError as exc:
        raise RuntimeError(
            f"Agent {agent_id!r} has no valid resource tag value ID."
        ) from exc


def ensure_dataset_agent_tag(
    client: Client,
    dataset_id: str,
    tag_value_id: str,
) -> bool:
    """Add a missing Agent tag, preserving the dataset and its other tags.

    Return True when a tagging was created, or False when it already existed.
    """
    dataset_id = _uuid_string(dataset_id, "Dataset ID")
    tag_value_id = _uuid_string(tag_value_id, "Tag value ID")
    response = client.request_with_retries(
        "GET",
        TAGS_PATH,
        request_kwargs={
            "params": {"resource_type": "dataset", "resource_id": dataset_id}
        },
        stop_after_attempt=3,
    )
    tags = response.json()
    if not isinstance(tags, list):
        raise RuntimeError("LangSmith returned invalid dataset resource tags.")
    for tag in tags:
        values = tag.get("values", []) if isinstance(tag, dict) else None
        if not isinstance(values, list) or not all(
            isinstance(value, dict) for value in values
        ):
            raise RuntimeError("LangSmith returned invalid dataset resource tags.")
        if any(value.get("id") == tag_value_id for value in values):
            return False

    client.request_with_retries(
        "POST",
        TAGGINGS_PATH,
        request_kwargs={
            "json": {
                "tag_value_id": tag_value_id,
                "resource_type": "dataset",
                "resource_id": dataset_id,
            }
        },
        stop_after_attempt=3,
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="Existing LangSmith Agent ID.")
    parser.add_argument(
        "--dataset-name", required=True, help="Exact name of an existing dataset."
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not configured.", file=sys.stderr)
        return 2

    client = Client()
    try:
        tag_value_id = get_agent_tag_value_id(client, args.agent)
        dataset = client.read_dataset(dataset_name=args.dataset_name)
        created = ensure_dataset_agent_tag(client, str(dataset.id), tag_value_id)
    except Exception as exc:
        print(f"Could not link the dataset to the Agent: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close(timeout=0)

    action = "Linked" if created else "Already linked"
    print(f"{action} dataset {args.dataset_name!r} to Agent {args.agent!r}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
