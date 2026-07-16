"""Upload the Chinook online evaluator to a LangSmith tracing project.

This is the online counterpart to ``evals/run-eval-offline.py``. It does not run a
dataset experiment. Instead, it uploads ``evals/evaluators-online.py`` as a
LangSmith online code evaluator and attaches it to a tracing project so it can
score live or historical traces.

Examples:

    # Upload and attach to the default LANGSMITH_PROJECT, or "agent".
    PYTHONPATH=. .venv/bin/python evals/run-eval-online.py --replace

    # Upload to a specific tracing project.
    PYTHONPATH=. .venv/bin/python evals/run-eval-online.py \
      --project agent --sampling-rate 1 --replace
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_FILE = PROJECT_ROOT / "evals" / "evaluators-online.py"
DEFAULT_EVALUATOR_NAME = "chinook-online-guardrails"
REUSABLE_EVALUATOR_VERSION = 3


def _load_env() -> None:
    # Reads the same local env file as the offline evaluator runner without
    # printing any values. Secrets stay in the environment.
    load_dotenv(PROJECT_ROOT / ".env")


def _sampling_rate(value: str) -> float:
    rate = float(value)
    if not 0 <= rate <= 1:
        raise argparse.ArgumentTypeError("sampling rate must be between 0 and 1")
    return rate


def _evaluator_source(function_name: str) -> str:
    source = EVALUATOR_FILE.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(EVALUATOR_FILE))
    functions = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function_name not in functions:
        raise ValueError(
            f"Function {function_name!r} was not found in {EVALUATOR_FILE}"
        )
    if function_name != "perform_eval":
        source += f"\n\nperform_eval = {function_name}\n"
    return source


async def _find_evaluator(client: Client, name: str) -> Any | None:
    matches = [
        evaluator
        async for evaluator in client.evaluators.list(
            name_contains=name,
            limit=100,
        )
        if evaluator.name == name
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple evaluators named {name!r}; "
            "rename or remove duplicates first."
        )
    return matches[0] if matches else None


async def _create_or_update_evaluator(
    client: Client,
    args: argparse.Namespace,
    source: str,
) -> tuple[Any, str]:
    existing = await _find_evaluator(client, args.name)
    code_evaluator = {"code": source, "language": "python"}

    if existing is None:
        result = await client.evaluators.create(
            name=args.name,
            type="code",
            code_evaluator=code_evaluator,
        )
        action = "Created"
    else:
        if existing.type != "code":
            raise RuntimeError(
                f"An evaluator named {args.name!r} exists but is not a code evaluator."
            )
        if existing.id is None:
            raise RuntimeError("LangSmith returned an evaluator without an ID.")
        if not args.replace:
            raise RuntimeError(
                f"Evaluator {args.name!r} already exists. Pass --replace to update it."
            )
        result = await client.evaluators.update(
            existing.id,
            name=args.name,
            code_evaluator=code_evaluator,
        )
        action = "Updated"

    evaluator = result.evaluator
    if evaluator is None or evaluator.id is None:
        raise RuntimeError("LangSmith did not return an evaluator ID.")
    return evaluator, action


def _matching_rules(
    client: Client,
    project_id: str,
    evaluator_id: str,
) -> list[dict[str, Any]]:
    response = client.request_with_retries(
        "GET",
        "/api/v1/runs/rules",
        request_kwargs={
            "params": {
                "session_id": project_id,
                "evaluator_id": evaluator_id,
            }
        },
        stop_after_attempt=3,
    )
    body = response.json()
    if isinstance(body, list):
        items = body
    elif isinstance(body, dict):
        items = body.get("items", [])
    else:
        items = None
    if not isinstance(items, list) or not all(
        isinstance(rule, dict) for rule in items
    ):
        raise RuntimeError("LangSmith returned an invalid run-rule response.")
    return [
        rule
        for rule in items
        if str(rule.get("session_id")) == project_id
        and str(rule.get("evaluator_id")) == evaluator_id
    ]


def _attach_evaluator(
    client: Client,
    args: argparse.Namespace,
    project_id: str,
    evaluator_id: str,
) -> str:
    rules = _matching_rules(client, project_id, evaluator_id)
    if len(rules) > 1:
        raise RuntimeError(
            "Found multiple run rules for this evaluator and project; "
            "remove duplicates first."
        )

    payload = {
        "display_name": args.name,
        "sampling_rate": args.sampling_rate,
        "session_id": project_id,
        "evaluator_id": evaluator_id,
        "evaluator_version": REUSABLE_EVALUATOR_VERSION,
        "is_enabled": True,
    }
    if rules:
        rule_id = rules[0].get("id")
        if not rule_id:
            raise RuntimeError("LangSmith returned a run rule without an ID.")
        client.request_with_retries(
            "PATCH",
            f"/api/v1/runs/rules/{rule_id}",
            request_kwargs={"json": payload},
            stop_after_attempt=3,
        )
        return "Updated"

    client.request_with_retries(
        "POST",
        "/api/v1/runs/rules",
        request_kwargs={"json": payload},
        stop_after_attempt=3,
    )
    return "Created"


async def upload_and_attach(args: argparse.Namespace) -> int:
    source = _evaluator_source(args.function)
    client = Client()
    try:
        project = client.read_project(project_name=args.project)
        if project.id is None:
            raise RuntimeError("LangSmith did not return a tracing project ID.")
        project_id = str(project.id)
        evaluator, evaluator_action = await _create_or_update_evaluator(
            client,
            args,
            source,
        )
        rule_action = _attach_evaluator(
            client,
            args,
            project_id,
            str(evaluator.id),
        )
    finally:
        client.close(timeout=0)

    print(f"{evaluator_action} evaluator {args.name!r} ({evaluator.id}).")
    print(
        f"{rule_action} its run rule for project {args.project!r} "
        f"with sampling rate {args.sampling_rate}."
    )
    return 0


def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=os.getenv("LANGSMITH_PROJECT", "agent"),
        help="LangSmith tracing project to attach the online evaluator to.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_EVALUATOR_NAME,
        help="Online evaluator name shown in LangSmith.",
    )
    parser.add_argument(
        "--function",
        default="perform_eval",
        help="Function inside evaluators-online.py to upload.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=_sampling_rate,
        default=1.0,
        help="Fraction of matching production runs to evaluate.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite an existing evaluator with the same name.",
    )
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print(
            "LANGSMITH_API_KEY is not set. Put it in .env or export it "
            "before uploading.",
            file=sys.stderr,
        )
        return 2

    try:
        return asyncio.run(upload_and_attach(args))
    except Exception as exc:
        print(f"Could not configure the online evaluator: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
