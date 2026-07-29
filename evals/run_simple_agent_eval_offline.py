"""Run offline LangSmith evaluations for ``simple-agent.py``.

Run from the project root:

    PYTHONPATH=. .venv/bin/python evals/run_simple_agent_eval_offline.py --replace

The runner uploads the small dataset, registers and binds the reusable code and
LLM evaluators, then invokes the agent without context or middleware. It reads
credentials from environment variables or the project ``.env`` file without
printing their values.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client, aevaluate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.helpers import final_text
from evals.simple_agent_dataset import to_langsmith_examples
from evals.simple_agent_llm_judge import (
    DEFAULT_JUDGE_MODEL,
    upsert_llm_evaluator,
)


SIMPLE_AGENT_FILE = PROJECT_ROOT / "simple-agent.py"
EVALUATOR_FILE = PROJECT_ROOT / "evals" / "simple_agent_evaluators_offline.py"
DEFAULT_DATASET_NAME = "simple-chinook-agent"
REUSABLE_EVALUATOR_VERSION = 3
CODE_EVALUATORS = {
    "simple-agent-offline-tool-trajectory": "tool_trajectory",
    "simple-agent-offline-output-contains": "output_contains",
    "simple-agent-offline-output-excludes": "output_excludes",
}
DATASET_DESCRIPTION = (
    "Beginner Chinook agent examples covering music-expert delegation and "
    "out-of-scope refusal."
)


def _load_env() -> None:
    """Load local settings without logging any values."""
    load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def _load_graph() -> Any:
    """Load the graph from the hyphenated ``simple-agent.py`` filename."""
    spec = importlib.util.spec_from_file_location(
        "simple_agent_evaluation_target",
        SIMPLE_AGENT_FILE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SIMPLE_AGENT_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.graph


def _upsert_dataset(client: Client, name: str) -> tuple[str, str]:
    """Create the dataset, or replace its examples when it already exists."""
    datasets = list(client.list_datasets(dataset_name=name))
    if len(datasets) > 1:
        raise RuntimeError(f"Multiple datasets named {name!r} found.")

    if datasets:
        dataset = datasets[0]
        for example in client.list_examples(dataset_id=dataset.id):
            client.delete_example(example.id)
        action = "Updated"
    else:
        dataset = client.create_dataset(
            dataset_name=name,
            description=DATASET_DESCRIPTION,
        )
        action = "Created"

    rows = to_langsmith_examples()
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[row["inputs"] for row in rows],
        outputs=[row["outputs"] for row in rows],
        metadata=[row["metadata"] for row in rows],
    )
    return str(dataset.id), action


def _code_source(function_name: str) -> str:
    """Build one uploadable evaluator with LangSmith's required entry point."""
    source = EVALUATOR_FILE.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(EVALUATOR_FILE))
    functions = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function_name not in functions:
        raise ValueError(f"Function {function_name!r} not found in {EVALUATOR_FILE}")
    return (
        f"{source}\n\n"
        "def perform_eval(run, example):\n"
        f"    return {function_name}(run, example)\n"
    )


async def _find_evaluator(client: Client, name: str) -> Any | None:
    matches = [
        evaluator
        async for evaluator in client.evaluators.list(name_contains=name, limit=100)
        if evaluator.name == name
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple evaluators named {name!r}.")
    return matches[0] if matches else None


async def _upsert_code_evaluator(
    client: Client,
    name: str,
    function_name: str,
    replace: bool,
) -> tuple[Any, str]:
    """Create or update one reusable evaluator in LangSmith."""
    existing = await _find_evaluator(client, name)
    code_evaluator = {
        "code": _code_source(function_name),
        "language": "python",
    }

    if existing is None:
        result = await client.evaluators.create(
            name=name,
            type="code",
            code_evaluator=code_evaluator,
        )
        action = "Created"
    else:
        if existing.type != "code":
            raise RuntimeError(f"Evaluator {name!r} exists but is not code-based.")
        if not replace:
            return existing, "Reused"
        result = await client.evaluators.update(
            str(existing.id),
            name=name,
            code_evaluator=code_evaluator,
        )
        action = "Updated"

    evaluator = result.evaluator
    if evaluator is None or evaluator.id is None:
        raise RuntimeError(f"LangSmith did not return an id for {name!r}.")
    return evaluator, action


def _matching_rules(
    client: Client,
    dataset_id: str,
    evaluator_id: str,
) -> list[dict[str, Any]]:
    response = client.request_with_retries(
        "GET",
        "/api/v1/runs/rules",
        request_kwargs={
            "params": {
                "dataset_id": dataset_id,
                "evaluator_id": evaluator_id,
            }
        },
        stop_after_attempt=3,
    )
    body = response.json()
    items = body if isinstance(body, list) else body.get("items", [])
    if not isinstance(items, list):
        raise RuntimeError("LangSmith returned an invalid run-rule response.")
    return [
        rule
        for rule in items
        if isinstance(rule, dict)
        and str(rule.get("dataset_id")) == dataset_id
        and str(rule.get("evaluator_id")) == evaluator_id
    ]


def _bind_to_dataset(client: Client, dataset_id: str, evaluator: Any) -> str:
    """Attach a reusable evaluator to this dataset at 100% sampling."""
    evaluator_id = str(evaluator.id)
    rules = _matching_rules(client, dataset_id, evaluator_id)
    if len(rules) > 1:
        raise RuntimeError(
            f"Found duplicate dataset rules for evaluator {evaluator.name!r}."
        )

    payload = {
        "display_name": evaluator.name,
        "sampling_rate": 1.0,
        "dataset_id": dataset_id,
        "evaluator_id": evaluator_id,
        "evaluator_version": REUSABLE_EVALUATOR_VERSION,
        "is_enabled": True,
    }
    if rules:
        rule_id = rules[0].get("id")
        if not rule_id:
            raise RuntimeError("LangSmith returned a run rule without an id.")
        client.request_with_retries(
            "PATCH",
            f"/api/v1/runs/rules/{rule_id}",
            request_kwargs={"json": payload},
            stop_after_attempt=3,
        )
        return "Updated binding"

    client.request_with_retries(
        "POST",
        "/api/v1/runs/rules",
        request_kwargs={"json": payload},
        stop_after_attempt=3,
    )
    return "Created binding"


async def _target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run one dataset question through the minimal agent."""
    result = await _load_graph().ainvoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]}
    )
    return {**result, "answer": final_text(result)}


async def amain(args: argparse.Namespace) -> int:
    if not os.getenv("LANGSMITH_API_KEY"):
        print(
            "LANGSMITH_API_KEY is not configured.",
            file=sys.stderr,
        )
        return 2

    client = Client()
    try:
        dataset_id, action = _upsert_dataset(client, args.dataset_name)
        print(f"{action} dataset {args.dataset_name!r} ({dataset_id}).")

        for name, function_name in CODE_EVALUATORS.items():
            evaluator, evaluator_action = await _upsert_code_evaluator(
                client,
                name,
                function_name,
                args.replace,
            )
            binding_action = _bind_to_dataset(client, dataset_id, evaluator)
            print(f"{evaluator_action} {evaluator.name!r}; {binding_action}.")

        llm_evaluator, evaluator_action = await upsert_llm_evaluator(
            client,
            args.judge_model,
            args.replace,
        )
        binding_action = _bind_to_dataset(client, dataset_id, llm_evaluator)
        print(f"{evaluator_action} {llm_evaluator.name!r}; {binding_action}.")

        experiment = await aevaluate(
            _target,
            data=args.dataset_name,
            experiment_prefix=args.experiment_prefix,
            metadata={
                "application": "simple-agent",
                "environment": "offline-evaluation",
            },
            max_concurrency=args.max_concurrency,
            client=client,
        )
        print(f"Experiment complete: {experiment}")
    finally:
        client.close(timeout=0)
    return 0


def main() -> int:
    _load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("SIMPLE_AGENT_DATASET_NAME", DEFAULT_DATASET_NAME),
    )
    parser.add_argument(
        "--experiment-prefix",
        default="simple-chinook-agent",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help="Model identifier used by the LLM correctness judge.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Maximum number of examples evaluated at the same time.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Update reusable evaluators that already exist in LangSmith.",
    )
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
