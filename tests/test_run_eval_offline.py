from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest


SCRIPT = Path(__file__).parents[1] / "evals" / "run-eval-offline.py"


@pytest.fixture(scope="module")
def run_eval_offline() -> Any:
    spec = importlib.util.spec_from_file_location("run_eval_offline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Client:
    def __init__(self, datasets: list[Any], examples: list[Any] | None = None) -> None:
        self.datasets = datasets
        self.examples = examples or []
        self.created_dataset: dict[str, Any] | None = None
        self.deleted_example_ids: list[str] = []
        self.created_examples: dict[str, Any] | None = None

    def list_datasets(self, *, dataset_name: str) -> list[Any]:
        return self.datasets

    def create_dataset(self, **kwargs: Any) -> Any:
        self.created_dataset = kwargs
        return SimpleNamespace(id="dataset-new")

    def list_examples(self, *, dataset_id: str) -> list[Any]:
        return self.examples

    def delete_example(self, example_id: str) -> None:
        self.deleted_example_ids.append(example_id)

    def create_examples(self, **kwargs: Any) -> None:
        self.created_examples = kwargs


async def test_target_passes_question_once_and_keeps_context_for_identity(
    run_eval_offline: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "I prefer jazz and PDF invoices."
    graph = SimpleNamespace(ainvoke=AsyncMock(return_value={
        "messages": [{"role": "assistant", "content": "Saved your preferences."}],
    }))
    monkeypatch.setitem(sys.modules, "agent", SimpleNamespace(graph=graph))

    result = await run_eval_offline._target({
        "question": question,
        "context": {"customer_id": 2},
    })

    graph.ainvoke.assert_awaited_once_with(
        {"messages": [{"role": "user", "content": question}]},
        context=run_eval_offline.UserContext(customer_id=2),
    )
    assert result["answer"] == "Saved your preferences."


def test_upsert_dataset_creates_dataset_and_examples(run_eval_offline: Any) -> None:
    client = _Client([])

    dataset_id, action = run_eval_offline._upsert_dataset(client, "my-dataset")

    assert (dataset_id, action) == ("dataset-new", "Created")
    assert client.created_dataset == {
        "dataset_name": "my-dataset",
        "description": run_eval_offline.DATASET_DESCRIPTION,
    }
    assert client.created_examples["dataset_id"] == "dataset-new"
    assert len(client.created_examples["inputs"]) > 0


def test_upsert_dataset_replaces_existing_examples(run_eval_offline: Any) -> None:
    client = _Client(
        [SimpleNamespace(id="dataset-existing")],
        [SimpleNamespace(id="example-1"), SimpleNamespace(id="example-2")],
    )

    dataset_id, action = run_eval_offline._upsert_dataset(client, "my-dataset")

    assert (dataset_id, action) == ("dataset-existing", "Updated")
    assert client.created_dataset is None
    assert client.deleted_example_ids == ["example-1", "example-2"]
    assert client.created_examples["dataset_id"] == "dataset-existing"


@pytest.mark.parametrize(
    ("builder_name", "feedback_key"),
    [
        ("_judge_chain", "llm_correctness"),
        ("_conciseness_chain", "llm_conciseness"),
    ],
)
def test_hosted_judges_upload_gateway_and_secret_reference(
    run_eval_offline: Any,
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    feedback_key: str,
) -> None:
    local_api_key = "test-local-key-not-a-secret"
    monkeypatch.setattr(run_eval_offline.os, "environ", {
        "OPENAI_API_KEY": local_api_key,
        "OPENAI_BASE_URL": "https://local-model.example.invalid/v1",
    })
    manifest = getattr(run_eval_offline, builder_name)("gpt-4o-mini")
    requests: list[dict[str, Any]] = []

    def request(method: str, path: str, **kwargs: Any) -> Any:
        requests.append({"method": method, "path": path, **kwargs})
        return SimpleNamespace(json=lambda: {
            "commit": {"commit_hash": "new-commit", "id": "commit-id"},
        })

    # Exercise the real SDK upload code without creating an authenticated client.
    client = SimpleNamespace(
        _prompt_exists=lambda *args: True,
        _get_latest_commit_hash=lambda *args: "previous-commit",
        request_with_retries=request,
        _get_prompt_url=lambda identifier: identifier,
    )
    client.create_commit = lambda *args, **kwargs: (
        run_eval_offline.Client.create_commit(client, *args, **kwargs)
    )

    run_eval_offline.Client.push_prompt(
        client, "test-owner/judge", object=manifest,
    )

    assert len(requests) == 1
    assert requests[0]["method"] == "POST"
    assert requests[0]["path"] == "/commits/test-owner/judge"
    uploaded = requests[0]["json"]["manifest"]
    assert uploaded is manifest
    assert uploaded["kwargs"]["first"]["kwargs"]["schema_"]["properties"][
        feedback_key
    ]["type"] == "boolean"
    model = uploaded["kwargs"]["last"]["kwargs"]["bound"]["kwargs"]
    assert model["model_name"] == "gpt-4o-mini"
    assert model["openai_api_base"] == (
        "https://gateway.smith.langchain.com/openai/v1"
    )
    assert model["openai_api_key"] == {
        "lc": 1,
        "type": "secret",
        "id": ["LC_GATEWAY_KEY"],
    }
    assert local_api_key not in json.dumps(uploaded)


@pytest.mark.parametrize("replace", [False, True], ids=["create", "update"])
@pytest.mark.parametrize(
    ("upsert_name", "builder_name", "expected_variables"),
    [
        (
            "_upsert_llm_evaluator",
            "_judge_chain",
            {
                "input": "What is 2 + 2?",
                "output": "2 + 2 = 4.",
                "reference": "The answer must be 4.",
            },
        ),
        (
            "_upsert_conciseness_evaluator",
            "_conciseness_chain",
            {"output": "2 + 2 = 4."},
        ),
    ],
)
async def test_llm_evaluator_mappings_resolve_hosted_run_data(
    run_eval_offline: Any,
    monkeypatch: pytest.MonkeyPatch,
    replace: bool,
    upsert_name: str,
    builder_name: str,
    expected_variables: dict[str, str],
) -> None:
    evaluator = SimpleNamespace(id="evaluator-1", type="llm")
    response = SimpleNamespace(evaluator=evaluator)
    client = SimpleNamespace(
        push_prompt=Mock(),
        get_prompt=Mock(return_value=SimpleNamespace(
            repo_handle="judge-prompt", last_commit_hash="judge-commit",
        )),
        evaluators=SimpleNamespace(
            create=AsyncMock(return_value=response),
            update=AsyncMock(return_value=response),
        ),
    )
    monkeypatch.setattr(run_eval_offline, builder_name, Mock(return_value={}))
    monkeypatch.setattr(
        run_eval_offline,
        "_find_evaluator",
        AsyncMock(return_value=evaluator if replace else None),
    )

    result, action = await getattr(run_eval_offline, upsert_name)(
        client, "gpt-4o-mini", replace=replace,
    )

    assert result is evaluator
    assert action == ("Updated" if replace else "Created")
    submitted = client.evaluators.update if replace else client.evaluators.create
    unused = client.evaluators.create if replace else client.evaluators.update
    submitted.assert_awaited_once()
    unused.assert_not_awaited()
    mapping = submitted.call_args.kwargs["llm_evaluator"]["variable_mapping"]

    # The hosted runtime exposes these singular roots, as verified by its
    # /runs/rules/validate endpoint. Plural inputs/outputs silently become None.
    hosted_context = {
        "input": {"question": "What is 2 + 2?"},
        "output": {"answer": "2 + 2 = 4."},
        "reference": {"reference_answer": "The answer must be 4."},
    }
    resolved_variables = {}
    for variable, path in mapping.items():
        source, key = path.split(".", 1)
        resolved_variables[variable] = hosted_context.get(source, {}).get(key)
    assert resolved_variables == expected_variables


@pytest.mark.parametrize("agent_id", [None, "fm-demo-app"])
def test_amain_configures_evaluators_before_running_experiment(
    run_eval_offline: Any,
    monkeypatch: pytest.MonkeyPatch,
    agent_id: str | None,
) -> None:
    events: list[str] = []
    client = SimpleNamespace(close=lambda **kwargs: events.append("close"))
    evaluator = SimpleNamespace(id="evaluator-1", name="judge")

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.setattr(run_eval_offline, "Client", lambda: client)
    monkeypatch.setattr(run_eval_offline, "CODE_EVALUATORS", {})

    def resolve_agent(selected_client: Any, selected_agent: str) -> str:
        assert selected_client is client
        assert selected_agent == agent_id
        events.append("agent")
        return "agent-tag"

    def associate_dataset(selected_client: Any, dataset_id: str, tag_id: str) -> bool:
        assert selected_client is client
        assert (dataset_id, tag_id) == ("dataset-1", "agent-tag")
        events.append("association")
        return True

    monkeypatch.setattr(run_eval_offline, "get_agent_tag_value_id", resolve_agent)
    monkeypatch.setattr(run_eval_offline, "ensure_dataset_agent_tag", associate_dataset)

    def upsert_dataset(selected_client: Any, name: str) -> tuple[str, str]:
        events.append("dataset")
        return "dataset-1", "Updated"

    monkeypatch.setattr(
        run_eval_offline,
        "_upsert_dataset",
        upsert_dataset,
    )

    async def upsert_evaluator(*args: Any) -> tuple[Any, str]:
        events.append("evaluator")
        return evaluator, "Reused"

    monkeypatch.setattr(run_eval_offline, "_upsert_llm_evaluator", upsert_evaluator)
    monkeypatch.setattr(
        run_eval_offline,
        "_upsert_conciseness_evaluator",
        upsert_evaluator,
    )
    monkeypatch.setattr(
        run_eval_offline,
        "_bind_to_dataset",
        lambda *args: events.append("binding") or "Updated binding",
    )

    async def evaluate(*args: Any, **kwargs: Any) -> str:
        events.append("experiment")
        assert args == (run_eval_offline._target,)
        assert kwargs == {
            "data": "my-dataset",
            "experiment_prefix": "my-experiment",
            "metadata": {
            "application": None,
            "environment": "offline-evaluation",
            },            
            "client": client,
        }
        return "experiment-result"

    monkeypatch.setattr(run_eval_offline, "aevaluate", evaluate)
    args = SimpleNamespace(
        agent=agent_id,
        dataset_name="my-dataset",
        judge_model="judge-model",
        experiment_prefix="my-experiment",
        replace=False,
    )

    assert asyncio.run(run_eval_offline.amain(args)) == 0
    assert events == [
        *(["agent"] if agent_id else []),
        "dataset",
        *(["association"] if agent_id else []),
        "evaluator",
        "evaluator",
        "binding",
        "binding",
        "experiment",
        "close",
    ]


def test_unknown_agent_stops_before_dataset_replacement(
    run_eval_offline: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = SimpleNamespace(close=lambda **kwargs: events.append("close"))
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(run_eval_offline, "Client", lambda: client)

    def resolve_agent(*args: Any) -> str:
        raise ValueError("Agent not found")

    monkeypatch.setattr(run_eval_offline, "get_agent_tag_value_id", resolve_agent)
    monkeypatch.setattr(
        run_eval_offline,
        "_upsert_dataset",
        lambda *args: pytest.fail("Must resolve the Agent before replacing examples"),
    )
    with pytest.raises(ValueError, match="Agent not found"):
        asyncio.run(run_eval_offline.amain(SimpleNamespace(agent="missing-agent")))

    assert events == ["close"]
