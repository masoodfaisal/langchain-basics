from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def test_amain_configures_evaluators_before_running_experiment(
    run_eval_offline: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = SimpleNamespace(close=lambda **kwargs: events.append("close"))
    evaluator = SimpleNamespace(id="evaluator-1", name="judge")

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(run_eval_offline, "Client", lambda: client)
    monkeypatch.setattr(run_eval_offline, "CODE_EVALUATORS", {})

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
        dataset_name="my-dataset",
        judge_model="judge-model",
        experiment_prefix="my-experiment",
        replace=False,
    )

    assert asyncio.run(run_eval_offline.amain(args)) == 0
    assert events == [
        "dataset",
        "evaluator",
        "evaluator",
        "binding",
        "binding",
        "experiment",
        "close",
    ]
