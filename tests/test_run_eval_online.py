from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


SCRIPT = Path(__file__).parents[1] / "evals" / "run-eval-online.py"


@pytest.fixture(scope="module")
def run_eval_online() -> Any:
    spec = importlib.util.spec_from_file_location("run_eval_online", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, body: Any) -> None:
        self._body = body

    def json(self) -> Any:
        return self._body


class _Client:
    def __init__(self, body: Any) -> None:
        self._body = body
        self.requests: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def request_with_retries(self, *args: Any, **kwargs: Any) -> _Response:
        self.requests.append((args, kwargs))
        return _Response(self._body)


@pytest.mark.parametrize(
    "body",
    [
        [
            {
                "id": "rule-1",
                "session_id": "project-1",
                "evaluator_id": "evaluator-1",
            }
        ],
        {
            "items": [
                {
                    "id": "rule-1",
                    "session_id": "project-1",
                    "evaluator_id": "evaluator-1",
                }
            ]
        },
    ],
)
def test_matching_rules_accepts_supported_response_shapes(
    run_eval_online: Any,
    body: Any,
) -> None:
    rules = run_eval_online._matching_rules(
        _Client(body),
        "project-1",
        "evaluator-1",
    )

    assert [rule["id"] for rule in rules] == ["rule-1"]


def test_matching_rules_rejects_invalid_response(run_eval_online: Any) -> None:
    with pytest.raises(RuntimeError, match="invalid run-rule response"):
        run_eval_online._matching_rules(
            _Client("not-a-rule-list"),
            "project-1",
            "evaluator-1",
        )


def test_attach_migrates_legacy_rule_to_reusable_version(
    run_eval_online: Any,
) -> None:
    client = _Client(
        [
            {
                "id": "rule-1",
                "session_id": "project-1",
                "evaluator_id": "evaluator-1",
                "evaluator_version": 2,
            }
        ]
    )
    args = type(
        "Args",
        (),
        {"name": "guardrails", "sampling_rate": 1.0},
    )()

    action = run_eval_online._attach_evaluator(
        client,
        args,
        "project-1",
        "evaluator-1",
    )

    assert action == "Updated"
    patch_args, patch_kwargs = client.requests[-1]
    assert patch_args[:2] == (
        "PATCH",
        "/api/v1/runs/rules/rule-1",
    )
    assert (
        patch_kwargs["request_kwargs"]["json"]["evaluator_version"]
        == run_eval_online.REUSABLE_EVALUATOR_VERSION
        == 3
    )
