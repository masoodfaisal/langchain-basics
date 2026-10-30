from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from evals import agent_associations


AGENT_ID = "fm-demo-app"
DATASET_ID = "e09be4b2-d7b3-4c78-91fb-e16494ce9372"
TAG_VALUE_ID = "5c0b4e09-c4aa-48b5-8414-d160648de123"
OTHER_TAG_ID = "21d22d73-8621-4095-91a6-ae3f6dfb4421"


class _Response:
    def __init__(self, body: Any) -> None:
        self.body = body

    def json(self) -> Any:
        return deepcopy(self.body)


class _Client:
    def __init__(self, *, tags: Any = None, agent: Any = None) -> None:
        self.tags = [] if tags is None else tags
        self.agent = (
            {"id": AGENT_ID, "tag_value_id": TAG_VALUE_ID}
            if agent is None
            else agent
        )
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.dataset_names: list[str] = []
        self.closed = False

    def request_with_retries(
        self, method: str, path: str, **kwargs: Any
    ) -> _Response:
        self.requests.append((method, path, kwargs))
        if method == "GET" and path == f"/api/v1/agents/{AGENT_ID}":
            return _Response(self.agent)
        if method == "GET" and path == agent_associations.TAGS_PATH:
            assert kwargs["request_kwargs"]["params"] == {
                "resource_type": "dataset",
                "resource_id": DATASET_ID,
            }
            return _Response(self.tags)
        if method == "POST" and path == agent_associations.TAGGINGS_PATH:
            payload = kwargs["request_kwargs"]["json"]
            assert payload == {
                "tag_value_id": TAG_VALUE_ID,
                "resource_type": "dataset",
                "resource_id": DATASET_ID,
            }
            self.tags.append({"key": "Agent", "values": [{"id": TAG_VALUE_ID}]})
            return _Response(payload)
        raise AssertionError(f"Unexpected request: {method} {path}")

    def read_dataset(self, *, dataset_name: str) -> Any:
        self.dataset_names.append(dataset_name)
        return SimpleNamespace(id=DATASET_ID)

    def close(self, **kwargs: Any) -> None:
        self.closed = True


def test_get_agent_tag_reads_exact_agent() -> None:
    client = _Client()

    assert agent_associations.get_agent_tag_value_id(client, AGENT_ID) == TAG_VALUE_ID
    assert [(method, path) for method, path, _ in client.requests] == [
        ("GET", f"/api/v1/agents/{AGENT_ID}")
    ]


@pytest.mark.parametrize("tag_value_id", [None, "", 123, "not-a-uuid"])
def test_agent_without_valid_tag_fails_before_any_mutation(tag_value_id: Any) -> None:
    client = _Client(agent={"id": AGENT_ID, "tag_value_id": tag_value_id})

    with pytest.raises(RuntimeError, match="no valid resource tag"):
        agent_associations.get_agent_tag_value_id(client, AGENT_ID)

    assert all(method == "GET" for method, _, _ in client.requests)


@pytest.mark.parametrize(
    "agent", [[], {}, {"id": "another-agent", "tag_value_id": TAG_VALUE_ID}]
)
def test_invalid_or_mismatched_agent_response_is_rejected(agent: Any) -> None:
    with pytest.raises(RuntimeError, match="requested Agent"):
        agent_associations.get_agent_tag_value_id(_Client(agent=agent), AGENT_ID)


def test_missing_tag_is_added_once_without_removing_unrelated_tags() -> None:
    existing_tag = {
        "key": "Application",
        "values": [{"id": OTHER_TAG_ID, "value": "shared-application"}],
    }
    client = _Client(tags=[deepcopy(existing_tag)])

    assert agent_associations.ensure_dataset_agent_tag(client, DATASET_ID, TAG_VALUE_ID)
    assert not agent_associations.ensure_dataset_agent_tag(
        client, DATASET_ID, TAG_VALUE_ID
    )

    assert client.tags[0] == existing_tag
    writes = [
        (method, path) for method, path, _ in client.requests if method != "GET"
    ]
    assert writes == [("POST", agent_associations.TAGGINGS_PATH)]


def test_existing_tag_does_not_create_a_duplicate() -> None:
    client = _Client(tags=[{"key": "Agent", "values": [{"id": TAG_VALUE_ID}]}])

    assert not agent_associations.ensure_dataset_agent_tag(
        client, DATASET_ID, TAG_VALUE_ID
    )

    assert all(method == "GET" for method, _, _ in client.requests)


@pytest.mark.parametrize(
    "tags", [{}, [None], [{"values": {}}], [{"values": [None]}]]
)
def test_invalid_dataset_tags_fail_before_mutation(tags: Any) -> None:
    client = _Client(tags=tags)

    with pytest.raises(RuntimeError, match="invalid dataset resource tags"):
        agent_associations.ensure_dataset_agent_tag(client, DATASET_ID, TAG_VALUE_ID)

    assert all(method == "GET" for method, _, _ in client.requests)


def test_cli_only_links_the_named_existing_dataset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _Client()
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(agent_associations, "load_dotenv", lambda path: None)
    monkeypatch.setattr(agent_associations, "Client", lambda: client)

    result = agent_associations.main(
        ["--agent", AGENT_ID, "--dataset-name", "my-existing-dataset"]
    )

    assert result == 0
    assert client.dataset_names == ["my-existing-dataset"]
    assert client.closed
    assert (
        "Linked dataset 'my-existing-dataset' to Agent 'fm-demo-app'."
        in capsys.readouterr().out
    )
    assert [(method, path) for method, path, _ in client.requests] == [
        ("GET", f"/api/v1/agents/{AGENT_ID}"),
        ("GET", agent_associations.TAGS_PATH),
        ("POST", agent_associations.TAGGINGS_PATH),
    ]
