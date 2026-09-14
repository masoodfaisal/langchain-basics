"""Check the Rego rules and HTTP adapters without calling live services."""

import json
from types import SimpleNamespace

import httpx
import pytest

from billing_policy import (
    ALLOW_QUERY,
    BILLING_POLICY_PATH,
    CRITERION_QUERY,
    BillingAssessor,
    OpaPolicy,
)
from rego_policy import RegoPolicy


@pytest.fixture(scope="module")
def policy():
    return RegoPolicy(BILLING_POLICY_PATH)


@pytest.fixture
def packet():
    return {
        "source_user_message": "Please email me a breakdown of invoice 42.",
        "invoice": {
            "invoice_id": 42, "invoice_date": "2026-09-01", "total": "1.98",
            "lines": [{"track": "Music track", "unit_price": "0.99", "quantity": 2}],
        },
        "tool": {
            "name": "send_invoice_explanation",
            "args": {"invoice_id": 42, "body": "Invoice 42: Music track, 2 at 0.99; total 1.98."},
        },
        "guardian": {"intent_match": True},
    }


async def test_real_policy_exposes_criterion_and_boolean_decision(policy, packet):
    criterion = await policy.evaluate(CRITERION_QUERY)
    assert isinstance(criterion, str) and criterion.strip()
    assert await policy.evaluate(ALLOW_QUERY, packet) is True
    packet["guardian"]["intent_match"] = False
    assert await policy.evaluate(ALLOW_QUERY, packet) is False


@pytest.mark.parametrize("value", [False, "true", 1, None])
async def test_real_policy_requires_positive_boolean_assessment(policy, packet, value):
    packet["guardian"]["intent_match"] = value
    assert await policy.evaluate(ALLOW_QUERY, packet) is False


@pytest.mark.parametrize("field", ["source_user_message", "invoice", "tool", "guardian"])
async def test_real_policy_denies_missing_input(policy, packet, field):
    del packet[field]
    assert await policy.evaluate(ALLOW_QUERY, packet) is False


async def test_real_policy_requires_the_send_tool_and_matching_invoice(policy, packet):
    packet["tool"]["name"] = "get_invoice_for_explanation"
    assert await policy.evaluate(ALLOW_QUERY, packet) is False
    packet["tool"]["name"] = "send_invoice_explanation"
    packet["tool"]["args"]["invoice_id"] = 43
    assert await policy.evaluate(ALLOW_QUERY, packet) is False


@pytest.mark.parametrize("value", ["", " \t\n"])
async def test_real_policy_requires_message_and_user_request(policy, packet, value):
    packet["tool"]["args"]["body"] = value
    assert await policy.evaluate(ALLOW_QUERY, packet) is False
    packet["tool"]["args"]["body"] = "Invoice 42 totals 1.98."
    packet["source_user_message"] = value
    assert await policy.evaluate(ALLOW_QUERY, packet) is False


@pytest.fixture
async def gateway():
    state = SimpleNamespace(requests=[], response=httpx.Response(200, json={}), error=None)

    async def transport(request):
        state.requests.append(request)
        if state.error:
            raise state.error
        return state.response

    async with httpx.AsyncClient(
        base_url="http://service.test/v1/", transport=httpx.MockTransport(transport),
    ) as client:
        state.client = client
        yield state


async def test_opa_gets_criterion_and_posts_input(gateway, packet):
    gateway.client.base_url = "http://service.test/"
    adapter = OpaPolicy(gateway.client)
    gateway.response = httpx.Response(200, json={"result": "Explain only the invoice."})
    assert await adapter.evaluate(CRITERION_QUERY) == "Explain only the invoice."
    gateway.response = httpx.Response(200, json={"result": False})
    assert await adapter.evaluate(ALLOW_QUERY, packet) is False

    get, post = gateway.requests
    assert get.method == "GET"
    assert get.url.path == "/v1/data/chinook/billing/criterion"
    assert post.method == "POST"
    assert post.url.path == "/v1/data/chinook/billing/allow"
    assert json.loads(post.content) == {"input": packet}
    assert not gateway.client.is_closed


async def test_opa_rejects_unsupported_queries(gateway):
    with pytest.raises(ValueError):
        await OpaPolicy(gateway.client).evaluate("data.chinook.billing.allow; true")
    assert gateway.requests == []


@pytest.mark.parametrize("response, error", [
    (httpx.Response(200, json={}), KeyError),
    (httpx.Response(200, text="not JSON"), ValueError),
    (httpx.Response(503), httpx.HTTPStatusError),
])
async def test_opa_reports_missing_results_and_http_failures(gateway, packet, response, error):
    gateway.response = response
    with pytest.raises(error):
        await OpaPolicy(gateway.client).evaluate(ALLOW_QUERY, packet)


@pytest.mark.parametrize("content, expected", [
    ("<score>yes</score>", True),
    (" <think>\n</think><score> no </score>\n", False),
])
async def test_assessor_maps_complete_granite_scores(gateway, packet, content, expected):
    gateway.response = httpx.Response(200, json={"choices": [{
        "finish_reason": "stop", "message": {"content": content},
    }]})

    assessor = BillingAssessor(gateway.client, model="granite-model")

    assert await assessor.assess(packet, "Trusted criterion") is expected
    assert not gateway.client.is_closed


async def test_assessor_sends_source_invoice_and_proposed_message(gateway, packet):
    gateway.response = httpx.Response(200, json={"choices": [{
        "finish_reason": "stop", "message": {"content": "<score>yes</score>"},
    }]})
    await BillingAssessor(gateway.client, model="granite-model").assess(
        packet, "Trusted criterion",
    )

    request = gateway.requests[0]
    assert request.url.path == "/v1/chat/completions"
    payload = json.loads(request.content)
    assert payload["model"] == "granite-model"
    messages = payload["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    context = json.loads(messages[0]["content"])
    assert json.loads(messages[1]["content"]) == packet["tool"]
    assert context["source_user_message"] == packet["source_user_message"]
    assert context["invoice"] == packet["invoice"]
    assert "Trusted criterion" in json.dumps(messages)


@pytest.mark.parametrize("unavailable", [False, True])
async def test_default_assessor_closes_its_client(packet, monkeypatch, unavailable):
    monkeypatch.setenv("GUARDIAN_BASE_URL", "http://granite.test/v1/")
    monkeypatch.setenv("GUARDIAN_MODEL", "granite-model")
    monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)
    clients = []
    client_class = httpx.AsyncClient

    async def transport(request):
        assert str(request.url) == "http://granite.test/v1/chat/completions"
        assert json.loads(request.content)["model"] == "granite-model"
        if unavailable:
            raise httpx.ReadTimeout("Granite unavailable")
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": "<score>yes</score>"},
        }]})

    def make_client(**kwargs):
        client = client_class(**kwargs, transport=httpx.MockTransport(transport))
        clients.append(client)
        return client

    monkeypatch.setattr("billing_policy.httpx.AsyncClient", make_client)
    assessor = BillingAssessor()
    if unavailable:
        with pytest.raises(httpx.ReadTimeout):
            await assessor.assess(packet, "Trusted criterion")
    else:
        assert await assessor.assess(packet, "Trusted criterion") is True

    assert len(clients) == 1
    assert clients[0].is_closed


@pytest.mark.parametrize("content", [
    "yes", "<score>Yes</score>",
    "<score>yes</score> explanation",
    "<think>reasoning</think><score>yes</score>",
    "<score>yes</score><score>no</score>",
])
async def test_assessor_rejects_ambiguous_or_malformed_scores(gateway, packet, content):
    gateway.response = httpx.Response(200, json={"choices": [{
        "finish_reason": "stop", "message": {"content": content},
    }]})
    with pytest.raises((ValueError, TypeError)):
        await BillingAssessor(gateway.client, model="granite-model").assess(packet, "criterion")


@pytest.mark.parametrize("message, finish_reason", [
    ({"content": "<score>yes</score>"}, "length"),
    ({"content": None}, "stop"),
    ({"content": "<score>yes</score>", "tool_calls": [{}]}, "stop"),
    ({"content": "<score>yes</score>", "refusal": "no"}, "stop"),
])
async def test_assessor_requires_a_complete_final_response(gateway, packet, message, finish_reason):
    gateway.response = httpx.Response(200, json={"choices": [{
        "finish_reason": finish_reason, "message": message,
    }]})
    with pytest.raises((ValueError, TypeError)):
        await BillingAssessor(gateway.client, model="model").assess(packet, "criterion")


@pytest.mark.parametrize("adapter", ["opa", "assessor"])
async def test_http_failure_reaches_the_caller(gateway, packet, adapter):
    gateway.error = httpx.ReadTimeout("service unavailable")
    with pytest.raises(httpx.ReadTimeout):
        if adapter == "opa":
            await OpaPolicy(gateway.client).evaluate(ALLOW_QUERY, packet)
        else:
            await BillingAssessor(gateway.client, model="model").assess(packet, "criterion")
    assert not gateway.client.is_closed
