"""Exercise memory authorization against the actual Rego policy."""

from pathlib import Path

import pytest

from rego_policy import RegoPolicy


POLICY_ID = "customer-memory-v4"


@pytest.fixture(scope="module")
def policy():
    return RegoPolicy(Path(__file__).resolve().parents[1] / "policies" / "guardian.rego")


@pytest.fixture
def packet():
    return {
        "subject": {"authenticated": True, "customer_id": 2},
        "tool": {
            "name": "remember",
            "args": {"fact": "Favor jazz recommendations and exclude heavy metal."},
        },
        "memory": {"customer_id": 2, "namespace": ["2", "memories"], "operation": "put"},
        "source_user_message": "I prefer jazz recommendations and do not want heavy metal suggestions.",
        "guardian": {"policy_id": POLICY_ID, "intent_match": True},
    }


@pytest.fixture
def recall_packet(packet):
    packet["tool"] = {"name": "recall", "args": {"query": "music preferences", "limit": 5}}
    packet["memory"]["operation"] = "search"
    del packet["source_user_message"]
    del packet["guardian"]
    return packet


async def test_write_requires_positive_current_policy_assessment(policy, packet):
    assert await policy.evaluate("data.chinook.guardian.decision", packet) == {
        "allow": True, "policy_id": POLICY_ID,
    }


async def test_read_needs_no_granite_assessment_or_user_message(policy, recall_packet):
    assert await policy.evaluate("data.chinook.guardian.decision", recall_packet) == {
        "allow": True, "policy_id": POLICY_ID,
    }


async def test_denied_write_does_not_prevent_reading_existing_memories(policy, recall_packet):
    recall_packet["guardian"] = {"policy_id": POLICY_ID, "intent_match": False}
    assert await policy.evaluate("data.chinook.guardian.allow", recall_packet) is True


@pytest.mark.parametrize("operation", ["remember", "recall"])
@pytest.mark.parametrize("section, field, value", [
    ("subject", "authenticated", False),
    ("subject", "authenticated", "true"),
    ("subject", "customer_id", 0),
    ("subject", "customer_id", -2),
    ("subject", "customer_id", 2.5),
    ("subject", "customer_id", "2"),
    ("subject", "customer_id", True),
    ("memory", "customer_id", 3),
    ("memory", "namespace", ["3", "memories"]),
    ("memory", "namespace", ["2", "private"]),
    ("memory", "namespace", ["2", "memories", "extra"]),
    ("memory", "namespace", [2, "memories"]),
    ("memory", "operation", "delete"),
    ("tool", "name", "delete_memory"),
])
async def test_reads_and_writes_require_authorized_customer_memory(
    policy, packet, operation, section, field, value,
):
    if operation == "recall":
        packet["tool"] = {"name": "recall", "args": {"query": "music", "limit": 5}}
        packet["memory"]["operation"] = "search"
    packet[section][field] = value
    assert await policy.evaluate("data.chinook.guardian.allow", packet) is False


@pytest.mark.parametrize("section", ["subject", "tool", "memory", "source_user_message", "guardian"])
async def test_missing_write_facts_deny(policy, packet, section):
    del packet[section]
    assert await policy.evaluate("data.chinook.guardian.allow", packet) is False


@pytest.mark.parametrize("field, value", [
    ("intent_match", False), ("intent_match", "true"), ("intent_match", None),
    ("policy_id", "previous-policy"), ("policy_id", None),
])
async def test_invalid_granite_assessment_denies_write(policy, packet, field, value):
    packet["guardian"][field] = value
    assert await policy.evaluate("data.chinook.guardian.allow", packet) is False


@pytest.mark.parametrize("value", [None, 5, "", " \t\n", "x" * 1001])
async def test_invalid_fact_denies_write(policy, packet, value):
    packet["tool"]["args"]["fact"] = value
    assert await policy.evaluate("data.chinook.guardian.allow", packet) is False


@pytest.mark.parametrize("value", [None, 5, "", " \t\n", "x" * 8001])
async def test_invalid_user_message_denies_write(policy, packet, value):
    packet["source_user_message"] = value
    assert await policy.evaluate("data.chinook.guardian.allow", packet) is False


@pytest.mark.parametrize("args", [{}, {"text": "Prefers jazz"}, {"fact": "Prefers jazz", "customer_id": 2}])
async def test_write_accepts_only_fact_argument(policy, packet, args):
    packet["tool"]["args"] = args
    assert await policy.evaluate("data.chinook.guardian.allow", packet) is False


@pytest.mark.parametrize("field, value", [
    ("query", None), ("query", 5), ("query", ""), ("query", " \t\n"),
    ("query", "x" * 1001), ("limit", 0), ("limit", 51), ("limit", 2.5),
    ("limit", "5"), ("limit", True),
])
async def test_read_requires_bounded_query_and_integer_limit(policy, recall_packet, field, value):
    recall_packet["tool"]["args"][field] = value
    assert await policy.evaluate("data.chinook.guardian.allow", recall_packet) is False


@pytest.mark.parametrize("args", [
    {}, {"query": "jazz"}, {"limit": 5},
    {"query": "jazz", "limit": 5, "customer_id": 2},
])
async def test_read_accepts_only_query_and_limit(policy, recall_packet, args):
    recall_packet["tool"]["args"] = args
    assert await policy.evaluate("data.chinook.guardian.allow", recall_packet) is False


async def test_write_cannot_use_search_operation(policy, packet):
    packet["memory"]["operation"] = "search"
    assert await policy.evaluate("data.chinook.guardian.allow", packet) is False


async def test_read_cannot_use_put_operation(policy, recall_packet):
    recall_packet["memory"]["operation"] = "put"
    assert await policy.evaluate("data.chinook.guardian.allow", recall_packet) is False


async def test_write_accepts_content_at_limits(policy, packet):
    packet["tool"]["args"]["fact"] = "x" * 1000
    packet["source_user_message"] = "x" * 8000
    assert await policy.evaluate("data.chinook.guardian.allow", packet) is True


@pytest.mark.parametrize("limit", [1, 50])
async def test_read_accepts_query_and_limit_boundaries(policy, recall_packet, limit):
    recall_packet["tool"]["args"] = {"query": "x" * 1000, "limit": limit}
    assert await policy.evaluate("data.chinook.guardian.allow", recall_packet) is True


async def test_empty_input_denies(policy):
    assert await policy.evaluate("data.chinook.guardian.allow", {}) is False
