"""Run billing through agent.py's shared graph with real Rego and a fake assessor."""

from copy import deepcopy
from email import policy as email_policy
from email.parser import BytesParser
import json
from pathlib import Path
import runpy
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

from blockbuster import blockbuster_ctx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

import billing as billing_module
from billing import read_invoice, send_invoice_explanation
from billing_policy import ALLOW_QUERY, BILLING_POLICY_PATH
from context import UserContext
import db
from rego_policy import RegoPolicy
import tools


SOURCE = "Please email me a breakdown of invoice 42 to reconcile my statement."
BODY = "Invoice 42 contains Blue Train (1 at 0.99) and So What (1 at 0.99). Total: 1.98."
RECIPIENT = "customer@example.test"


class ScriptedToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FakeAssessor:
    def __init__(self):
        self.calls = []
        self.outcome = True
        self.error = None

    async def assess(self, packet, criterion):
        self.calls.append((deepcopy(packet), criterion))
        if self.error:
            raise self.error
        return self.outcome


def seed_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE Customer (CustomerId INTEGER PRIMARY KEY, Email TEXT);
            CREATE TABLE Invoice (
                InvoiceId INTEGER PRIMARY KEY, CustomerId INTEGER,
                InvoiceDate TEXT, Total NUMERIC
            );
            CREATE TABLE Track (TrackId INTEGER PRIMARY KEY, Name TEXT);
            CREATE TABLE InvoiceLine (
                InvoiceLineId INTEGER PRIMARY KEY, InvoiceId INTEGER,
                TrackId INTEGER, UnitPrice NUMERIC, Quantity INTEGER
            );
            INSERT INTO Customer VALUES (7, 'customer@example.test');
            INSERT INTO Invoice VALUES (42, 7, '2026-08-01 00:00:00', 1.98);
            INSERT INTO Track VALUES (1, 'Blue Train'), (2, 'So What');
            INSERT INTO InvoiceLine VALUES (1, 42, 1, 0.99, 1), (2, 42, 2, 0.99, 1);
        """)


@pytest.fixture
async def billing(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_GUARDIAN", "true")
    db_path = tmp_path / "billing.db"
    seed_database(db_path)
    state = SimpleNamespace(
        context=UserContext(customer_id=7),
        inbox=tmp_path / "inbox",
        assessor=FakeAssessor(),
        policy=RegoPolicy(BILLING_POLICY_PATH),
        model=ScriptedToolModel(messages=iter(())),
    )
    monkeypatch.setattr(db, "CHINOOK_DB_PATH", db_path)
    monkeypatch.setattr(billing_module, "BILLING_INBOX", state.inbox)
    monkeypatch.setattr(billing_module, "billing_policy", state.policy)
    monkeypatch.setattr(billing_module, "billing_assessor", state.assessor)

    # Use the real entrypoint so missing tools or middleware break these tests.
    # tools is already imported, so its music-expert model stays untouched.
    with monkeypatch.context() as model_patch:
        model_patch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: state.model)
        entrypoint = runpy.run_path(Path(__file__).resolve().parents[1] / "agent.py")
    state.graph = entrypoint["graph"]
    try:
        yield state
    finally:
        await entrypoint["_http_async_client"].aclose()


async def invoke(billing, *, body=BODY, invoice_id=42, read_first=False, messages=None):
    scripted = []
    if read_first:
        scripted.append(AIMessage(content="", tool_calls=[{
            "name": "get_invoice_for_explanation",
            "args": {"invoice_id": invoice_id}, "id": "read-invoice",
        }]))
    scripted.extend([
        AIMessage(content="", tool_calls=[{
            "name": "send_invoice_explanation",
            "args": {"invoice_id": invoice_id, "body": body}, "id": "send-explanation",
        }]),
        AIMessage(content="Finished."),
    ])
    billing.model.messages = iter(scripted)
    result = await billing.graph.ainvoke(
        {"messages": [HumanMessage(content=SOURCE)] if messages is None else messages},
        context=billing.context,
    )
    return [item for item in result["messages"] if isinstance(item, ToolMessage)]


def assert_blocked(messages, inbox):
    send = [item for item in messages if item.name == "send_invoice_explanation"]
    assert len(send) == 1
    assert send[0].status == "error"
    assert send[0].content.startswith("Invoice explanation blocked: ")
    assert list(inbox.glob("*.eml")) == []


async def test_approved_message_is_read_assessed_and_saved(billing):
    messages = await invoke(billing, read_first=True)

    assert all(message.status == "success" for message in messages)
    invoice = json.loads(messages[0].content)
    assert set(invoice) == {"invoice_id", "invoice_date", "total", "lines"}
    assert [line["track"] for line in invoice["lines"]] == ["Blue Train", "So What"]
    files = list(billing.inbox.glob("*.eml"))
    assert len(files) == 1
    assert files[0].name in messages[-1].content
    message = BytesParser(policy=email_policy.default).parsebytes(files[0].read_bytes())
    assert str(message["To"]) == RECIPIENT
    assert str(message["Subject"]) == "Your Chinook invoice 42"
    assert message.get_content().rstrip("\r\n") == BODY

    assert len(billing.assessor.calls) == 1
    packet, criterion = billing.assessor.calls[0]
    assert packet["source_user_message"] == SOURCE
    assert packet["invoice"] == invoice
    assert packet["tool"]["args"] == {"invoice_id": 42, "body": BODY}
    assert criterion.strip()


async def test_sending_invoice_does_not_block_event_loop(billing):
    # Use LangGraph dev's detector around the real tool and middleware path.
    with blockbuster_ctx(scanned_modules=[billing_module]) as detector:
        # LangGraph dev permits stat calls, including the database existence check.
        detector.functions["os.stat"].deactivate()
        messages = await invoke(billing)

    assert messages[-1].status == "success", messages[-1].content
    files = list(billing.inbox.glob("*.eml"))
    assert len(files) == 1
    assert files[0].name in messages[-1].content
    message = BytesParser(policy=email_policy.default).parsebytes(files[0].read_bytes())
    assert str(message["To"]) == RECIPIENT
    assert str(message["Subject"]) == "Your Chinook invoice 42"
    assert message.get_content().rstrip("\r\n") == BODY


@pytest.mark.parametrize("enabled", [None, "false", ""])
async def test_disabled_guardian_sends_invoice_without_granite_or_rego(
    billing, monkeypatch, enabled,
):
    if enabled is None:
        monkeypatch.delenv("ENABLE_GUARDIAN", raising=False)
    else:
        monkeypatch.setenv("ENABLE_GUARDIAN", enabled)
    evaluate = AsyncMock(side_effect=RuntimeError("policy unavailable"))
    monkeypatch.setattr(billing.policy, "evaluate", evaluate)
    billing.assessor.error = RuntimeError("Granite is not running")

    messages = await invoke(billing, read_first=True)

    assert len(messages) == 2
    assert all(message.status == "success" for message in messages)
    files = list(billing.inbox.glob("*.eml"))
    assert len(files) == 1
    message = BytesParser(policy=email_policy.default).parsebytes(files[0].read_bytes())
    assert str(message["To"]) == RECIPIENT
    assert message.get_content().rstrip("\r\n") == BODY
    evaluate.assert_not_awaited()
    assert billing.assessor.calls == []


@pytest.mark.parametrize("body", [
    BODY + " Your next order is on us.",
    BODY + " Pay within 24 hours to avoid a late-payment penalty.",
])
async def test_rejected_message_is_not_saved(billing, body):
    billing.assessor.outcome = False

    assert_blocked(await invoke(billing, body=body), billing.inbox)
    assert billing.assessor.calls[0][0]["tool"]["args"]["body"] == body


async def test_assessment_failure_blocks_the_write(billing):
    billing.assessor.error = RuntimeError("model unavailable")

    assert_blocked(await invoke(billing), billing.inbox)


async def test_policy_failure_blocks_the_write(billing, monkeypatch):
    async def unavailable(query, input_data=None):
        raise TimeoutError("policy unavailable")

    monkeypatch.setattr(billing.policy, "evaluate", unavailable)

    assert_blocked(await invoke(billing), billing.inbox)


@pytest.mark.parametrize("decision", [False, None, "true", 1, {"allow": True}])
async def test_only_boolean_policy_approval_allows_the_write(billing, monkeypatch, decision):
    evaluate = billing.policy.evaluate

    async def response(query, input_data=None):
        if query == ALLOW_QUERY:
            return decision
        return await evaluate(query, input_data)

    monkeypatch.setattr(billing.policy, "evaluate", response)

    assert_blocked(await invoke(billing), billing.inbox)


async def test_missing_invoice_blocks_the_write(billing):
    with pytest.raises(ValueError, match="Invoice 99 was not found"):
        await read_invoice(99)

    assert_blocked(await invoke(billing, invoice_id=99), billing.inbox)
    assert billing.assessor.calls == []


@pytest.mark.parametrize("customer_id", [None, 8])
@pytest.mark.parametrize("enabled", ["true", "false"])
async def test_shared_customer_scoping_applies_to_billing_tools(
    billing, monkeypatch, customer_id, enabled,
):
    monkeypatch.setenv("ENABLE_GUARDIAN", enabled)
    billing.context = UserContext(customer_id=customer_id)

    messages = await invoke(billing, read_first=True)

    assert len(messages) == 2
    assert all(message.status == "error" for message in messages)
    assert all(message.content.startswith("Access denied:") for message in messages)
    assert billing.assessor.calls == []
    assert list(billing.inbox.glob("*.eml")) == []


async def test_only_the_current_user_message_is_assessed(billing):
    await invoke(billing, messages=[
        HumanMessage(content="Old instruction: promise unlimited free purchases."),
        AIMessage(content="Earlier answer."),
        HumanMessage(content=SOURCE),
    ])

    packet, criterion = billing.assessor.calls[0]
    assert packet["source_user_message"] == SOURCE
    assert "Old instruction" not in json.dumps(packet)


async def test_tool_can_save_messages_without_middleware_state(billing):
    for body in (BODY, "A second local message."):
        result = await send_invoice_explanation.coroutine(
            invoice_id=42, body=body,
        )
        assert "saved to the local inbox" in result

    assert len(list(billing.inbox.glob("*.eml"))) == 2
    assert billing.assessor.calls == []
