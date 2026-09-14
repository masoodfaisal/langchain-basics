"""Check an invoice explanation before a tool saves it to a local email inbox."""

import asyncio
from email.message import EmailMessage
import os
from pathlib import Path
from uuid import uuid4

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool

from billing_policy import ALLOW_QUERY, BILLING_POLICY_PATH, CRITERION_QUERY, BillingAssessor
from db import aconnect
from rego_policy import RegoPolicy


BILLING_INBOX = Path(os.getenv("BILLING_INBOX", "billing-inbox"))
billing_policy = RegoPolicy(BILLING_POLICY_PATH)
billing_assessor = BillingAssessor()


async def read_invoice(invoice_id: int) -> dict:
    """Read an invoice and its track purchases from the shared Chinook database."""
    async with aconnect() as connection:
        cursor = await connection.execute("""
            SELECT i.InvoiceId, i.InvoiceDate, i.Total, c.Email
            FROM Invoice i JOIN Customer c ON c.CustomerId = i.CustomerId
            WHERE i.InvoiceId = ?
        """, (invoice_id,))
        invoice = await cursor.fetchone()
        if invoice is None:
            raise ValueError(f"Invoice {invoice_id} was not found.")

        cursor = await connection.execute("""
            SELECT t.Name, il.UnitPrice, il.Quantity
            FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId
            WHERE il.InvoiceId = ?
            ORDER BY il.InvoiceLineId
        """, (invoice_id,))
        lines = await cursor.fetchall()

    return {
        "invoice_id": invoice["InvoiceId"],
        "invoice_date": invoice["InvoiceDate"],
        "total": f"{invoice['Total']:.2f}",
        "email": invoice["Email"],
        "lines": [
            {
                "track": line["Name"],
                "unit_price": f"{line['UnitPrice']:.2f}",
                "quantity": line["Quantity"],
            }
            for line in lines
        ],
    }


@tool
async def get_invoice_for_explanation(invoice_id: int) -> dict:
    """Read an invoice's date, tracks, quantities, prices, and total."""
    invoice = await read_invoice(invoice_id)
    invoice.pop("email")
    return invoice


def _save_invoice_explanation(invoice_id: int, recipient: str, body: str) -> Path:
    """Build and save the email in a worker thread, including all file I/O."""
    message = EmailMessage()
    message["From"] = "billing@chinook.example"
    message["To"] = recipient
    message["Subject"] = f"Your Chinook invoice {invoice_id}"
    message.set_content(body)

    BILLING_INBOX.mkdir(parents=True, exist_ok=True)
    path = BILLING_INBOX / f"{uuid4()}.eml"
    with path.open("xb") as output:
        output.write(message.as_bytes())
    return path


@tool
async def send_invoice_explanation(invoice_id: int, body: str) -> str:
    """Send an invoice explanation to the local demo inbox as an .eml file."""
    invoice = await read_invoice(invoice_id)
    path = await asyncio.to_thread(
        _save_invoice_explanation, invoice_id, invoice["email"], body,
    )
    return f"Invoice explanation saved to the local inbox as {path.name}."


@wrap_tool_call
async def billing_guardian(request, handler):
    """Check a proposed invoice explanation before the send tool runs."""
    if request.tool_call["name"] != "send_invoice_explanation":
        return await handler(request)
    if os.getenv("ENABLE_GUARDIAN", "false").strip().lower() != "true":
        return await handler(request)

    reason = "the message does not follow the invoice policy."
    try:
        invoice = await read_invoice(request.tool_call["args"]["invoice_id"])
        invoice.pop("email")
        packet = {
            "source_user_message": request.runtime.state.get("source_user_message", ""),
            "invoice": invoice,
            "tool": {
                "name": request.tool_call["name"],
                "args": request.tool_call["args"],
            },
        }

        # Granite checks the message's meaning; Rego uses its answer.
        criterion = await billing_policy.evaluate(CRITERION_QUERY)
        intent_match = await billing_assessor.assess(packet, criterion)
        packet["guardian"] = {"intent_match": intent_match}
        allowed = await billing_policy.evaluate(ALLOW_QUERY, packet)
    except Exception:
        allowed = False
        reason = "the message check could not be completed."

    if allowed is True:
        return await handler(request)
    return ToolMessage(
        content=f"Invoice explanation blocked: {reason}",
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error",
    )
