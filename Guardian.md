# Checking invoice explanations with LangChain and Rego

A customer asks:

> Please email me a breakdown of invoice 1.

The support agent in [agent.py](agent.py) reads the invoice and writes an
explanation. With `ENABLE_GUARDIAN=true`, LangChain middleware asks Granite
Guardian whether the message matches the customer's request and the invoice
facts before the send tool runs. A Rego policy uses that answer to allow or block
the tool call.

An accurate explanation is useful. Adding “Your next order is on us” invents a
commercial commitment. The tool arguments can look ordinary while the message
itself needs checking.

## Read the example

Start with [billing.py](billing.py). It contains two ordinary tools and one
middleware function:

- `get_invoice_for_explanation` reads an invoice and its line items.
- `send_invoice_explanation` saves the message as an `.eml` file in a local inbox.
- `billing_guardian`, decorated with `@wrap_tool_call`, checks a proposed send
  before calling `handler(request)`.

[agent.py](agent.py) registers these tools and middleware in its existing
`create_agent` call, alongside music discovery, account support and memory.

With `ENABLE_GUARDIAN=true`, the middleware follows a short sequence:

1. Load the invoice and the latest customer message.
2. Read the judging criterion from the Rego policy.
3. Ask the guardian to assess the proposed explanation.
4. Give the assessment to Rego.
5. Run the tool when Rego allows it; otherwise return an error `ToolMessage`.

[billing_policy.py](billing_policy.py) contains the HTTP calls and response parsing
for Granite Guardian and the optional OPA service.
[policies/billing_guardian.rego](policies/billing_guardian.rego) holds the criterion
and the allow rule. When checks are enabled,
[rego_policy.py](rego_policy.py) evaluates that policy in process with RegoPy.

The guardian judges the text; Rego applies the rule; the middleware controls
whether the tool runs. A missing or invalid model response blocks the send when
checks are enabled. `ENABLE_GUARDIAN=false` (the default) skips both the Granite
assessment and Rego evaluation, so the demo can run without local Granite.

## Use the support agent

Configure the acting model as described in [README.md](README.md), then choose
the demo mode in [Local Granite with Ollama and ENABLE_GUARDIAN](README.md#local-granite-with-ollama-and-enable_guardian).
That section includes the Ollama model setup and commands for both flag values.

With checks enabled, Granite needs its custom-criterion chat template. The
assessment expects a complete `<score>yes</score>` or `<score>no</score>` response and blocks missing,
malformed or truncated responses. RegoPy evaluates the policy locally; no OPA
server is needed.

Use invoice `1`, which belongs to customer `2` in the standard Chinook database.
Invoke the existing graph from `agent.py`:

```python
from agent import graph
from context import UserContext

result = await graph.ainvoke(
    {"messages": [{
        "role": "user",
        "content": "Please email me a breakdown of invoice 1.",
    }]},
    context=UserContext(customer_id=2),
)
```

The billing tools read the existing database configured by `CHINOOK_DB_PATH`.
The send tool looks up the recipient on the invoice and saves the message body
under `BILLING_INBOX` (default: `billing-inbox`). No new tables are created.

The existing customer middleware checks invoice ownership. The billing example
saves local messages; connecting an email delivery service is outside its scope.

See [Least Agency Demo](README.md#least-agency-demo) for the flow, prompts and
expected outcomes.

The focused tests run without model services:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_billing.py tests/test_billing_policy.py -q
```
