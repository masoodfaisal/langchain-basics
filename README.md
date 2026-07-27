# Chinook Music Store Support Agent

An asynchronous LangChain/LangGraph customer-support agent built on the
[Chinook sample database](https://github.com/lerocha/chinook-database). It can
recommend music, answer questions about a signed-in customer's invoices, and
retain customer preferences across conversation threads.

## Components

| Component | File | Responsibility |
| --- | --- | --- |
| Agent graph | `agent.py` | Configures the chat model, system prompt, tools, request context, and middleware, then exports the `graph` used by LangGraph. |
| Invocation context | `context.py` | Defines `UserContext`, which carries the authenticated `customer_id` into each request. |
| Agent tools | `tools.py` | Provides seven async tools for music discovery, order support, long-term memory, and delegated music reasoning. |
| Authorization middleware | `middleware.py` | Blocks anonymous account access and prevents customers from reading invoices belonging to another customer. |
| Database layer | `db.py` | Opens async SQLite connections and resolves the configurable Chinook database path. |
| Long-term memory | `memory.py` | Wraps the LangGraph store and isolates saved memories in per-customer namespaces. |
| Embeddings | `embeddings.py` | Generates local FastEmbed vectors used for semantic memory search. |
| Runtime configuration | `langgraph.json` | Registers the graph and configures the runtime-managed memory store and its vector index. |
| Database bootstrap | `scripts/bootstrap_chinook.py` | Downloads the Chinook SQL dataset and creates the local SQLite database. |
| Evaluations | `evals/` | Defines the evaluation dataset plus offline experiment and online guardrail evaluators for LangSmith. |
| Tests | `tests/` | Covers tools, authorization, memory isolation, checkpointing, evaluators, and optional server-backed memory flows. |

The seven tools are grouped by capability:

- Music discovery: `find_similar_albums`, `popular_in_genre`
- Account support: `list_my_orders`, `get_invoice_details`
- Long-term memory: `remember`, `recall`
- LLM delegation: `ask_music_expert`

## Architecture

```text
User request + UserContext
           |
           v
   LangGraph agent (agent.py)
           |
           v
 Customer-scoping middleware
           |
     +-----+-------------------+
     |                         |
     v                         v
Catalog/account tools      Memory tools
     |                         |
     v                         v
Chinook SQLite DB       LangGraph managed store
                           |
                           v
                    FastEmbed vector index
```

Catalog tools are available to anonymous users. Account and memory tools require
an authenticated customer. Invoice ownership is checked in middleware before
the underlying tool executes, and the tool repeats the ownership check as a
defense-in-depth measure. Long-term memories are namespaced by customer so they
can persist across threads without being shared between customers.

## Requirements

- Python 3.13
- [Socket Firewall Free (`sfw`)](https://github.com/SocketDev/sfw-free)
- An OpenAI-compatible chat-completions endpoint
- `OPENAI_API_KEY` available in the shell
- A LangSmith account and key only if running the evaluation workflows

## Setup

1. Install Python 3.13 and ensure `python3.13` is available on `PATH`.

2. Create the virtual environment:

   ```bash
   python3.13 -m venv .venv
   ```

3. Install the Python dependencies:

   ```bash
   sfw .venv/bin/python -m pip install --requirement requirements.txt
   ```

4. Activate the environment:

   ```bash
   source .venv/bin/activate
   ```

Provision the Chinook SQLite database for a fresh checkout:

```bash
python scripts/bootstrap_chinook.py
```

Configure the model endpoint in `.env` or export the values from your shell.
Do not commit `.env` or secret values.

```dotenv
OPENAI_API_KEY=<your-api-key>
OPENAI_BASE_URL=<openai-compatible-base-url>
MODEL_NAME=<chat-model-name>
```

Optional configuration:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_VERIFY_SSL` | `true` | Set to `false` only for a trusted development gateway using a self-signed certificate. |
| `TOOL_LLM_MODEL` | value of `MODEL_NAME` | Selects the secondary model called by `ask_music_expert`. |
| `TOOL_LLM_API_KEY` | value of `OPENAI_API_KEY` | Overrides the API key for the secondary model. |
| `TOOL_LLM_BASE_URL` | value of `OPENAI_BASE_URL` | Overrides the gateway for the secondary model. |
| `TOOL_LLM_VERIFY_SSL` | value of `OPENAI_VERIFY_SSL` | Controls TLS verification for the secondary model gateway. |
| `CHINOOK_DB_PATH` | `chinook.db` | Overrides the SQLite database location. |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Selects the local FastEmbed model. Its output must match the 768 dimensions configured in `langgraph.json`. |
| `EMBEDDING_CACHE_DIR` | `.cache/fastembed` | Changes the local embedding-model cache directory. |
| `LANGSMITH_API_KEY` | none | Enables LangSmith evaluation workflows. |
| `LANGSMITH_PROJECT` | `agent` | Selects the LangSmith tracing project. |
| `EVAL_JUDGE_MODEL` | `gpt-4o-mini` | Selects the judge model used by the offline runner. |

The embedding model may be downloaded from Hugging Face on first startup.

## Run

Start the LangGraph development server:

```bash
./run-langgraph-dev.sh
```

The script uses the project virtual environment when available and forwards any
additional arguments to `langgraph dev`. Invoke the graph asynchronously and
provide a `UserContext` for authenticated account or memory operations:

```python
from agent import graph
from context import UserContext

result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "What did I buy recently?"}]},
    context=UserContext(customer_id=1),
)
```

Use `customer_id=None` for an anonymous caller. Because the database, tools, and
middleware are async, use `ainvoke` or `astream` rather than synchronous graph
methods.

## Test

Run the offline unit and integration tests:

```bash
pytest
```

Tests marked `e2e` require a running `langgraph dev` server and are skipped when
one is not reachable:

```bash
pytest -m e2e
```

## Evaluations

The canonical examples in `evals/dataset.py` cover music discovery, account
lookups, authentication failures, and cross-customer data isolation.

Run a dataset-backed offline experiment:

```bash
PYTHONPATH=. .venv/bin/python evals/run-eval-offline.py --replace
```

Upload and attach the online production guardrails:

```bash
PYTHONPATH=. .venv/bin/python evals/run-eval-online.py --project agent --replace
```

Both workflows require `LANGSMITH_API_KEY`. The offline workflow scores tool
trajectory, required output, data leakage, PII, correctness, and conciseness.
The online evaluator checks live traces for valid responses, appropriate account
tool use, internal-detail exposure, PII, and known invoice-data leakage.


```bash
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="XXX"
LANGSMITH_WORKSPACE_ID=XXXXX
OPENAI_BASE_URL=XXX
EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
EMBEDDING_CACHE_DIR=.cache/fastembed
MODEL_NAME=gpt-5.4-mini
LANGSMITH_DEPLOYMENT_NAME='XXX'
OPENAI_BASE_URL=XXXX
# Set LANGSMITH_API_KEY in your shell, not here.
# Set OPENAI_API_KEY in your shell, not here.
```
