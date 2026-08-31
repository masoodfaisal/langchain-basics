# RemoteGraph sample

This graph is a thin wrapper around a LangSmith deployment containing a graph
named `agent`.

Create the local environment file:

```bash
cp remote-graph-sample/.env.example remote-graph-sample/.env
```

Set `REMOTE_GRAPH_URL` in that file to the API URL copied from the deployment's
LangSmith page, and set `LANGSMITH_API_KEY` to a valid LangSmith API key. The
deployment URL is the Agent Server API endpoint, not its `smith.langchain.com`
dashboard URL.

Start the wrapper graph from the repository root:

```bash
.venv/bin/langgraph dev \
  --config remote-graph-sample/langgraph.json \
  --port 2024 \
  --no-browser
```

The wrapper is exposed as `remote_agent` and forwards its message state to the
remote graph named `agent`. If the wrapper is also deployed, configure both
variables as runtime environment secrets and deploy it separately from the
remote `agent`; a `RemoteGraph` must not call its own deployment.
