# Local Keycloak SSO demo

This demo runs Keycloak in a loopback-only Docker container, configures a
realm and public PKCE client, starts the local LangGraph Agent Server with
custom authentication, and serves a small browser client.

It uses fixed local endpoints so the Keycloak client, Agent Server, and static
browser callback cannot drift apart:

- Keycloak: `http://127.0.0.1:8080`
- Agent Server: `http://127.0.0.1:2024`
- Browser demo: `http://127.0.0.1:3000/index.html`

## 1. Check prerequisites

You need a running Docker daemon, `curl`, `jq`, and Python 3. The project also
needs its dependencies installed in `.venv` (including `langgraph` and
`PyJWT[crypto]`); the launcher falls back to executables on `PATH` if there is
no project virtual environment.

## 2. Supply a local-only password

Set the one required value without printing or committing it:

```bash
printf 'Keycloak admin and demo-user password: ' >&2
IFS= read -r -s KC_BOOTSTRAP_ADMIN_PASSWORD
printf '\n' >&2
export KC_BOOTSTRAP_ADMIN_PASSWORD
```

The existing repository `.env` remains responsible for model-provider and
LangSmith configuration. The Agent Server loads it through `langgraph.json`;
the launchers do not inspect or print it.

Use a throwaway local password. Docker retains the bootstrap admin environment
value in the local container metadata, and rerunning against the same retained
container requires the same value. The `demo-user` password is reset to that
value on each run.

## 3. Start the complete demo

From the repository root:

```bash
./keycloak-sso/run-keycloak-and-agent-server.sh
```

The command is idempotent. It creates or updates:

- Keycloak container `langchain-keycloak`, pinned to Keycloak 26.7.2.
- Realm `langchain-agent`.
- Public browser client `langsmith-agent-ui` using Authorization Code + PKCE.
- Resource-server audience `langsmith-agent-api`.
- Realm role `agent-user`.
- Demo user `demo-user`, assigned to Chinook customer `1`.
- Audience and `customer_id` access-token mappers.

Keycloak remains running when the Agent Server stops, so its realm persists in
the retained container.

## 4. Demonstrate SSO

Open <http://127.0.0.1:3000/index.html>, select **Sign in with Keycloak**, and
log in as `demo-user` using `KC_BOOTSTRAP_ADMIN_PASSWORD`. Then select
**Call authenticated agent**.

The browser performs Authorization Code with PKCE and sends the access token
to the Agent Server. `auth.py` verifies the signature, issuer, audience,
authorized party, expiry, token type, and `agent-user` role before creating a
user-owned thread. Unhandled resources are denied by default, Store namespaces
are prefixed with the authenticated Keycloak subject, and the alternative
Studio authentication path is disabled for this SSO-only demo.

An unauthenticated request should fail:

```bash
curl --include --request POST \
  --header 'Content-Type: application/json' \
  --data '{}' \
  http://127.0.0.1:2024/threads
```

## Run the components separately

```bash
./keycloak-sso/start-keycloak.sh
./keycloak-sso/start-agent-server.sh
```

Press Ctrl-C to stop the Agent Server and browser demo. Stop Keycloak without
deleting its data:

```bash
docker stop langchain-keycloak
```

This setup uses HTTP and Keycloak's development mode only on `127.0.0.1`.
Use HTTPS, an external database, managed secrets, and a reachable stable
issuer before deploying outside local development.

## Custom Auth versus Agent Auth

This example implements **Custom Auth**: a browser user authenticates with
Keycloak, then the Agent Server validates that user's bearer token. LangSmith
**Agent Auth** is a separate beta feature for an already-running agent to get
OAuth tokens for downstream services. It is not needed for this inbound SSO
flow. Also, LangSmith Cloud cannot reach a Keycloak issuer bound to
`127.0.0.1`; use a self-hosted/reachable IdP before applying this setup there.
