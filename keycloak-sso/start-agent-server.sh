#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"

# Match the fixed topology provisioned by start-keycloak.sh and used by the
# static browser demo. auth.py itself remains configurable for real deployments.
KEYCLOAK_ISSUER="http://127.0.0.1:8080/realms/langchain-agent"
KEYCLOAK_AUDIENCE="langsmith-agent-api"
KEYCLOAK_JWKS_URL="${KEYCLOAK_ISSUER}/protocol/openid-connect/certs"
KEYCLOAK_ALLOWED_CLIENTS="langsmith-agent-ui"
KEYCLOAK_ALLOWED_ROLES="agent-user"
KEYCLOAK_REQUIRED_ROLE="agent-user"

export KEYCLOAK_ISSUER
export KEYCLOAK_AUDIENCE
export KEYCLOAK_JWKS_URL
export KEYCLOAK_ALLOWED_CLIENTS
export KEYCLOAK_ALLOWED_ROLES
export KEYCLOAK_REQUIRED_ROLE

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || die "curl was not found"

if [[ -x "$project_dir/.venv/bin/langgraph" ]]; then
  langgraph_bin="$project_dir/.venv/bin/langgraph"
else
  command -v langgraph >/dev/null 2>&1 \
    || die "langgraph was not found in .venv or PATH"
  langgraph_bin="$(command -v langgraph)"
fi

if [[ -x "$project_dir/.venv/bin/python" ]]; then
  python_bin="$project_dir/.venv/bin/python"
else
  command -v python3 >/dev/null 2>&1 || die "python3 was not found"
  python_bin="$(command -v python3)"
fi

[[ -f "$project_dir/langgraph.json" ]] \
  || die "langgraph.json was not found at the repository root"
[[ -f "$project_dir/auth.py" ]] \
  || die "auth.py was not found at the repository root"
[[ -f "$script_dir/demo/index.html" ]] \
  || die "the browser demo is missing"

"$python_bin" -c 'import jwt' \
  || die "PyJWT with crypto support is not installed in the selected Python environment"

if ! curl --fail --silent --show-error \
  "$KEYCLOAK_ISSUER/.well-known/openid-configuration" \
  >/dev/null; then
  die "Keycloak is not reachable; run keycloak-sso/start-keycloak.sh first"
fi

demo_pid=""
agent_pid=""

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$agent_pid" ]] && kill -0 "$agent_pid" >/dev/null 2>&1; then
    kill "$agent_pid" >/dev/null 2>&1 || true
    wait "$agent_pid" 2>/dev/null || true
  fi
  if [[ -n "$demo_pid" ]] && kill -0 "$demo_pid" >/dev/null 2>&1; then
    kill "$demo_pid" >/dev/null 2>&1 || true
    wait "$demo_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Discard http.server access logs because the OAuth callback URL contains a
# short-lived authorization code that must not be written to terminal logs.
"$python_bin" \
  -m http.server 3000 \
  --bind 127.0.0.1 \
  --directory "$script_dir/demo" \
  >/dev/null 2>&1 &
demo_pid="$!"
sleep 0.25
kill -0 "$demo_pid" >/dev/null 2>&1 \
  || die "could not start the browser demo on 127.0.0.1:3000"

printf 'Starting the authenticated Agent Server...\n'
printf '  Browser SSO demo: http://127.0.0.1:3000/index.html\n'
printf '  Agent API:       http://127.0.0.1:2024\n'
printf '  API docs:        http://127.0.0.1:2024/docs\n'
printf 'Press Ctrl-C to stop the Agent Server and browser demo.\n\n'

cd "$project_dir"
"$langgraph_bin" dev \
  --config "$project_dir/langgraph.json" \
  --host 127.0.0.1 \
  --port 2024 \
  --no-browser \
  "$@" &
agent_pid="$!"

wait "$agent_pid"
