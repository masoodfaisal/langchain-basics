#!/usr/bin/env bash

set -euo pipefail

KEYCLOAK_IMAGE="${KEYCLOAK_IMAGE:-quay.io/keycloak/keycloak:26.7.2}"
KEYCLOAK_CONTAINER_NAME="${KEYCLOAK_CONTAINER_NAME:-langchain-keycloak}"
# Keep the public topology fixed: the static browser client and Agent Server
# launcher use these exact local-only values.
readonly KEYCLOAK_PORT="8080"
readonly KEYCLOAK_REALM="langchain-agent"
readonly KEYCLOAK_BROWSER_CLIENT_ID="langsmith-agent-ui"
readonly KEYCLOAK_API_AUDIENCE="langsmith-agent-api"
readonly KEYCLOAK_REQUIRED_ROLE="agent-user"
KEYCLOAK_DEMO_USERNAME="${KEYCLOAK_DEMO_USERNAME:-demo-user}"
KEYCLOAK_DEMO_EMAIL="${KEYCLOAK_DEMO_EMAIL:-demo-user@example.invalid}"
KEYCLOAK_DEMO_CUSTOMER_ID="${KEYCLOAK_DEMO_CUSTOMER_ID:-1}"

KEYCLOAK_BASE_URL="http://127.0.0.1:${KEYCLOAK_PORT}"
KEYCLOAK_ISSUER="${KEYCLOAK_BASE_URL}/realms/${KEYCLOAK_REALM}"
DEMO_ORIGIN="http://127.0.0.1:3000"
DEMO_REDIRECT_URI="${DEMO_ORIGIN}/index.html"

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command '$1' was not found"
}

require_secret() {
  local variable_name="$1"
  if [[ -z "${!variable_name:-}" ]]; then
    die "$variable_name is not set; export it without printing its value"
  fi
}

require_command curl
require_command docker
require_command jq
require_command python3
require_secret KC_BOOTSTRAP_ADMIN_PASSWORD

KC_BOOTSTRAP_ADMIN_USERNAME="${KC_BOOTSTRAP_ADMIN_USERNAME:-admin}"
export KC_BOOTSTRAP_ADMIN_USERNAME
export KC_BOOTSTRAP_ADMIN_PASSWORD

[[ "$KEYCLOAK_PORT" =~ ^[0-9]+$ ]] || die "KEYCLOAK_PORT must be numeric"
(( KEYCLOAK_PORT >= 1024 && KEYCLOAK_PORT <= 65535 )) \
  || die "KEYCLOAK_PORT must be between 1024 and 65535"
[[ "$KEYCLOAK_DEMO_CUSTOMER_ID" =~ ^[1-9][0-9]*$ ]] \
  || die "KEYCLOAK_DEMO_CUSTOMER_ID must be a positive integer"
[[ "$KEYCLOAK_REALM" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "KEYCLOAK_REALM contains unsupported characters"
[[ "$KEYCLOAK_BROWSER_CLIENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "KEYCLOAK_BROWSER_CLIENT_ID contains unsupported characters"
[[ "$KEYCLOAK_API_AUDIENCE" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "KEYCLOAK_API_AUDIENCE contains unsupported characters"
[[ "$KEYCLOAK_REQUIRED_ROLE" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "KEYCLOAK_REQUIRED_ROLE contains unsupported characters"
[[ "$KEYCLOAK_DEMO_USERNAME" =~ ^[A-Za-z0-9@._-]+$ ]] \
  || die "KEYCLOAK_DEMO_USERNAME contains unsupported characters"
[[ "$KEYCLOAK_IMAGE" != *:latest ]] \
  || die "KEYCLOAK_IMAGE must use a pinned version or digest, not latest"

docker info >/dev/null 2>&1 \
  || die "Docker is not running or is not accessible to this user"

if docker container inspect "$KEYCLOAK_CONTAINER_NAME" >/dev/null 2>&1; then
  existing_image="$(
    docker container inspect \
      --format '{{.Config.Image}}' \
      "$KEYCLOAK_CONTAINER_NAME"
  )"
  if [[ "$existing_image" != "$KEYCLOAK_IMAGE" ]]; then
    die "container $KEYCLOAK_CONTAINER_NAME uses $existing_image; expected $KEYCLOAK_IMAGE"
  fi

  # `docker port` reads runtime network state, which can be empty while a
  # retained container is Created or Exited. HostConfig.PortBindings is the
  # persisted configuration Docker uses when the container starts.
  existing_port_binding="$(
    docker container inspect \
      --format '{{json .HostConfig.PortBindings}}' \
      "$KEYCLOAK_CONTAINER_NAME" \
      | jq --raw-output '
          .["8080/tcp"] as $bindings
          | if ($bindings | type) == "array" and ($bindings | length) == 1
            then "\($bindings[0].HostIp):\($bindings[0].HostPort)"
            else ""
            end
        '
  )"
  if [[ "$existing_port_binding" != "127.0.0.1:${KEYCLOAK_PORT}" ]]; then
    die "container $KEYCLOAK_CONTAINER_NAME must publish 127.0.0.1:${KEYCLOAK_PORT}:8080"
  fi

  if [[ "$(
    docker container inspect \
      --format '{{.State.Running}}' \
      "$KEYCLOAK_CONTAINER_NAME"
  )" != "true" ]]; then
    printf 'Starting existing Keycloak container %s...\n' \
      "$KEYCLOAK_CONTAINER_NAME"
    docker start "$KEYCLOAK_CONTAINER_NAME" >/dev/null
  fi
else
  printf 'Starting Keycloak %s in a loopback-only container...\n' \
    "$KEYCLOAK_IMAGE"
  docker run \
    --detach \
    --name "$KEYCLOAK_CONTAINER_NAME" \
    --publish "127.0.0.1:${KEYCLOAK_PORT}:8080" \
    --env KC_BOOTSTRAP_ADMIN_USERNAME \
    --env KC_BOOTSTRAP_ADMIN_PASSWORD \
    --env KC_HEALTH_ENABLED=true \
    "$KEYCLOAK_IMAGE" \
    start-dev \
    --hostname="$KEYCLOAK_BASE_URL" \
    >/dev/null
fi

printf 'Waiting for Keycloak discovery...\n'
keycloak_ready=false
for _attempt in $(seq 1 60); do
  if curl --fail --silent --show-error \
    "${KEYCLOAK_BASE_URL}/realms/master/.well-known/openid-configuration" \
    >/dev/null 2>&1; then
    keycloak_ready=true
    break
  fi

  if [[ "$(
    docker container inspect \
      --format '{{.State.Running}}' \
      "$KEYCLOAK_CONTAINER_NAME" 2>/dev/null || true
  )" != "true" ]]; then
    docker logs --tail 80 "$KEYCLOAK_CONTAINER_NAME" >&2 || true
    die "Keycloak stopped before becoming ready"
  fi
  sleep 2
done
[[ "$keycloak_ready" == "true" ]] \
  || die "Keycloak did not become ready within 120 seconds"

kcadm_login() {
  local login_error

  # Pass the bootstrap password to the container by environment-variable name.
  # It is never embedded in this script, printed, or placed in a host argument.
  if ! login_error="$(
    docker exec \
      --env KC_BOOTSTRAP_ADMIN_USERNAME \
      --env KC_BOOTSTRAP_ADMIN_PASSWORD \
      "$KEYCLOAK_CONTAINER_NAME" \
      /bin/sh -eu -c '
        exec /opt/keycloak/bin/kcadm.sh config credentials \
          --server http://127.0.0.1:8080 \
          --realm master \
          --user "$KC_BOOTSTRAP_ADMIN_USERNAME" \
          --password "$KC_BOOTSTRAP_ADMIN_PASSWORD"
      ' \
      2>&1 >/dev/null
  )"; then
    if [[ "$login_error" == *'HTTP 401 Unauthorized'* ]]; then
      die "Keycloak admin login failed: KC_BOOTSTRAP_ADMIN_PASSWORD must match the password used to create retained container $KEYCLOAK_CONTAINER_NAME"
    fi
    printf '%s\n' "$login_error" >&2
    return 1
  fi
}

kcadm_login

kcadm() {
  docker exec \
    "$KEYCLOAK_CONTAINER_NAME" \
    /opt/keycloak/bin/kcadm.sh \
    "$@"
}

kcadm_input() {
  docker exec \
    --interactive \
    "$KEYCLOAK_CONTAINER_NAME" \
    /opt/keycloak/bin/kcadm.sh \
    "$@"
}

client_uuid() {
  kcadm get clients \
    --target-realm "$KEYCLOAK_REALM" \
    --query "clientId=$1" \
    | jq --raw-output '.[0].id // empty'
}

if kcadm get "realms/$KEYCLOAK_REALM" >/dev/null 2>&1; then
  kcadm update "realms/$KEYCLOAK_REALM" \
    --set enabled=true \
    --set sslRequired=none \
    --set registrationAllowed=false \
    >/dev/null
else
  printf 'Creating realm %s...\n' "$KEYCLOAK_REALM"
  kcadm create realms \
    --set "realm=$KEYCLOAK_REALM" \
    --set enabled=true \
    --set sslRequired=none \
    --set registrationAllowed=false \
    --set loginWithEmailAllowed=true \
    >/dev/null

  # Refresh the access token after Keycloak adds the new realm's management
  # grants, then use --target-realm for all administration within that realm.
  kcadm_login
fi

api_client_uuid="$(client_uuid "$KEYCLOAK_API_AUDIENCE")"
if [[ -z "$api_client_uuid" ]]; then
  printf 'Creating API audience %s...\n' "$KEYCLOAK_API_AUDIENCE"
  api_client_uuid="$(
    kcadm create clients \
      --target-realm "$KEYCLOAK_REALM" \
      --set "clientId=$KEYCLOAK_API_AUDIENCE" \
      --set enabled=true \
      --set protocol=openid-connect \
      --set bearerOnly=true \
      --set publicClient=false \
      --set standardFlowEnabled=false \
      --set directAccessGrantsEnabled=false \
      --set serviceAccountsEnabled=false \
      --id
  )"
else
  kcadm update "clients/$api_client_uuid" \
    --target-realm "$KEYCLOAK_REALM" \
    --set enabled=true \
    --set bearerOnly=true \
    --set publicClient=false \
    --set standardFlowEnabled=false \
    --set directAccessGrantsEnabled=false \
    --set serviceAccountsEnabled=false \
    >/dev/null
fi

browser_client_uuid="$(client_uuid "$KEYCLOAK_BROWSER_CLIENT_ID")"
browser_redirects="[\"$DEMO_REDIRECT_URI\"]"
browser_origins="[\"$DEMO_ORIGIN\"]"
browser_attributes="{\"pkce.code.challenge.method\":\"S256\",\"post.logout.redirect.uris\":\"$DEMO_REDIRECT_URI\"}"

if [[ -z "$browser_client_uuid" ]]; then
  printf 'Creating public PKCE browser client %s...\n' \
    "$KEYCLOAK_BROWSER_CLIENT_ID"
  browser_client_uuid="$(
    kcadm create clients \
      --target-realm "$KEYCLOAK_REALM" \
      --set "clientId=$KEYCLOAK_BROWSER_CLIENT_ID" \
      --set enabled=true \
      --set protocol=openid-connect \
      --set publicClient=true \
      --set bearerOnly=false \
      --set standardFlowEnabled=true \
      --set implicitFlowEnabled=false \
      --set directAccessGrantsEnabled=false \
      --set serviceAccountsEnabled=false \
      --set fullScopeAllowed=false \
      --set "redirectUris=$browser_redirects" \
      --set "webOrigins=$browser_origins" \
      --set "attributes=$browser_attributes" \
      --id
  )"
else
  kcadm update "clients/$browser_client_uuid" \
    --target-realm "$KEYCLOAK_REALM" \
    --set enabled=true \
    --set publicClient=true \
    --set bearerOnly=false \
    --set standardFlowEnabled=true \
    --set implicitFlowEnabled=false \
    --set directAccessGrantsEnabled=false \
    --set serviceAccountsEnabled=false \
    --set fullScopeAllowed=false \
    --set "redirectUris=$browser_redirects" \
    --set "webOrigins=$browser_origins" \
    --set "attributes=$browser_attributes" \
    >/dev/null
fi

upsert_protocol_mapper() {
  local mapper_name="$1"
  local mapper_json="$2"
  local mapper_id

  mapper_id="$(
    kcadm get \
      "clients/$browser_client_uuid/protocol-mappers/models" \
      --target-realm "$KEYCLOAK_REALM" \
      | jq --raw-output \
          --arg name "$mapper_name" \
          '.[] | select(.name == $name) | .id' \
      | sed -n '1p'
  )"

  if [[ -z "$mapper_id" ]]; then
    printf '%s' "$mapper_json" \
      | kcadm_input create \
          "clients/$browser_client_uuid/protocol-mappers/models" \
          --target-realm "$KEYCLOAK_REALM" \
          --file - \
          >/dev/null
  else
    printf '%s' "$mapper_json" \
      | kcadm_input update \
          "clients/$browser_client_uuid/protocol-mappers/models/$mapper_id" \
          --target-realm "$KEYCLOAK_REALM" \
          --file - \
          >/dev/null
  fi
}

audience_mapper_name="${KEYCLOAK_API_AUDIENCE}-audience"
audience_mapper_json="$(
  jq --null-input \
    --arg name "$audience_mapper_name" \
    --arg audience "$KEYCLOAK_API_AUDIENCE" \
    '{
      name: $name,
      protocol: "openid-connect",
      protocolMapper: "oidc-audience-mapper",
      consentRequired: false,
      config: {
        "included.client.audience": $audience,
        "id.token.claim": "false",
        "access.token.claim": "true",
        "introspection.token.claim": "true"
      }
    }'
)"
upsert_protocol_mapper "$audience_mapper_name" "$audience_mapper_json"

customer_mapper_name="customer-id-claim"
customer_mapper_json="$(
  jq --null-input \
    --arg name "$customer_mapper_name" \
    '{
      name: $name,
      protocol: "openid-connect",
      protocolMapper: "oidc-usermodel-attribute-mapper",
      consentRequired: false,
      config: {
        "user.attribute": "customer_id",
        "claim.name": "customer_id",
        "jsonType.label": "int",
        "id.token.claim": "true",
        "access.token.claim": "true",
        "userinfo.token.claim": "true",
        "introspection.token.claim": "true",
        "multivalued": "false"
      }
    }'
)"
upsert_protocol_mapper "$customer_mapper_name" "$customer_mapper_json"

if ! kcadm get \
  "roles/$KEYCLOAK_REQUIRED_ROLE" \
  --target-realm "$KEYCLOAK_REALM" \
  >/dev/null 2>&1; then
  printf 'Creating realm role %s...\n' "$KEYCLOAK_REQUIRED_ROLE"
  kcadm create roles \
    --target-realm "$KEYCLOAK_REALM" \
    --set "name=$KEYCLOAK_REQUIRED_ROLE" \
    --set 'description=May invoke the local LangSmith agent' \
    >/dev/null
fi

# The browser client has fullScopeAllowed=false. Explicitly add only the
# realm role this API trusts so it is present in realm_access.roles.
if ! kcadm get \
  "clients/$browser_client_uuid/scope-mappings/realm/composite" \
  --target-realm "$KEYCLOAK_REALM" \
  | jq --exit-status \
      --arg role "$KEYCLOAK_REQUIRED_ROLE" \
      'any(.name == $role)' \
      >/dev/null; then
  kcadm get \
    "roles/$KEYCLOAK_REQUIRED_ROLE" \
    --target-realm "$KEYCLOAK_REALM" \
    | jq '[.]' \
    | kcadm_input create \
        "clients/$browser_client_uuid/scope-mappings/realm" \
        --target-realm "$KEYCLOAK_REALM" \
        --file - \
        >/dev/null
fi

demo_user_uuid="$(
  kcadm get users \
    --target-realm "$KEYCLOAK_REALM" \
    --query "username=$KEYCLOAK_DEMO_USERNAME" \
    | jq --raw-output '.[0].id // empty'
)"
demo_attributes="{\"customer_id\":[\"$KEYCLOAK_DEMO_CUSTOMER_ID\"]}"

if [[ -z "$demo_user_uuid" ]]; then
  printf 'Creating demo user %s...\n' "$KEYCLOAK_DEMO_USERNAME"
  demo_user_uuid="$(
    kcadm create users \
      --target-realm "$KEYCLOAK_REALM" \
      --set "username=$KEYCLOAK_DEMO_USERNAME" \
      --set "email=$KEYCLOAK_DEMO_EMAIL" \
      --set emailVerified=true \
      --set enabled=true \
      --set firstName=Demo \
      --set lastName=User \
      --set "attributes=$demo_attributes" \
      --id
  )"
else
  kcadm update "users/$demo_user_uuid" \
    --target-realm "$KEYCLOAK_REALM" \
    --set enabled=true \
    --set "email=$KEYCLOAK_DEMO_EMAIL" \
    --set emailVerified=true \
    --set "attributes=$demo_attributes" \
    >/dev/null
fi

# Use the same local-only password for the demo user. Build its credential JSON
# from the environment and stream it directly to kcadm without writing a file.
python3 -c '
import json
import os
import sys

json.dump(
    {
        "type": "password",
        "temporary": False,
        "value": os.environ["KC_BOOTSTRAP_ADMIN_PASSWORD"],
    },
    sys.stdout,
)
' \
  | kcadm_input update \
      "users/$demo_user_uuid/reset-password" \
      --target-realm "$KEYCLOAK_REALM" \
      --file - \
      >/dev/null

if ! kcadm get \
  "users/$demo_user_uuid/role-mappings/realm/composite" \
  --target-realm "$KEYCLOAK_REALM" \
  | jq --exit-status \
      --arg role "$KEYCLOAK_REQUIRED_ROLE" \
      'any(.name == $role)' \
      >/dev/null; then
  kcadm add-roles \
    --target-realm "$KEYCLOAK_REALM" \
    --uusername "$KEYCLOAK_DEMO_USERNAME" \
    --rolename "$KEYCLOAK_REQUIRED_ROLE" \
    >/dev/null
fi

printf '\nKeycloak SSO is ready.\n'
printf '  Admin console: %s/admin/\n' "$KEYCLOAK_BASE_URL"
printf '  Issuer:        %s\n' "$KEYCLOAK_ISSUER"
printf '  Realm:         %s\n' "$KEYCLOAK_REALM"
printf '  Browser client:%s\n' "$KEYCLOAK_BROWSER_CLIENT_ID"
printf '  API audience:  %s\n' "$KEYCLOAK_API_AUDIENCE"
printf '  Demo username: %s\n' "$KEYCLOAK_DEMO_USERNAME"
printf '  Demo password: supplied by KC_BOOTSTRAP_ADMIN_PASSWORD (not printed)\n'
