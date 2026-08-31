"""Keycloak-backed authentication for the local LangGraph Agent Server.

The browser obtains an access token from Keycloak and sends it as a bearer
token.  This module validates the token locally against Keycloak's cached
JWKS; it never stores access or refresh tokens in graph state.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, cast
from urllib.parse import urlparse

import jwt
from langgraph_sdk.auth import Auth


auth = Auth()

KEYCLOAK_ISSUER = os.getenv(
    "KEYCLOAK_ISSUER",
    "http://127.0.0.1:8080/realms/langchain-agent",
).rstrip("/")
KEYCLOAK_AUDIENCE = os.getenv(
    "KEYCLOAK_AUDIENCE",
    "langsmith-agent-api",
)
KEYCLOAK_JWKS_URL = os.getenv(
    "KEYCLOAK_JWKS_URL",
    f"{KEYCLOAK_ISSUER}/protocol/openid-connect/certs",
)
KEYCLOAK_ALLOWED_CLIENTS = frozenset(
    client.strip()
    for client in os.getenv(
        "KEYCLOAK_ALLOWED_CLIENTS",
        "langsmith-agent-ui",
    ).split(",")
    if client.strip()
)
KEYCLOAK_ALLOWED_ROLES = frozenset(
    role.strip()
    for role in os.getenv(
        "KEYCLOAK_ALLOWED_ROLES",
        "agent-user,agent-admin",
    ).split(",")
    if role.strip()
)
KEYCLOAK_REQUIRED_ROLE = os.getenv("KEYCLOAK_REQUIRED_ROLE", "agent-user")

_issuer = urlparse(KEYCLOAK_ISSUER)
if _issuer.scheme not in {"http", "https"} or not _issuer.hostname:
    raise RuntimeError("KEYCLOAK_ISSUER must be an absolute HTTP(S) URL")
if _issuer.scheme == "http" and _issuer.hostname not in {"127.0.0.1", "localhost"}:
    raise RuntimeError("Use HTTPS for a non-loopback KEYCLOAK_ISSUER")
if not KEYCLOAK_AUDIENCE:
    raise RuntimeError("KEYCLOAK_AUDIENCE must not be empty")
if not KEYCLOAK_ALLOWED_CLIENTS:
    raise RuntimeError("KEYCLOAK_ALLOWED_CLIENTS must contain at least one client")
if KEYCLOAK_REQUIRED_ROLE not in KEYCLOAK_ALLOWED_ROLES:
    raise RuntimeError("KEYCLOAK_REQUIRED_ROLE must be in KEYCLOAK_ALLOWED_ROLES")

_jwks_client = jwt.PyJWKClient(
    KEYCLOAK_JWKS_URL,
    cache_jwk_set=True,
    lifespan=300,
)


def _decode_token(token: str) -> dict[str, Any]:
    """Verify a Keycloak access token and return its claims."""
    signing_key = _jwks_client.get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=KEYCLOAK_AUDIENCE,
        issuer=KEYCLOAK_ISSUER,
        leeway=30,
        options={"require": ["sub", "iss", "aud", "exp", "iat"]},
    )

    if claims.get("azp") not in KEYCLOAK_ALLOWED_CLIENTS:
        raise jwt.InvalidTokenError("Unexpected authorized party")
    if claims.get("typ") not in {None, "Bearer"}:
        raise jwt.InvalidTokenError("Expected a bearer access token")
    return claims


def _unauthorized(detail: str = "Invalid bearer token") -> Auth.exceptions.HTTPException:
    return Auth.exceptions.HTTPException(status_code=401, detail=detail)


@auth.authenticate
async def authenticate(
    authorization: str | None,
) -> Auth.types.MinimalUserDict:
    """Authenticate one Agent Server request with a Keycloak access token."""
    if not authorization:
        raise _unauthorized("Bearer token required")

    scheme, separator, token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not token
        or " " in token
        or len(token) > 16_384
    ):
        raise _unauthorized()

    try:
        claims = await asyncio.to_thread(_decode_token, token)
    except jwt.PyJWTError as exc:
        raise _unauthorized() from exc

    realm_access = claims.get("realm_access")
    role_claim = (
        realm_access.get("roles", []) if isinstance(realm_access, dict) else []
    )
    if not isinstance(role_claim, list) or not all(
        isinstance(role, str) for role in role_claim
    ):
        raise _unauthorized()
    realm_roles = set(role_claim)
    if KEYCLOAK_REQUIRED_ROLE not in realm_roles:
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail="Agent access denied",
        )

    user: dict[str, Any] = {
        "identity": str(claims["sub"]),
        "is_authenticated": True,
        "permissions": sorted(realm_roles & KEYCLOAK_ALLOWED_ROLES),
        "display_name": str(
            claims.get("preferred_username") or claims["sub"]
        ),
    }

    # The demo realm maps this optional user attribute into the access token.
    # It is useful for the Chinook account tools, but `sub` remains the stable
    # security identity used for Agent Server resource ownership.
    customer_id = claims.get("customer_id")
    if customer_id is not None:
        try:
            parsed_customer_id = int(customer_id)
        except (TypeError, ValueError) as exc:
            raise _unauthorized() from exc
        if parsed_customer_id <= 0:
            raise _unauthorized()
        user["customer_id"] = parsed_customer_id

    return cast(Auth.types.MinimalUserDict, user)


@auth.on
async def deny_unhandled_resources(
    ctx: Auth.types.AuthContext,
    value: Any,
) -> bool:
    """Deny resources unless a more-specific handler allows them."""
    del ctx, value
    return False


@auth.on.threads
async def scope_threads_to_authenticated_user(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.value,
) -> dict[str, str]:
    """Attach and enforce per-user ownership for every thread operation."""
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)

    # TODO: optionally you can add customer_id from WT claims to the context
    # if ctx.action == "create_run":
    #     customer_id = (
    #         ctx.user["customer_id"]
    #         if "customer_id" in ctx.user
    #         else None
    #     )
    return filters


@auth.on.store
async def scope_store_to_authenticated_user(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.store.value,
) -> None:
    """Prefix Store namespaces so one Keycloak user cannot read another's."""
    namespace = tuple(value["namespace"]) if value.get("namespace") else ()
    if not namespace or namespace[0] != ctx.user.identity:
        namespace = (ctx.user.identity, *namespace)
    value["namespace"] = namespace
