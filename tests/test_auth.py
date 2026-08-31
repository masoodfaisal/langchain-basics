"""Focused unit tests for the Keycloak Agent Server auth handler."""

from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from langgraph_sdk.auth import Auth

import auth as auth_module


def _claims(**overrides):
    claims = {
        "sub": "keycloak-user-id",
        "azp": "langsmith-agent-ui",
        "typ": "Bearer",
        "preferred_username": "demo-user",
        "realm_access": {"roles": ["agent-user"]},
        "customer_id": 1,
    }
    claims.update(overrides)
    return claims


def _encoded_token(private_key, **overrides):
    now = int(time.time())
    claims = {
        "sub": "keycloak-user-id",
        "iss": auth_module.KEYCLOAK_ISSUER,
        "aud": auth_module.KEYCLOAK_AUDIENCE,
        "exp": now + 300,
        "iat": now,
        "azp": sorted(auth_module.KEYCLOAK_ALLOWED_CLIENTS)[0],
        "typ": "Bearer",
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256")


@pytest.fixture
def signing_key(monkeypatch):
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    monkeypatch.setattr(
        auth_module._jwks_client,
        "get_signing_key_from_jwt",
        lambda _token: SimpleNamespace(key=private_key.public_key()),
    )
    return private_key


@pytest.mark.asyncio
async def test_authenticate_requires_bearer_token():
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await auth_module.authenticate(None)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_authenticate_returns_verified_identity(monkeypatch):
    monkeypatch.setattr(auth_module, "_decode_token", lambda _token: _claims())

    user = await auth_module.authenticate("Bearer signed-token")

    assert user["identity"] == "keycloak-user-id"
    assert user["display_name"] == "demo-user"
    assert user["permissions"] == ["agent-user"]
    assert user["customer_id"] == 1


@pytest.mark.asyncio
async def test_authenticate_rejects_invalid_jwt(monkeypatch):
    def reject(_token):
        raise jwt.InvalidTokenError("test error")

    monkeypatch.setattr(auth_module, "_decode_token", reject)

    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await auth_module.authenticate("Bearer invalid-token")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid bearer token"


@pytest.mark.asyncio
async def test_authenticate_requires_agent_role(monkeypatch):
    monkeypatch.setattr(
        auth_module,
        "_decode_token",
        lambda _token: _claims(realm_access={"roles": ["offline_access"]}),
    )

    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await auth_module.authenticate("Bearer signed-token")

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_authenticate_rejects_malformed_role_claim(monkeypatch):
    monkeypatch.setattr(
        auth_module,
        "_decode_token",
        lambda _token: _claims(realm_access={"roles": None}),
    )

    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await auth_module.authenticate("Bearer signed-token")

    assert exc_info.value.status_code == 401


def test_decode_token_verifies_signature_and_expected_claims(signing_key):
    token = _encoded_token(signing_key)

    claims = auth_module._decode_token(token)

    assert claims["sub"] == "keycloak-user-id"


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("aud", "another-api"),
        ("iss", "http://127.0.0.1:8080/realms/another-realm"),
        ("azp", "another-client"),
        ("exp", 1),
    ],
)
def test_decode_token_rejects_untrusted_claims(signing_key, override, value):
    token = _encoded_token(signing_key, **{override: value})

    with pytest.raises(jwt.PyJWTError):
        auth_module._decode_token(token)


@pytest.mark.asyncio
async def test_thread_scope_stamps_and_filters_owner():
    ctx = SimpleNamespace(user=SimpleNamespace(identity="user-a"))
    value = {"metadata": {"source": "test"}}

    filters = await auth_module.scope_threads_to_authenticated_user(ctx, value)

    assert filters == {"owner": "user-a"}
    assert value["metadata"] == {"source": "test", "owner": "user-a"}


@pytest.mark.asyncio
async def test_store_scope_prefixes_authenticated_identity():
    ctx = SimpleNamespace(user=SimpleNamespace(identity="user-a"))
    value = {"namespace": ["memories"]}

    await auth_module.scope_store_to_authenticated_user(ctx, value)

    assert value["namespace"] == ("user-a", "memories")


@pytest.mark.asyncio
async def test_unhandled_resources_are_denied():
    assert await auth_module.deny_unhandled_resources(None, {}) is False
