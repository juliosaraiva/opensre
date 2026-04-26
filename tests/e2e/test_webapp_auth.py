"""E2E coverage for the Phase 5 standalone FastAPI surface.

The Clerk JWT verification machinery is patched at the boundary
(``verify_jwt_async``) so we exercise the full middleware → endpoint flow
without standing up a real JWKS server. Investigation and chat runners are
patched to deterministic stubs — the goal is to prove auth, tenant tagging,
and request/response wiring, not the agent itself.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_auth import (
    JWTClaims,
    JWTExpiredError,
    JWTVerificationError,
)
from app.auth.middleware import UserContext, tenant_filter
from app.webapp import app

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _claims() -> JWTClaims:
    return JWTClaims(
        sub="user-1",
        organization="org-1",
        organization_slug="acme",
        email="alice@example.com",
        full_name="Alice",
        issuer="https://clerk.example.com",
        exp=9_999_999_999,
        iat=0,
    )


def _patch_verify(returns: JWTClaims | Exception) -> AsyncMock:
    """Patch ``verify_jwt_async`` on the middleware module."""
    if isinstance(returns, Exception):
        mock = AsyncMock(side_effect=returns)
    else:
        mock = AsyncMock(return_value=returns)
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# UserContext / tenant_filter
# ─────────────────────────────────────────────────────────────────────────────


def test_user_context_from_claims_carries_all_identity_fields() -> None:
    user = UserContext.from_claims(_claims(), raw_token="tok-xyz")
    assert user.org_id == "org-1"
    assert user.user_id == "user-1"
    assert user.user_email == "alice@example.com"
    assert user.user_name == "Alice"
    assert user.organization_slug == "acme"
    assert user.raw_token == "tok-xyz"


def test_user_context_as_state_fields_matches_inject_auth_node_keys() -> None:
    """``UserContext.as_state_fields()`` must produce the same shape that the
    legacy ``inject_auth_node`` writes, so endpoints can pre-populate state
    without needing the auth node to run."""
    fields = UserContext.from_claims(_claims(), raw_token="tok").as_state_fields()
    assert set(fields) == {
        "org_id",
        "user_id",
        "user_email",
        "user_name",
        "organization_slug",
        "_auth_token",
    }


def test_tenant_filter_returns_org_scoped_dict() -> None:
    user = UserContext.from_claims(_claims())
    assert tenant_filter(user) == {"org_id": "org-1"}


# ─────────────────────────────────────────────────────────────────────────────
# Health is unauthenticated
# ─────────────────────────────────────────────────────────────────────────────


def test_health_does_not_require_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "version" in body and "ok" in body


# ─────────────────────────────────────────────────────────────────────────────
# Auth dependency
# ─────────────────────────────────────────────────────────────────────────────


def test_post_endpoint_returns_401_without_authorization_header(
    client: TestClient,
) -> None:
    response = client.post("/investigations", json={"alert_name": "x"})
    assert response.status_code == 401
    assert "Authorization" in response.json()["detail"]


def test_post_endpoint_returns_401_with_malformed_authorization(
    client: TestClient,
) -> None:
    response = client.post(
        "/investigations",
        json={"alert_name": "x"},
        headers={"Authorization": "Token abc"},
    )
    assert response.status_code == 401
    assert "Bearer" in response.json()["detail"]


def test_post_endpoint_returns_401_when_jwt_is_expired(client: TestClient) -> None:
    with patch(
        "app.auth.middleware.verify_jwt_async",
        new=_patch_verify(JWTExpiredError("expired")),
    ):
        response = client.post(
            "/investigations",
            json={"alert_name": "x"},
            headers={"Authorization": "Bearer dead.beef.token"},
        )
    assert response.status_code == 401
    assert response.json()["detail"] == "JWT expired"


def test_post_endpoint_returns_401_when_jwt_signature_is_invalid(
    client: TestClient,
) -> None:
    with patch(
        "app.auth.middleware.verify_jwt_async",
        new=_patch_verify(JWTVerificationError("bad signature")),
    ):
        response = client.post(
            "/investigations",
            json={"alert_name": "x"},
            headers={"Authorization": "Bearer dead.beef.token"},
        )
    assert response.status_code == 401
    assert "bad signature" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# Investigation endpoint — tenant tagging
# ─────────────────────────────────────────────────────────────────────────────


def test_post_investigations_passes_authenticated_user_into_state(
    client: TestClient,
) -> None:
    """The endpoint must inject the user's identity into state before the runner sees it."""
    captured: dict[str, Any] = {}

    def _stub_runner(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        return state

    with (
        patch("app.auth.middleware.verify_jwt_async", new=_patch_verify(_claims())),
        patch("app.webapp._run_investigation_sync", new=_stub_runner),
    ):
        response = client.post(
            "/investigations",
            json={
                "alert_name": "PipelineError",
                "pipeline_name": "etl",
                "severity": "high",
                "raw_alert": {"text": "boom"},
            },
            headers={"Authorization": "Bearer good.token"},
        )

    assert response.status_code == 200
    assert captured["org_id"] == "org-1"
    assert captured["user_id"] == "user-1"
    assert captured["organization_slug"] == "acme"
    assert captured["alert_name"] == "PipelineError"
    assert captured["raw_alert"] == {"text": "boom"}


def test_post_investigations_returns_runner_result(client: TestClient) -> None:
    def _stub_runner(state: dict[str, Any]) -> dict[str, Any]:
        state.update({"report": "FINAL", "validity_score": 0.9})
        return state

    with (
        patch("app.auth.middleware.verify_jwt_async", new=_patch_verify(_claims())),
        patch("app.webapp._run_investigation_sync", new=_stub_runner),
    ):
        response = client.post(
            "/investigations",
            json={"alert_name": "x", "severity": "low"},
            headers={"Authorization": "Bearer good.token"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["report"] == "FINAL"
    assert body["validity_score"] == 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────────────────────────────────────


def test_post_chat_pre_seeds_messages_and_user(client: TestClient) -> None:
    captured: dict[str, Any] = {}

    def _stub_chat(state: dict[str, Any]) -> dict[str, Any]:
        captured.update(state)
        state["messages"] = [*state.get("messages", []), {"role": "assistant", "content": "hi"}]
        return state

    with (
        patch("app.auth.middleware.verify_jwt_async", new=_patch_verify(_claims())),
        patch("app.webapp._run_chat_sync", new=_stub_chat),
    ):
        response = client.post(
            "/chat",
            json={"message": "hello"},
            headers={"Authorization": "Bearer good.token"},
        )

    assert response.status_code == 200
    assert captured["org_id"] == "org-1"
    last_user = captured["messages"][-1]
    assert last_user["role"] == "user"
    assert last_user["content"] == "hello"
    assert response.json()["messages"][-1]["content"] == "hi"


def test_post_chat_rejects_empty_message(client: TestClient) -> None:
    with patch("app.auth.middleware.verify_jwt_async", new=_patch_verify(_claims())):
        response = client.post(
            "/chat",
            json={"message": ""},
            headers={"Authorization": "Bearer good.token"},
        )
    # Pydantic's min_length=1 → 422 Unprocessable Entity
    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Streaming
# ─────────────────────────────────────────────────────────────────────────────


def test_post_investigations_stream_emits_sse_frames(client: TestClient) -> None:
    """The /investigations/stream endpoint must serialise StreamEvents as SSE frames."""

    async def _fake_sse(_state: dict[str, Any]):
        yield b'event: events\ndata: {"node": "extract_alert", "kind": "on_chain_start"}\n\n'
        yield b'event: end\ndata: {"node": "publish", "kind": "on_chain_end"}\n\n'

    with (
        patch("app.auth.middleware.verify_jwt_async", new=_patch_verify(_claims())),
        patch("app.webapp._investigation_sse", new=_fake_sse),
    ):
        response = client.post(
            "/investigations/stream",
            json={"alert_name": "x"},
            headers={"Authorization": "Bearer good.token"},
        )
        body = response.text

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: events" in body
    assert "event: end" in body
    assert "extract_alert" in body
    assert "on_chain_start" in body
