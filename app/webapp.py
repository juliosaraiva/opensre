"""Standalone FastAPI surface for OpenSRE.

Phase 5 of the LangGraph→Claude Agent SDK migration: the webapp is now
self-hosting (``uvicorn app.webapp:app``) instead of being mounted under the
LangGraph SDK. It exposes:

- ``GET  /health``                        — unauthenticated liveness probe
- ``POST /investigations``                — synchronous RCA run
- ``POST /investigations/stream``         — RCA run as Server-Sent Events
- ``POST /chat``                          — single chat turn
- ``POST /chat/stream``                   — chat turn streamed as SSE

All POST endpoints require a Clerk JWT in ``Authorization: Bearer ...`` —
verification is in :mod:`app.auth.middleware`. The agent state is
pre-populated with the authenticated user's identity, so legacy nodes that
read ``state["org_id"]`` / ``state["user_id"]`` continue to work unchanged
(``inject_auth_node`` is still in the graph).

The investigation runner respects ``OPENSRE_RUNNER`` (see
:mod:`app.pipeline.runners`); the chat path still goes through the legacy
chat nodes wired in ``app.pipeline.runners.run_chat`` until a later phase.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any, cast

from fastapi import FastAPI, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth.middleware import AuthenticatedUser
from app.config import LLMSettings, get_environment
from app.state import AgentState
from app.version import get_version


class HealthResponse(BaseModel):
    ok: bool
    version: str
    graph_loaded: bool
    llm_configured: bool
    env: str


class InvestigationRequest(BaseModel):
    alert_name: str
    pipeline_name: str = ""
    severity: str = "unknown"
    raw_alert: str | dict[str, Any] | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict[str, str]] | None = None


app = FastAPI(title="OpenSRE", version=get_version())


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


def _graph_loaded() -> bool:
    return "app.graph_pipeline" in sys.modules


def _llm_configured() -> bool:
    try:
        LLMSettings.from_env()
    except Exception:
        return False
    return True


def get_health_response() -> HealthResponse:
    graph_loaded = _graph_loaded()
    llm_configured = _llm_configured()

    return HealthResponse(
        ok=graph_loaded and llm_configured,
        version=get_version(),
        graph_loaded=graph_loaded,
        llm_configured=llm_configured,
        env=get_environment().value,
    )


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    health_response = get_health_response()
    response.status_code = (
        status.HTTP_200_OK if health_response.ok else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return health_response


# ─────────────────────────────────────────────────────────────────────────────
# Investigations
# ─────────────────────────────────────────────────────────────────────────────


def _build_initial_investigation_state(
    req: InvestigationRequest, user: AuthenticatedUser
) -> AgentState:
    """Create the initial investigation state pre-populated with auth context."""
    from app.state import make_initial_state  # lazy: pulls in heavy modules

    state = make_initial_state(
        req.alert_name,
        req.pipeline_name,
        req.severity,
        raw_alert=req.raw_alert,
    )
    cast(dict[str, Any], state).update(user.as_state_fields())
    return state


def _run_investigation_sync(state: AgentState) -> AgentState:
    """Dispatch to the active runner — procedural or LangGraph — based on env."""
    from app.pipeline.runners import _runner_choice  # noqa: PLC0415

    if _runner_choice() == "procedural":
        from app.pipeline.driver import run_investigation_async  # noqa: PLC0415

        return asyncio.run(run_investigation_async(state))

    from app.pipeline.graph import graph as compiled_graph  # noqa: PLC0415

    return cast(AgentState, compiled_graph.invoke(state))


@app.post("/investigations")
async def post_investigation(req: InvestigationRequest, user: AuthenticatedUser) -> dict[str, Any]:
    """Run an investigation synchronously and return the final state."""
    state = _build_initial_investigation_state(req, user)
    final = await asyncio.to_thread(_run_investigation_sync, state)
    return cast(dict[str, Any], final)


@app.post("/investigations/stream")
async def post_investigation_stream(
    req: InvestigationRequest, user: AuthenticatedUser
) -> StreamingResponse:
    """Stream investigation events as SSE.

    Wire format: each frame is ``event: <type>\\n data: <json>\\n\\n``, matching
    what ``app.remote.stream.parse_sse_stream`` already consumes. Procedural
    runs use the queue-based driver; LangGraph runs use ``astream_events``
    via the existing mapper.
    """
    state = _build_initial_investigation_state(req, user)
    return StreamingResponse(_investigation_sse(state), media_type="text/event-stream")


async def _investigation_sse(state: AgentState) -> AsyncIterator[bytes]:
    from app.pipeline.runners import _astream_procedural, _runner_choice  # noqa: PLC0415

    if _runner_choice() == "procedural":
        async for event in _astream_procedural(state):
            yield _format_sse(event.event_type, asdict(event))
        return

    from app.pipeline.graph import graph as compiled_graph  # noqa: PLC0415
    from app.pipeline.runners import _map_langgraph_event  # noqa: PLC0415

    async for raw_event in compiled_graph.astream_events(state, version="v2"):
        event = _map_langgraph_event(dict(raw_event))
        yield _format_sse(event.event_type, asdict(event))


def _format_sse(event_type: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n".encode()


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────


def _build_chat_state(req: ChatRequest, user: AuthenticatedUser) -> AgentState:
    from app.state import make_chat_state  # noqa: PLC0415

    history = list(req.history or [])
    history.append({"role": "user", "content": req.message})
    state = make_chat_state(
        org_id=user.org_id,
        user_id=user.user_id,
        user_email=user.user_email,
        user_name=user.user_name,
        organization_slug=user.organization_slug,
        messages=history,  # type: ignore[arg-type]
    )
    cast(dict[str, Any], state).update(user.as_state_fields())
    return state


def _run_chat_sync(state: AgentState) -> AgentState:
    from app.pipeline.runners import run_chat  # noqa: PLC0415

    return run_chat(state)


@app.post("/chat")
async def post_chat(req: ChatRequest, user: AuthenticatedUser) -> dict[str, Any]:
    """Single chat turn — routing + response — using the existing chat nodes."""
    state = _build_chat_state(req, user)
    final = await asyncio.to_thread(_run_chat_sync, state)
    return cast(dict[str, Any], final)


@app.post("/chat/stream")
async def post_chat_stream(req: ChatRequest, user: AuthenticatedUser) -> StreamingResponse:
    """Chat turn as SSE — emits a single ``message`` frame with the final state.

    The legacy chat path (LangGraph chat agent + tool executor) does not stream
    intermediate tokens, so this is currently a thin wrapper that lets clients
    use the same SSE plumbing. A later phase swaps in :class:`ChatSession`
    streaming via ``app.pipeline.stream_adapter.map_sdk_message``.
    """
    state = _build_chat_state(req, user)

    async def _stream() -> AsyncIterator[bytes]:
        final = await asyncio.to_thread(_run_chat_sync, state)
        yield _format_sse("message", cast(dict[str, Any], final))

    return StreamingResponse(_stream(), media_type="text/event-stream")
