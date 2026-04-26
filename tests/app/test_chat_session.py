"""Tests for the Phase 4 :class:`ChatSession` and ``route_chat_intent``.

Uses a fake :class:`ChatClientProtocol` implementation so the tests don't
require the Claude Code CLI subprocess. The real production path uses
``ClaudeSDKClient(options=options)`` which is exercised in higher-level
integration tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from app.pipeline.chat_session import ChatSession, ChatTurn, route_chat_intent

# ─────────────────────────────────────────────────────────────────────────────
# Fake SDK client
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeClient:
    options: ClaudeAgentOptions
    pending_responses: list[list[Message]] = field(default_factory=list)
    queries: list[tuple[str, str]] = field(default_factory=list)
    connected: bool = False
    disconnected: bool = False

    async def connect(self, prompt: Any = None) -> None:  # noqa: ARG002
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.queries.append((prompt, session_id))

    def receive_response(self) -> AsyncIterator[Message]:
        return self._stream_next()

    async def _stream_next(self) -> AsyncIterator[Message]:
        if not self.pending_responses:
            return
        for msg in self.pending_responses.pop(0):
            yield msg


def _assistant(*blocks: Any, session_id: str = "sess-1") -> AssistantMessage:
    return AssistantMessage(
        content=list(blocks),
        model="claude-opus-4",
        parent_tool_use_id=None,
        error=None,
        usage=None,
        message_id="m-1",
        stop_reason=None,
        session_id=session_id,
        uuid="u-1",
    )


def _result(*, is_error: bool = False, session_id: str = "sess-1") -> ResultMessage:
    return ResultMessage(
        subtype="success" if not is_error else "error_during_execution",
        duration_ms=10,
        duration_api_ms=5,
        is_error=is_error,
        num_turns=1,
        session_id=session_id,
        stop_reason="end_turn",
        total_cost_usd=0.0021,
        usage={},
        result="ok",
        structured_output=None,
        model_usage=None,
        permission_denials=None,
        errors=None,
        uuid="u-result",
    )


def _make_session(responses: list[list[Message]]) -> tuple[ChatSession, _FakeClient]:
    fake = _FakeClient(options=ClaudeAgentOptions(), pending_responses=responses)
    session = ChatSession(ClaudeAgentOptions(), client_factory=lambda _opts: fake)
    return session, fake


# ─────────────────────────────────────────────────────────────────────────────
# ChatSession behavior
# ─────────────────────────────────────────────────────────────────────────────


def test_send_collects_text_blocks_into_chat_turn() -> None:
    session, fake = _make_session(
        [
            [
                _assistant(TextBlock(text="hello, "), TextBlock(text="world")),
                _result(),
            ]
        ]
    )

    async def _run() -> ChatTurn:
        async with session as chat:
            return await chat.send("hi")

    turn = asyncio.run(_run())
    assert turn.text == "hello, world"
    assert turn.session_id == "sess-1"
    assert turn.cost_usd == pytest.approx(0.0021)
    assert turn.stop_reason == "end_turn"
    assert turn.is_error is False
    assert fake.connected is True
    assert fake.disconnected is True
    assert fake.queries == [("hi", "default")]


def test_send_captures_tool_use_blocks_in_order() -> None:
    session, _fake = _make_session(
        [
            [
                _assistant(
                    TextBlock(text="calling tools"),
                    ToolUseBlock(id="t-1", name="search_logs", input={"q": "errors"}),
                    ToolUseBlock(id="t-2", name="get_metrics", input={"window": "5m"}),
                ),
                _result(),
            ]
        ]
    )

    async def _run() -> ChatTurn:
        async with session as chat:
            return await chat.send("debug it")

    turn = asyncio.run(_run())
    assert [t["name"] for t in turn.tool_uses] == ["search_logs", "get_metrics"]
    assert turn.tool_uses[0]["input"] == {"q": "errors"}
    assert turn.tool_uses[1]["id"] == "t-2"


def test_send_captures_thinking_blocks() -> None:
    session, _fake = _make_session(
        [
            [
                _assistant(
                    ThinkingBlock(thinking="planning…", signature="sig-1"),
                    TextBlock(text="answer"),
                ),
                _result(),
            ]
        ]
    )

    async def _run() -> ChatTurn:
        async with session as chat:
            return await chat.send("ponder")

    turn = asyncio.run(_run())
    assert turn.thinking == ["planning…"]
    assert turn.text == "answer"


def test_send_propagates_error_flag_from_result() -> None:
    session, _fake = _make_session(
        [
            [
                _assistant(TextBlock(text="partial")),
                _result(is_error=True),
            ]
        ]
    )

    async def _run() -> ChatTurn:
        async with session as chat:
            return await chat.send("oops")

    turn = asyncio.run(_run())
    assert turn.is_error is True


def test_session_supports_multi_turn_with_separate_responses() -> None:
    session, fake = _make_session(
        [
            [_assistant(TextBlock(text="first")), _result()],
            [_assistant(TextBlock(text="second")), _result()],
        ]
    )

    async def _run() -> tuple[ChatTurn, ChatTurn]:
        async with session as chat:
            t1 = await chat.send("q1")
            t2 = await chat.send("q2", session_id="custom")
            return t1, t2

    t1, t2 = asyncio.run(_run())
    assert t1.text == "first"
    assert t2.text == "second"
    assert fake.queries == [("q1", "default"), ("q2", "custom")]


def test_send_before_connect_raises_runtime_error() -> None:
    session, _fake = _make_session([[]])

    async def _run() -> None:
        await session.send("hi")

    with pytest.raises(RuntimeError, match="not connected"):
        asyncio.run(_run())


def test_repeat_connect_is_idempotent() -> None:
    session, fake = _make_session([[_assistant(TextBlock(text="ok")), _result()]])

    async def _run() -> ChatTurn:
        await session.connect()
        await session.connect()  # second call must be a no-op
        try:
            return await session.send("ping")
        finally:
            await session.disconnect()

    turn = asyncio.run(_run())
    assert turn.text == "ok"
    assert fake.connected is True


def test_unknown_message_types_are_collected_into_raw_messages() -> None:
    """SystemMessage isn't TextBlock / ToolUse / Result — it must still land in raw."""
    session, _fake = _make_session(
        [
            [
                SystemMessage(subtype="init", data={"model": "claude-opus-4"}),
                _assistant(TextBlock(text="hi")),
                _result(),
            ]
        ]
    )

    async def _run() -> ChatTurn:
        async with session as chat:
            return await chat.send("ping")

    turn = asyncio.run(_run())
    assert any(isinstance(m, SystemMessage) for m in turn.raw_messages)
    assert turn.text == "hi"


# ─────────────────────────────────────────────────────────────────────────────
# route_chat_intent
# ─────────────────────────────────────────────────────────────────────────────


class _RouterReply:
    def __init__(self, content: str) -> None:
        self.content = content


def test_route_chat_intent_returns_tracer_data_when_router_says_so() -> None:
    invocations: list[list[dict[str, str]]] = []

    def _invoke(msgs: list[dict[str, str]]) -> _RouterReply:
        invocations.append(msgs)
        return _RouterReply("tracer_data")

    assert route_chat_intent("show me errors", invoke=_invoke) == "tracer_data"
    assert invocations[0][1]["content"] == "show me errors"
    assert invocations[0][0]["role"] == "system"


def test_route_chat_intent_falls_back_to_general_for_anything_else() -> None:
    def _invoke(_msgs: list[dict[str, str]]) -> _RouterReply:
        return _RouterReply("definitely not a known route")

    assert route_chat_intent("hi", invoke=_invoke) == "general"


def test_route_chat_intent_normalises_whitespace_and_case() -> None:
    def _invoke(_msgs: list[dict[str, str]]) -> _RouterReply:
        return _RouterReply("  TRACER_DATA\n")

    assert route_chat_intent("anything", invoke=_invoke) == "tracer_data"
