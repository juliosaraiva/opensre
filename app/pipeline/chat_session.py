"""Multi-turn chat surface backed by ``ClaudeSDKClient``.

Phase 4 of the LangGraph→Claude Agent SDK migration. This is the eventual
replacement for ``app.nodes.chat.chat_agent_node`` and ``tool_executor_node``:
the SDK owns the LLM↔tool loop instead of the bespoke loop in
``tool_executor_node``. Both implementations coexist for now (this module is
additive); a later phase wires the chat sub-pipeline to use ``ChatSession``.

Design notes:

- The session is an async context manager so the underlying SDK client is
  connected/disconnected deterministically.
- The SDK client is injected via ``client_factory`` so tests can substitute a
  fake without depending on the Claude Code CLI subprocess. Production
  defaults to ``ClaudeSDKClient(options=...)``.
- ``send(prompt)`` returns a :class:`ChatTurn` summarising the model's reply
  (concatenated text, tool-use blocks, terminal cost/error flags). Callers
  that need raw messages get them via ``ChatTurn.raw_messages``.
- ``ChatSession`` does not build :class:`ClaudeAgentOptions` itself — callers
  use :func:`app.pipeline.sdk_runtime.build_chat_options` and pass the result
  in. This keeps the session class focused on conversation state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, Protocol, Self, runtime_checkable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    Message,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from app.constants.prompts import ROUTER_PROMPT
from app.services import get_llm_for_tools


@runtime_checkable
class ChatClientProtocol(Protocol):
    """Minimal slice of ``ClaudeSDKClient`` that :class:`ChatSession` depends on."""

    async def connect(self, prompt: Any = None) -> None: ...

    async def disconnect(self) -> None: ...

    async def query(self, prompt: str, session_id: str = ...) -> None: ...

    def receive_response(self) -> AsyncIterator[Message]: ...


ClientFactory = Callable[[ClaudeAgentOptions], ChatClientProtocol]


@dataclass(frozen=True)
class ChatTurn:
    """The model's reply to one user message."""

    text: str
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    raw_messages: list[Message] = field(default_factory=list)
    is_error: bool = False
    cost_usd: float | None = None
    session_id: str | None = None
    stop_reason: str | None = None


def _default_client_factory(options: ClaudeAgentOptions) -> ChatClientProtocol:
    return ClaudeSDKClient(options=options)


class ChatSession:
    """Multi-turn chat backed by a Claude Agent SDK client.

    Use as an async context manager::

        async with ChatSession(options) as chat:
            turn = await chat.send("hello")
            print(turn.text)
            turn = await chat.send("follow up")  # same session
    """

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._options = options
        self._client_factory = client_factory or _default_client_factory
        self._client: ChatClientProtocol | None = None

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = self._client_factory(self._options)
        await self._client.connect()

    async def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.disconnect()
        finally:
            self._client = None

    async def send(self, prompt: str, *, session_id: str = "default") -> ChatTurn:
        if self._client is None:
            raise RuntimeError(
                "ChatSession is not connected — use as async context manager or call connect()."
            )
        await self._client.query(prompt, session_id=session_id)
        return await _collect_turn(self._client)


async def _collect_turn(client: ChatClientProtocol) -> ChatTurn:
    """Drain one round of messages from the client into a :class:`ChatTurn`."""
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    thinking_parts: list[str] = []
    raw: list[Message] = []
    is_error = False
    cost_usd: float | None = None
    session_id: str | None = None
    stop_reason: str | None = None

    async for msg in client.receive_response():
        raw.append(msg)
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    thinking_parts.append(block.thinking)
                elif isinstance(block, ToolUseBlock):
                    tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
            if msg.session_id and session_id is None:
                session_id = msg.session_id
        elif isinstance(msg, ResultMessage):
            is_error = bool(msg.is_error)
            cost_usd = msg.total_cost_usd
            stop_reason = msg.stop_reason
            session_id = msg.session_id or session_id

    return ChatTurn(
        text="".join(text_parts),
        tool_uses=tool_uses,
        thinking=thinking_parts,
        raw_messages=raw,
        is_error=is_error,
        cost_usd=cost_usd,
        session_id=session_id,
        stop_reason=stop_reason,
    )


def route_chat_intent(
    user_text: str,
    *,
    invoke: Callable[[list[dict[str, str]]], Awaitable[Any] | Any] | None = None,
) -> Literal["tracer_data", "general"]:
    """Decide whether a user message should hit Tracer-data tools or the general LLM.

    Pure-function counterpart to ``app.nodes.chat.router_node``: takes only the
    user's text and returns the route. ``invoke`` is the LLM callable; defaults
    to ``get_llm_for_tools().invoke`` so production wiring is unchanged.
    """
    invoker = invoke or (lambda msgs: get_llm_for_tools().invoke(msgs))
    response = invoker(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": user_text},
        ]
    )
    route = str(getattr(response, "content", response)).strip().lower()
    return "tracer_data" if route == "tracer_data" else "general"


__all__ = [
    "ChatClientProtocol",
    "ChatSession",
    "ChatTurn",
    "ClientFactory",
    "route_chat_intent",
]
