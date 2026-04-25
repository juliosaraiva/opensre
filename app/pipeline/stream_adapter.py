"""Map Claude Agent SDK messages to OpenSRE :class:`StreamEvent`.

Phase 1 scaffolding: this module is currently unwired. It will be the only
place that knows how SDK messages map onto the legacy ``StreamEvent`` shape
that ``app/remote/renderer.py`` and ``app/remote/client.py`` consume — keeping
the wire format identical lets us swap the producer without touching the
consumers.

Mapping:

============================================  ====================  =========================
SDK message                                   ``event_type``        ``kind``
============================================  ====================  =========================
``SystemMessage(subtype="init")``             ``metadata``          ``on_chain_start``
``AssistantMessage`` w/ ``TextBlock``         ``events``            ``on_chat_model_stream``
``AssistantMessage`` w/ ``ThinkingBlock``     ``events``            ``on_chain_thought``
``AssistantMessage`` w/ ``ToolUseBlock``      ``events``            ``on_tool_start``
``UserMessage`` w/ tool result content        ``events``            ``on_tool_end``
``ResultMessage``                             ``end``               ``on_chain_end``
============================================  ====================  =========================

For any other message type :func:`map_sdk_message` returns ``None`` so the
caller can decide whether to skip silently.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from app.remote.stream import StreamEvent


def map_sdk_message(message: Message, *, node_name: str = "") -> list[StreamEvent]:
    """Map one SDK message to zero or more :class:`StreamEvent`.

    A single ``AssistantMessage`` may contain multiple content blocks (e.g. a
    text reply followed by a tool call), so this returns a list rather than
    ``Optional[StreamEvent]``.

    Parameters:
        message: SDK message yielded by ``query()`` / ``client.receive_response()``.
        node_name: Optional pipeline-stage label injected by the procedural
            driver (``"extract_alert"``, ``"plan_actions"``, …). Mirrors the
            ``langgraph_node`` field that LangGraph's ``astream_events`` used
            to populate.
    """
    events: list[StreamEvent] = []

    if isinstance(message, SystemMessage):
        events.append(
            StreamEvent(
                event_type="metadata",
                data={"subtype": message.subtype, **_safe_dict(message.data)},
                node_name=node_name,
                kind="on_chain_start" if message.subtype == "init" else "",
            )
        )
        return events

    if isinstance(message, AssistantMessage):
        events.extend(_map_assistant(message, node_name=node_name))
        return events

    if isinstance(message, UserMessage):
        events.extend(_map_user(message, node_name=node_name))
        return events

    if isinstance(message, ResultMessage):
        events.append(
            StreamEvent(
                event_type="end",
                data={
                    "subtype": message.subtype,
                    "is_error": bool(message.is_error),
                    "duration_ms": message.duration_ms,
                    "num_turns": message.num_turns,
                    "session_id": message.session_id,
                    "stop_reason": message.stop_reason,
                    "result": message.result,
                    "total_cost_usd": message.total_cost_usd,
                },
                node_name=node_name,
                kind="on_chain_end",
                run_id=message.session_id or "",
            )
        )
        return events

    return events


def map_sdk_messages(messages: list[Message], *, node_name: str = "") -> Iterator[StreamEvent]:
    """Convenience: flatten a sequence of SDK messages into ``StreamEvent``s."""
    for msg in messages:
        yield from map_sdk_message(msg, node_name=node_name)


def stage_event(node_name: str, kind: str, data: dict[str, Any] | None = None) -> StreamEvent:
    """Emit a synthetic stage marker the procedural driver can push at boundaries.

    Mirrors the LangGraph ``on_chain_start`` / ``on_chain_end`` events that
    ``app/remote/renderer.py`` keys off ``node_name`` for stage progress UI.
    """
    return StreamEvent(
        event_type="events",
        data=data or {},
        node_name=node_name,
        kind=kind,
    )


def _map_assistant(message: AssistantMessage, *, node_name: str) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            events.append(
                StreamEvent(
                    event_type="events",
                    data={"text": block.text, "model": message.model},
                    node_name=node_name,
                    kind="on_chat_model_stream",
                    run_id=message.session_id or "",
                )
            )
        elif isinstance(block, ThinkingBlock):
            events.append(
                StreamEvent(
                    event_type="events",
                    data={"thinking": block.thinking},
                    node_name=node_name,
                    kind="on_chain_thought",
                    run_id=message.session_id or "",
                )
            )
        elif isinstance(block, ToolUseBlock):
            events.append(
                StreamEvent(
                    event_type="events",
                    data={
                        "tool_use_id": block.id,
                        "name": block.name,
                        "input": block.input,
                    },
                    node_name=node_name,
                    kind="on_tool_start",
                    run_id=message.session_id or "",
                )
            )
    return events


def _map_user(message: UserMessage, *, node_name: str) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    content = message.content
    if isinstance(content, str):
        return events
    for block in content:
        if isinstance(block, ToolResultBlock):
            events.append(
                StreamEvent(
                    event_type="events",
                    data={
                        "tool_use_id": block.tool_use_id,
                        "content": _normalize_tool_result(block.content),
                        "is_error": bool(block.is_error),
                    },
                    node_name=node_name,
                    kind="on_tool_end",
                )
            )
    return events


def _normalize_tool_result(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Tool-result blocks may contain a list of text/image dicts. Keep text
        # and stringify everything else for transport safety.
        flat: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                flat.append(item.get("text", ""))
            else:
                flat.append(str(item))
        return flat
    return str(content)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "map_sdk_message",
    "map_sdk_messages",
    "stage_event",
]
