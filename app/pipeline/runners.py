"""Standalone runners for the procedural investigation pipeline and the chat path.

Phase 7 removed the LangGraph alternative; ``run_investigation`` and
``astream_investigation`` now go straight through
:mod:`app.pipeline.driver`. The chat helpers are kept as a thin convenience
that calls the chat node functions sequentially — the chat sub-pipeline still
uses the LangChain message types for now (a follow-up phase migrates it onto
:class:`app.pipeline.chat_session.ChatSession`).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from app.nodes.chat import chat_agent_node, general_node, router_node
from app.pipeline.driver import EventQueue, run_investigation_async
from app.remote.stream import StreamEvent
from app.state import AgentState, make_initial_state


def _merge_state(state: AgentState, updates: dict[str, Any]) -> None:
    if not updates:
        return
    state_any = cast(dict[str, Any], state)
    for key, value in updates.items():
        if key == "messages":
            messages = list(state_any.get("messages", []))
            messages.extend(value) if isinstance(value, list) else messages.append(value)
            state_any["messages"] = messages
            continue
        state_any[key] = value


def run_chat(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """Run chat routing + response."""
    cfg = config or {"configurable": {}}
    _merge_state(state, router_node(state))
    if state.get("route") == "tracer_data":
        _merge_state(state, chat_agent_node(state, cfg))
    else:
        _merge_state(state, general_node(state, cfg))
    return state


def run_investigation(
    alert_name: str,
    pipeline_name: str,
    severity: str,
    raw_alert: str | dict[str, Any] | None = None,
    resolved_integrations: dict[str, Any] | None = None,
) -> AgentState:
    """Run the investigation pipeline. Pure function: inputs in, state out.

    Args:
        resolved_integrations: Optional pre-resolved integrations dict. When provided,
            node_resolve_integrations is skipped — useful for synthetic testing where a
            FixtureGrafanaBackend should be injected without real credential resolution.
    """
    initial = make_initial_state(alert_name, pipeline_name, severity, raw_alert=raw_alert)
    if resolved_integrations is not None:
        cast(dict[str, Any], initial)["resolved_integrations"] = resolved_integrations
    return asyncio.run(run_investigation_async(initial))


async def astream_investigation(
    alert_name: str,
    pipeline_name: str,
    severity: str,
    raw_alert: str | dict[str, Any] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream investigation events as :class:`StreamEvent` objects.

    Driven by the queue-based driver in :mod:`app.pipeline.driver`. The wire
    format matches what ``app.remote.renderer`` and ``app.remote.client``
    already consume, so the terminal/web UX is unchanged.
    """
    initial = make_initial_state(alert_name, pipeline_name, severity, raw_alert=raw_alert)
    async for event in _astream_procedural(initial):
        yield event


async def _astream_procedural(initial: AgentState) -> AsyncIterator[StreamEvent]:
    """Drain :func:`run_investigation_async`'s event queue as it executes."""
    queue: EventQueue = asyncio.Queue()
    runner = asyncio.create_task(run_investigation_async(initial, queue=queue))
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        if not runner.done():
            # The driver-level stage emitted an on_chain_error event already;
            # we only need to drain the task so we don't leak it.
            with contextlib.suppress(Exception):
                await runner


@dataclass
class SimpleAgent:
    def invoke(self, state: AgentState, _config: RunnableConfig | None = None) -> AgentState:
        return asyncio.run(run_investigation_async(state))


__all__ = [
    "SimpleAgent",
    "astream_investigation",
    "run_chat",
    "run_investigation",
]
