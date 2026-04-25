"""Standalone runners for testing and CLI — run the pipeline without LangGraph."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig

from app.nodes.chat import chat_agent_node, general_node, router_node
from app.pipeline.driver import EventQueue, run_investigation_async
from app.remote.stream import StreamEvent
from app.state import AgentState, make_initial_state

RunnerChoice = Literal["langgraph", "procedural"]
_RUNNER_ENV_VAR = "OPENSRE_RUNNER"
# Phase 6: procedural is the default. The LangGraph runner remains available
# as an opt-in fallback (``OPENSRE_RUNNER=langgraph``) for one release window
# while we validate the procedural path under production load. Phase 7
# removes the LangGraph branch and this flag entirely.
_DEFAULT_RUNNER: RunnerChoice = "procedural"


def _runner_choice() -> RunnerChoice:
    """Return the active investigation runner.

    Defaults to ``procedural`` (Phase 6 onward). Set ``OPENSRE_RUNNER=langgraph``
    to fall back to the legacy ``StateGraph`` driver. Read on every invocation
    so tests and CLI calls can flip the flag at runtime without restarting
    the process.
    """
    raw = (os.environ.get(_RUNNER_ENV_VAR) or "").strip().lower()
    if raw == "langgraph":
        return "langgraph"
    if raw == "procedural":
        return "procedural"
    return _DEFAULT_RUNNER


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
    """Run chat routing + response without LangGraph (for testing)."""
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
    """Run investigation pipeline. Pure function: inputs in, state out.

    Dispatches to LangGraph or the procedural driver based on
    ``OPENSRE_RUNNER`` (``langgraph`` by default; set ``procedural`` to use
    :func:`app.pipeline.driver.run_investigation_async`).

    Args:
        resolved_integrations: Optional pre-resolved integrations dict. When provided,
            node_resolve_integrations is skipped — useful for synthetic testing where a
            FixtureGrafanaBackend should be injected without real credential resolution.
    """
    initial = make_initial_state(alert_name, pipeline_name, severity, raw_alert=raw_alert)
    if resolved_integrations is not None:
        cast(dict[str, Any], initial)["resolved_integrations"] = resolved_integrations

    if _runner_choice() == "procedural":
        return asyncio.run(run_investigation_async(initial))

    from app.pipeline.graph import graph as compiled_graph  # lazy to avoid circular import

    return cast(AgentState, compiled_graph.invoke(initial))


async def astream_investigation(
    alert_name: str,
    pipeline_name: str,
    severity: str,
    raw_alert: str | dict[str, Any] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream investigation events.

    Dispatches to LangGraph's ``astream_events`` or the procedural driver's
    queue-based stream based on ``OPENSRE_RUNNER``. Either way, yields
    :class:`StreamEvent` objects compatible with the remote
    ``StreamRenderer`` so local and remote investigations share the same
    terminal UX.
    """
    initial = make_initial_state(alert_name, pipeline_name, severity, raw_alert=raw_alert)

    if _runner_choice() == "procedural":
        async for event in _astream_procedural(initial):
            yield event
        return

    from app.pipeline.graph import graph as compiled_graph  # lazy to avoid circular import

    async for raw_event in compiled_graph.astream_events(initial, version="v2"):
        yield _map_langgraph_event(dict(raw_event))


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


def _map_langgraph_event(event: dict[str, Any]) -> StreamEvent:
    """Convert a raw LangGraph ``astream_events`` dict to a ``StreamEvent``."""
    kind = event.get("event", "")
    name = event.get("name", "")
    metadata = event.get("metadata", {})
    node_name = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
    tags = event.get("tags", [])
    run_id = event.get("run_id", "")
    data = {
        "event": kind,
        "name": name,
        "data": event.get("data", {}),
        "metadata": metadata,
    }

    return StreamEvent(
        event_type="events",
        data=data,
        node_name=node_name or name,
        kind=kind,
        run_id=run_id,
        tags=list(tags) if isinstance(tags, list) else [],
    )


@dataclass
class SimpleAgent:
    def invoke(self, state: AgentState, config: RunnableConfig | None = None) -> AgentState:
        cfg = config or {"configurable": {}}

        if _runner_choice() == "procedural":
            return asyncio.run(run_investigation_async(state, cfg))

        from app.pipeline.graph import graph as compiled_graph  # lazy to avoid circular import

        return cast(AgentState, compiled_graph.invoke(state, cfg))


__all__ = [
    "RunnerChoice",
    "SimpleAgent",
    "astream_investigation",
    "run_chat",
    "run_investigation",
]
