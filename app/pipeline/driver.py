"""Procedural investigation driver — same behavior as the LangGraph pipeline,
no graph engine.

Phase 3 of the LangGraph→Claude Agent SDK migration. This module replaces the
``StateGraph`` in :mod:`app.pipeline.graph` with a plain ``async`` function
that calls the existing pure-function nodes in sequence and applies the same
routing decisions inline. Both runners coexist behind the ``OPENSRE_RUNNER``
flag (handled in :mod:`app.pipeline.runners`) so we can A/B them in CI.

The driver:

- Mutates a single ``AgentState`` dict in place (no concurrent reducers, so
  the LangGraph ``add_messages`` reducer is unnecessary; for parity with the
  legacy runner, ``messages`` is still appended via ``_merge_state``).
- Calls every node synchronously inside the ``async`` body — the nodes are
  themselves sync today, and an async surface lets later phases adopt
  Claude SDK streaming without breaking the API.
- Emits :class:`StreamEvent`-shaped progress markers through an optional
  ``asyncio.Queue`` so the existing remote/web frontend can consume the same
  wire format as ``astream_events`` did.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from app.nodes import (
    node_diagnose_root_cause,
    node_extract_alert,
    node_plan_actions,
    node_publish_findings,
    node_resolve_integrations,
)
from app.nodes.auth import inject_auth_node
from app.nodes.investigate.node import node_investigate
from app.pipeline.routing import route_after_extract, route_investigation_loop
from app.pipeline.stream_adapter import stage_event
from app.remote.stream import StreamEvent
from app.state import AgentState

logger = logging.getLogger(__name__)

EventQueue = asyncio.Queue[StreamEvent | None]


def _merge_state(state: AgentState, updates: dict[str, Any]) -> None:
    """Apply node updates to ``state`` with append-semantics for ``messages``.

    Mirrors :func:`app.pipeline.runners._merge_state` so chat-mode and
    investigation-mode mutations stay consistent across the procedural and
    LangGraph runners.
    """
    if not updates:
        return
    state_any = cast(dict[str, Any], state)
    for key, value in updates.items():
        if key == "messages":
            messages = list(state_any.get("messages", []))
            if isinstance(value, list):
                messages.extend(value)
            else:
                messages.append(value)
            state_any["messages"] = messages
            continue
        state_any[key] = value


async def _emit(queue: EventQueue | None, event: StreamEvent) -> None:
    if queue is not None:
        await queue.put(event)


async def _run_stage(
    queue: EventQueue | None,
    name: str,
    func,
    /,
    *args,
    **kwargs,
) -> dict[str, Any]:
    """Run one node, emit on_chain_start/on_chain_end markers, return its update."""
    await _emit(queue, stage_event(name, "on_chain_start"))
    try:
        update: dict[str, Any] = func(*args, **kwargs) or {}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage %s failed", name)
        await _emit(queue, stage_event(name, "on_chain_error", {"error": str(exc)}))
        raise
    await _emit(queue, stage_event(name, "on_chain_end"))
    return update


async def run_investigation_async(
    initial: AgentState,
    config: RunnableConfig | None = None,
    *,
    queue: EventQueue | None = None,
) -> AgentState:
    """Run the investigation pipeline procedurally.

    Equivalent in behavior to ``app.pipeline.graph.graph.invoke(initial, config)``
    for ``mode="investigation"`` states. The function mutates ``initial`` and
    returns the same dict.
    """
    cfg: RunnableConfig = config or {"configurable": {}}
    state = initial

    _merge_state(state, await _run_stage(queue, "inject_auth", inject_auth_node, state, cfg))

    _merge_state(state, await _run_stage(queue, "extract_alert", node_extract_alert, state))
    if route_after_extract(state) == "end":
        if queue is not None:
            await queue.put(None)
        return state

    _merge_state(
        state,
        await _run_stage(queue, "resolve_integrations", node_resolve_integrations, state, cfg),
    )

    while True:
        _merge_state(state, await _run_stage(queue, "plan_actions", node_plan_actions, state))
        _merge_state(state, await _run_stage(queue, "investigate", node_investigate, state))
        _merge_state(state, await _run_stage(queue, "diagnose", node_diagnose_root_cause, state))
        if route_investigation_loop(state) == "publish":
            break

    _merge_state(state, await _run_stage(queue, "publish", node_publish_findings, state))

    if queue is not None:
        await queue.put(None)  # sentinel: stream complete
    return state


__all__ = ["EventQueue", "run_investigation_async"]
