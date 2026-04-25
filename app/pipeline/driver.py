"""Procedural investigation driver — the only investigation runner.

Originally introduced in Phase 3 as an alternative to the LangGraph
``StateGraph`` (selected via ``OPENSRE_RUNNER=procedural``). Phase 7
removed the LangGraph runner entirely; this is now the single path.

The driver:

- Mutates a single ``AgentState`` dict in place. There is no graph, so no
  reducer infrastructure is needed; ``messages`` is appended via
  :func:`_merge_state` for parity with how the legacy LangGraph
  ``add_messages`` reducer behaved.
- Calls every node synchronously inside the ``async`` body — the nodes are
  themselves sync today, and an async surface lets later phases adopt
  Claude SDK streaming without breaking the API.
- Emits :class:`StreamEvent`-shaped progress markers through an optional
  ``asyncio.Queue`` so the existing remote/web frontend can consume the same
  wire format that LangGraph's ``astream_events`` used to produce.

Auth context is no longer injected here. The standalone webapp
(``app.webapp``) populates ``state`` with ``UserContext.as_state_fields()``
before invoking the driver, and direct callers (CLI, tests) supply state
themselves. The legacy ``inject_auth_node`` is gone.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from app.investigation_constants import MAX_INVESTIGATION_LOOPS
from app.nodes import (
    node_diagnose_root_cause,
    node_extract_alert,
    node_plan_actions,
    node_publish_findings,
    node_resolve_integrations,
)
from app.nodes.investigate.node import node_investigate
from app.pipeline.stream_adapter import stage_event
from app.remote.stream import StreamEvent
from app.state import AgentState

logger = logging.getLogger(__name__)

EventQueue = asyncio.Queue[StreamEvent | None]


def _merge_state(state: AgentState, updates: dict[str, Any]) -> None:
    """Apply node updates to ``state`` with append-semantics for ``messages``."""
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


def should_continue_investigation(state: AgentState) -> bool:
    """Decide whether the plan→investigate→diagnose loop should run again.

    Inlined from the deleted ``app.pipeline.routing.should_continue_investigation``.
    Returns True to loop; False to publish.
    """
    try:
        if not state.get("available_action_names", []):
            return False
        if state.get("investigation_loop_count", 0) > MAX_INVESTIGATION_LOOPS:
            return False
        return bool(state.get("investigation_recommendations", []))
    except Exception:  # noqa: BLE001
        logger.exception("loop-decision check failed; publishing")
        return False


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
    *,
    queue: EventQueue | None = None,
) -> AgentState:
    """Run the investigation pipeline procedurally.

    Mutates ``initial`` and returns it. Auth context is read from ``state``
    directly — populate ``state["org_id"]``, ``state["user_id"]``, etc. before
    calling (the standalone webapp does this via ``UserContext.as_state_fields()``).
    """
    state = initial

    _merge_state(state, await _run_stage(queue, "extract_alert", node_extract_alert, state))
    if state.get("is_noise"):
        if queue is not None:
            await queue.put(None)
        return state

    _merge_state(
        state,
        await _run_stage(queue, "resolve_integrations", node_resolve_integrations, state),
    )

    while True:
        _merge_state(state, await _run_stage(queue, "plan_actions", node_plan_actions, state))
        _merge_state(state, await _run_stage(queue, "investigate", node_investigate, state))
        _merge_state(state, await _run_stage(queue, "diagnose", node_diagnose_root_cause, state))
        if not should_continue_investigation(state):
            break

    _merge_state(state, await _run_stage(queue, "publish", node_publish_findings, state))

    if queue is not None:
        await queue.put(None)  # sentinel: stream complete
    return state


__all__ = ["EventQueue", "run_investigation_async", "should_continue_investigation"]
