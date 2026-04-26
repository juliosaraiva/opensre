"""Tests for the procedural investigation driver.

The driver replaced the LangGraph ``StateGraph`` runner in Phase 7 — there is
no longer an alternative runtime to A/B against. These tests monkeypatch the
node functions on the driver module so they exercise the wiring (call order,
loop control, message append-semantics, stage streaming) without depending on
real LLM calls or live integrations.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.pipeline import driver as driver_mod
from app.pipeline.driver import run_investigation_async, should_continue_investigation
from app.pipeline.runners import _astream_procedural, run_investigation
from app.state import AgentState, make_initial_state

# ─────────────────────────────────────────────────────────────────────────────
# Stub nodes — record call order, bump loop counter, drive the loop control
# ─────────────────────────────────────────────────────────────────────────────


class _Recorder:
    """Track which nodes ran in which order."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def step(self, name: str) -> None:
        self.calls.append(name)


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_noise: bool = False,
    loops: int = 1,
    extra_extract: dict[str, Any] | None = None,
) -> _Recorder:
    """Patch every node function on the driver module with deterministic stubs.

    ``loops`` controls how many investigate→diagnose passes run before
    ``should_continue_investigation`` flips to False (recommendations exhausted).
    """
    recorder = _Recorder()
    state_holder = {"loops_remaining": loops}

    def _extract(state: AgentState) -> dict[str, Any]:
        recorder.step("extract_alert")
        update: dict[str, Any] = {"is_noise": is_noise}
        if extra_extract:
            update.update(extra_extract)
        return update

    def _resolve(state: AgentState) -> dict[str, Any]:
        recorder.step("resolve_integrations")
        return {"resolved_integrations": {"datadog": {"id": "dd-1"}}}

    def _plan(state: AgentState) -> dict[str, Any]:
        recorder.step("plan_actions")
        return {
            "planned_actions": ["a"],
            "available_action_names": ["a"],
        }

    def _investigate(state: AgentState) -> dict[str, Any]:
        recorder.step("investigate")
        return {"evidence": {"a": "ok"}}

    def _diagnose(state: AgentState) -> dict[str, Any]:
        recorder.step("diagnose")
        state_holder["loops_remaining"] -= 1
        recommendations = ["next"] if state_holder["loops_remaining"] > 0 else []
        return {
            "investigation_recommendations": recommendations,
            "investigation_loop_count": state.get("investigation_loop_count", 0) + 1,
            "validity_score": 0.9,
        }

    def _publish(state: AgentState) -> dict[str, Any]:
        recorder.step("publish")
        return {"report": "FINAL", "slack_message": "done"}

    monkeypatch.setattr(driver_mod, "node_extract_alert", _extract)
    monkeypatch.setattr(driver_mod, "node_resolve_integrations", _resolve)
    monkeypatch.setattr(driver_mod, "node_plan_actions", _plan)
    monkeypatch.setattr(driver_mod, "node_investigate", _investigate)
    monkeypatch.setattr(driver_mod, "node_diagnose_root_cause", _diagnose)
    monkeypatch.setattr(driver_mod, "node_publish_findings", _publish)
    return recorder


# ─────────────────────────────────────────────────────────────────────────────
# Driver behavior
# ─────────────────────────────────────────────────────────────────────────────


def test_driver_runs_full_pipeline_in_correct_order(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install_stubs(monkeypatch, loops=1)
    state = make_initial_state("alert", "pipeline", "high", raw_alert={"text": "boom"})

    final = asyncio.run(run_investigation_async(state))

    assert recorder.calls == [
        "extract_alert",
        "resolve_integrations",
        "plan_actions",
        "investigate",
        "diagnose",
        "publish",
    ]
    assert final["report"] == "FINAL"
    assert final["resolved_integrations"] == {"datadog": {"id": "dd-1"}}


def test_driver_short_circuits_on_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install_stubs(monkeypatch, is_noise=True)
    state = make_initial_state("alert", "pipeline", "high", raw_alert={"text": "ok"})

    final = asyncio.run(run_investigation_async(state))

    assert recorder.calls == ["extract_alert"]
    assert final["is_noise"] is True
    assert final.get("report", "") == ""  # publish stub never ran, so default stays


def test_driver_loops_until_recommendations_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install_stubs(monkeypatch, loops=3)
    state = make_initial_state("alert", "pipeline", "high", raw_alert={"text": "boom"})

    asyncio.run(run_investigation_async(state))

    assert recorder.calls.count("investigate") == 3
    assert recorder.calls.count("diagnose") == 3
    assert recorder.calls.count("plan_actions") == 3
    assert recorder.calls[-1] == "publish"


def test_driver_appends_messages_rather_than_replacing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stubs(monkeypatch, extra_extract={"messages": [{"role": "system", "content": "hi"}]})
    state = make_initial_state("alert", "pipeline", "high", raw_alert={"text": "boom"})
    state["messages"] = [{"role": "user", "content": "context"}]  # type: ignore[typeddict-item]

    final = asyncio.run(run_investigation_async(state))

    assert final["messages"] == [
        {"role": "user", "content": "context"},
        {"role": "system", "content": "hi"},
    ]


def test_run_investigation_dispatches_to_procedural_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_investigation`` is now a thin wrapper around the procedural driver."""
    _install_stubs(monkeypatch, loops=1)
    final = run_investigation("alert", "pipeline", "high", raw_alert={"text": "boom"})
    assert final["report"] == "FINAL"


# ─────────────────────────────────────────────────────────────────────────────
# Loop control
# ─────────────────────────────────────────────────────────────────────────────


def test_should_continue_investigation_loops_when_recommendations_present() -> None:
    state: AgentState = {  # type: ignore[typeddict-item]
        "investigation_recommendations": ["check logs"],
        "available_action_names": ["fetch_logs"],
        "investigation_loop_count": 0,
    }
    assert should_continue_investigation(state) is True


def test_should_continue_investigation_stops_when_no_actions_available() -> None:
    state: AgentState = {  # type: ignore[typeddict-item]
        "investigation_recommendations": ["check logs"],
        "available_action_names": [],
        "investigation_loop_count": 0,
    }
    assert should_continue_investigation(state) is False


def test_should_continue_investigation_stops_when_no_recommendations() -> None:
    state: AgentState = {  # type: ignore[typeddict-item]
        "investigation_recommendations": [],
        "available_action_names": ["fetch_logs"],
        "investigation_loop_count": 0,
    }
    assert should_continue_investigation(state) is False


# ─────────────────────────────────────────────────────────────────────────────
# Streaming
# ─────────────────────────────────────────────────────────────────────────────


def test_astream_procedural_emits_stage_markers_for_each_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stubs(monkeypatch, loops=1)
    state = make_initial_state("alert", "pipeline", "high", raw_alert={"text": "boom"})

    async def _collect() -> list[str]:
        names: list[str] = []
        async for event in _astream_procedural(state):
            names.append(f"{event.node_name}:{event.kind}")
        return names

    events = asyncio.run(_collect())

    expected_stages = [
        "extract_alert",
        "resolve_integrations",
        "plan_actions",
        "investigate",
        "diagnose",
        "publish",
    ]
    for stage in expected_stages:
        assert f"{stage}:on_chain_start" in events
        assert f"{stage}:on_chain_end" in events
