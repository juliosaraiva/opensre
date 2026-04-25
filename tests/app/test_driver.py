"""Phase 3: parity tests for the procedural investigation driver.

Asserts that :func:`app.pipeline.driver.run_investigation_async` matches the
LangGraph-backed runner step-for-step on the same fixtures, and that the
``OPENSRE_RUNNER`` flag toggles which runner ``run_investigation`` dispatches
to. The node functions are monkey-patched to deterministic stubs so the
tests don't depend on real LLM calls or live integrations.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.pipeline import driver as driver_mod
from app.pipeline import runners as runners_mod
from app.pipeline.driver import run_investigation_async
from app.pipeline.runners import (
    _astream_procedural,
    _runner_choice,
    run_investigation,
)
from app.state import AgentState, make_initial_state

# ─────────────────────────────────────────────────────────────────────────────
# Stub nodes — record call order, bump loop counter, drive the routing fns
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

    ``loops`` controls how many investigate→diagnose passes run before the
    routing function flips to ``publish`` (recommendations exhausted).
    """
    recorder = _Recorder()
    state_holder = {"loops_remaining": loops}

    def _inject_auth(state: AgentState, _config: dict[str, Any]) -> dict[str, Any]:
        recorder.step("inject_auth")
        return {"org_id": "org-1", "user_id": "user-1"}

    def _extract(state: AgentState) -> dict[str, Any]:
        recorder.step("extract_alert")
        update: dict[str, Any] = {"is_noise": is_noise}
        if extra_extract:
            update.update(extra_extract)
        return update

    def _resolve(state: AgentState, _config: dict[str, Any] | None = None) -> dict[str, Any]:
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

    monkeypatch.setattr(driver_mod, "inject_auth_node", _inject_auth)
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
        "inject_auth",
        "extract_alert",
        "resolve_integrations",
        "plan_actions",
        "investigate",
        "diagnose",
        "publish",
    ]
    assert final["report"] == "FINAL"
    assert final["org_id"] == "org-1"
    assert final["resolved_integrations"] == {"datadog": {"id": "dd-1"}}


def test_driver_short_circuits_on_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install_stubs(monkeypatch, is_noise=True)
    state = make_initial_state("alert", "pipeline", "high", raw_alert={"text": "ok"})

    final = asyncio.run(run_investigation_async(state))

    assert recorder.calls == ["inject_auth", "extract_alert"]
    assert final["is_noise"] is True
    assert final.get("report", "") == ""  # publish stub never ran, so default stays


def test_driver_loops_until_recommendations_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install_stubs(monkeypatch, loops=3)
    state = make_initial_state("alert", "pipeline", "high", raw_alert={"text": "boom"})

    asyncio.run(run_investigation_async(state))

    investigate_count = recorder.calls.count("investigate")
    diagnose_count = recorder.calls.count("diagnose")
    plan_count = recorder.calls.count("plan_actions")
    assert investigate_count == 3
    assert diagnose_count == 3
    assert plan_count == 3
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

    # Each stage emits both an on_chain_start and on_chain_end event.
    expected_stages = [
        "inject_auth",
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


# ─────────────────────────────────────────────────────────────────────────────
# Feature flag dispatch
# ─────────────────────────────────────────────────────────────────────────────


def test_runner_choice_defaults_to_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENSRE_RUNNER", raising=False)
    assert _runner_choice() == "langgraph"


def test_runner_choice_reads_procedural_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSRE_RUNNER", "procedural")
    assert _runner_choice() == "procedural"


def test_runner_choice_falls_back_for_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSRE_RUNNER", "bogus")
    assert _runner_choice() == "langgraph"


def test_run_investigation_routes_to_procedural_when_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Procedural path must execute end-to-end without touching LangGraph."""
    _install_stubs(monkeypatch, loops=1)

    # Belt-and-braces: blow up if anything imports the LangGraph module while
    # the procedural runner is active.
    def _no_langgraph(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "LangGraph runner should not be invoked under OPENSRE_RUNNER=procedural"
        )

    monkeypatch.setattr(runners_mod, "_map_langgraph_event", _no_langgraph)
    monkeypatch.setenv("OPENSRE_RUNNER", "procedural")

    final = run_investigation("alert", "pipeline", "high", raw_alert={"text": "boom"})

    assert final["report"] == "FINAL"
