"""Phase 1 scaffolding tests for the Claude Agent SDK bridge.

Covers ``app.tools.sdk_bridge`` and ``app.pipeline.stream_adapter`` so the
scaffolding stays correct as we adopt the SDK in subsequent phases. No SDK
network calls are made — tests exercise the wrapping/mapping layer only.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.pipeline.stream_adapter import map_sdk_message, stage_event
from app.tools.registered_tool import RegisteredTool
from app.tools.registry import get_registered_tools
from app.tools.sdk_bridge import (
    allowed_tools_for,
    build_mcp_server,
    build_mcp_servers,
    build_sdk_tool,
    mcp_tool_qualified_name,
)

_FAKE_SOURCE = "storage"  # Any value in app.types.evidence.EvidenceSource Literal works.

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_fake_tool(
    *,
    name: str = "fake_tool",
    runner: Any = None,
    available: bool = True,
    surfaces: tuple[str, ...] = ("chat",),
) -> RegisteredTool:
    """Construct a minimal RegisteredTool for unit testing the bridge."""

    def _default_run(value: str = "x") -> dict[str, Any]:
        return {"echo": value}

    return RegisteredTool(
        name=name,
        description="fake tool for bridge tests",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": [],
        },
        source=_FAKE_SOURCE,  # type: ignore[arg-type]
        run=runner or _default_run,
        surfaces=surfaces,  # type: ignore[arg-type]
        is_available=lambda _sources: available,
    )


def _run_handler(sdk_tool: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Execute the async handler attached to an SdkMcpTool synchronously."""
    return asyncio.run(sdk_tool.handler(args))


# ─────────────────────────────────────────────────────────────────────────────
# build_sdk_tool — single-tool bridging
# ─────────────────────────────────────────────────────────────────────────────


def test_build_sdk_tool_carries_name_description_and_schema() -> None:
    rt = _make_fake_tool(name="echo_tool")
    sdk_tool = build_sdk_tool(rt)
    assert sdk_tool.name == "echo_tool"
    assert sdk_tool.description == rt.description
    assert sdk_tool.input_schema == rt.input_schema


def test_build_sdk_tool_invokes_underlying_run_and_serialises_dict_result() -> None:
    rt = _make_fake_tool()
    sdk_tool = build_sdk_tool(rt)
    response = _run_handler(sdk_tool, {"value": "hello"})
    assert response == {"content": [{"type": "text", "text": json.dumps({"echo": "hello"})}]}


def test_build_sdk_tool_passes_string_results_through_unchanged() -> None:
    rt = _make_fake_tool(runner=lambda value="x": f"plain text {value}")
    sdk_tool = build_sdk_tool(rt)
    response = _run_handler(sdk_tool, {"value": "y"})
    assert response == {"content": [{"type": "text", "text": "plain text y"}]}


def test_build_sdk_tool_surfaces_exception_as_is_error() -> None:
    def _boom(**_kwargs: Any) -> Any:
        raise RuntimeError("nope")

    rt = _make_fake_tool(runner=_boom)
    sdk_tool = build_sdk_tool(rt)
    response = _run_handler(sdk_tool, {})
    assert response["is_error"] is True
    assert "RuntimeError: nope" in response["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Server construction
# ─────────────────────────────────────────────────────────────────────────────


def test_build_mcp_server_uses_surface_specific_name() -> None:
    server = build_mcp_server("chat")
    # McpSdkServerConfig is a TypedDict-shaped dict
    assert server["type"] == "sdk"
    assert server["name"] == "opensre_chat"


def test_build_mcp_server_rejects_unknown_surface() -> None:
    with pytest.raises(ValueError):
        build_mcp_server("oops")  # type: ignore[arg-type]


def test_build_mcp_servers_default_returns_both_surfaces() -> None:
    servers = build_mcp_servers()
    assert set(servers) == {"opensre_chat", "opensre_investigation"}


def test_build_mcp_servers_subset_only_returns_requested() -> None:
    servers = build_mcp_servers("chat")
    assert set(servers) == {"opensre_chat"}


# ─────────────────────────────────────────────────────────────────────────────
# allowed_tools_for / qualified naming
# ─────────────────────────────────────────────────────────────────────────────


def test_mcp_tool_qualified_name_uses_double_underscore_separator() -> None:
    assert mcp_tool_qualified_name("chat", "foo") == "mcp__opensre_chat__foo"
    assert mcp_tool_qualified_name("investigation", "bar") == "mcp__opensre_investigation__bar"


def test_allowed_tools_for_returns_qualified_names_only_for_available() -> None:
    chat_tools = allowed_tools_for("chat", resolved_integrations={})
    # Every entry must follow the mcp__<server>__<tool> shape and reference
    # the chat server.
    assert all(t.startswith("mcp__opensre_chat__") for t in chat_tools)
    # And every one must exist in the chat surface registry.
    chat_names = {rt.name for rt in get_registered_tools("chat")}
    derived_names = {t.removeprefix("mcp__opensre_chat__") for t in chat_tools}
    assert derived_names.issubset(chat_names)


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip: every registered tool must build cleanly
# ─────────────────────────────────────────────────────────────────────────────


def test_every_registered_tool_round_trips_through_bridge() -> None:
    """No tool in the registry should fail SDK wrapping.

    This is the whole-registry contract that protects later phases — if any
    tool's metadata isn't SDK-compatible we want to know now, not at runtime.
    """
    failures: list[str] = []
    for rt in get_registered_tools():
        try:
            sdk_tool = build_sdk_tool(rt)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{rt.name}: {type(exc).__name__}: {exc}")
            continue
        if sdk_tool.name != rt.name:
            failures.append(f"{rt.name}: name mismatch ({sdk_tool.name})")
    assert not failures, "Tools failed SDK round-trip:\n" + "\n".join(failures)


# ─────────────────────────────────────────────────────────────────────────────
# Stream adapter
# ─────────────────────────────────────────────────────────────────────────────


def test_stream_adapter_maps_system_init_to_metadata_event() -> None:
    from claude_agent_sdk import SystemMessage  # noqa: PLC0415

    msg = SystemMessage(subtype="init", data={"model": "claude-opus-4"})
    [event] = map_sdk_message(msg, node_name="bootstrap")
    assert event.event_type == "metadata"
    assert event.kind == "on_chain_start"
    assert event.node_name == "bootstrap"
    assert event.data["subtype"] == "init"
    assert event.data["model"] == "claude-opus-4"


def test_stream_adapter_maps_assistant_text_to_chat_model_stream() -> None:
    from claude_agent_sdk import AssistantMessage, TextBlock  # noqa: PLC0415

    msg = AssistantMessage(
        content=[TextBlock(text="hello")],
        model="claude-opus-4",
        parent_tool_use_id=None,
        error=None,
        usage=None,
        message_id="m-1",
        stop_reason=None,
        session_id="sess-1",
        uuid="u-1",
    )
    [event] = map_sdk_message(msg, node_name="chat")
    assert event.event_type == "events"
    assert event.kind == "on_chat_model_stream"
    assert event.data["text"] == "hello"
    assert event.run_id == "sess-1"


def test_stream_adapter_maps_assistant_tool_use_to_tool_start() -> None:
    from claude_agent_sdk import AssistantMessage, ToolUseBlock  # noqa: PLC0415

    msg = AssistantMessage(
        content=[ToolUseBlock(id="t-1", name="search", input={"q": "errors"})],
        model="claude-opus-4",
        parent_tool_use_id=None,
        error=None,
        usage=None,
        message_id="m-2",
        stop_reason=None,
        session_id="sess-1",
        uuid="u-2",
    )
    [event] = map_sdk_message(msg, node_name="investigate")
    assert event.kind == "on_tool_start"
    assert event.data == {"tool_use_id": "t-1", "name": "search", "input": {"q": "errors"}}


def test_stream_adapter_maps_user_tool_result_to_tool_end() -> None:
    from claude_agent_sdk import ToolResultBlock, UserMessage  # noqa: PLC0415

    msg = UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id="t-1",
                content=[{"type": "text", "text": "results"}],
                is_error=False,
            )
        ],
        uuid="u-3",
        parent_tool_use_id=None,
        tool_use_result=None,
    )
    [event] = map_sdk_message(msg)
    assert event.kind == "on_tool_end"
    assert event.data["tool_use_id"] == "t-1"
    assert event.data["content"] == ["results"]
    assert event.data["is_error"] is False


def test_stream_adapter_maps_result_message_to_end_event() -> None:
    from claude_agent_sdk import ResultMessage  # noqa: PLC0415

    msg = ResultMessage(
        subtype="success",
        duration_ms=42,
        duration_api_ms=20,
        is_error=False,
        num_turns=1,
        session_id="sess-1",
        stop_reason="end_turn",
        total_cost_usd=0.0,
        usage={},
        result="done",
        structured_output=None,
        model_usage=None,
        permission_denials=None,
        errors=None,
        uuid="u-4",
    )
    [event] = map_sdk_message(msg, node_name="diagnose")
    assert event.event_type == "end"
    assert event.kind == "on_chain_end"
    assert event.data["result"] == "done"
    assert event.run_id == "sess-1"


def test_stage_event_produces_synthetic_node_marker() -> None:
    event = stage_event("plan_actions", "on_chain_start", {"loop": 1})
    assert event.event_type == "events"
    assert event.kind == "on_chain_start"
    assert event.node_name == "plan_actions"
    assert event.data == {"loop": 1}
