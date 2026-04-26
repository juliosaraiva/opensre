"""Factories for Claude Agent SDK ``query()`` calls and ``ClaudeSDKClient`` sessions.

Phase 1 scaffolding: this module centralises construction of
:class:`ClaudeAgentOptions` so callers in later phases (chat session,
structured-output planner, diagnosis) do not each duplicate model selection,
permission mode, MCP-server wiring, and allowed-tool plumbing.

Two factories are exposed:

- :func:`build_chat_options`         — chat surface, tool-using
- :func:`build_structured_options`   — single forced tool for ``with_structured_output`` parity

Both pull the model from the existing OpenSRE ``LLMSettings`` so we do not
introduce a parallel configuration surface. Nothing here is wired into the
runtime yet — Phases 2 and 4 will adopt it.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, McpSdkServerConfig, PermissionMode

from app.tools.sdk_bridge import (
    allowed_tools_for,
    build_mcp_server,
    mcp_tool_qualified_name,
)
from app.types.tools import ToolSurface

_DEFAULT_PERMISSION_MODE: PermissionMode = "bypassPermissions"


def _resolve_default_model() -> str | None:
    """Best-effort resolution of the configured Anthropic model.

    Falls back to ``None`` (SDK default) if config import fails, so that this
    helper stays importable in test environments without full app config.
    """
    try:
        from app.config import LLMSettings  # noqa: PLC0415 — lazy to avoid import cycles
    except Exception:  # noqa: BLE001
        return None
    try:
        settings = LLMSettings()
        model = getattr(settings, "anthropic_reasoning_model", None) or getattr(
            settings, "anthropic_model", None
        )
        return str(model) if model else None
    except Exception:  # noqa: BLE001
        return None


def build_chat_options(
    *,
    system_prompt: str,
    resolved_integrations: dict[str, dict] | None = None,
    model: str | None = None,
    permission_mode: PermissionMode = _DEFAULT_PERMISSION_MODE,
    extra_disallowed_tools: list[str] | None = None,
) -> ClaudeAgentOptions:
    """Build options for the chat surface.

    Registers the ``opensre_chat`` MCP server and limits ``allowed_tools`` to
    those whose ``is_available(resolved_integrations)`` returns True.
    """
    server = build_mcp_server("chat")
    return ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model or _resolve_default_model(),
        mcp_servers={"opensre_chat": server},
        allowed_tools=allowed_tools_for("chat", resolved_integrations),
        disallowed_tools=list(extra_disallowed_tools or []),
        permission_mode=permission_mode,
    )


def build_structured_options(
    *,
    system_prompt: str,
    forced_tool: tuple[str, str, dict[str, Any]],
    model: str | None = None,
    permission_mode: PermissionMode = _DEFAULT_PERMISSION_MODE,
) -> tuple[ClaudeAgentOptions, str]:
    """Build options that force the model to call a single tool.

    ``forced_tool`` is ``(name, description, input_schema)`` — typically derived
    from a Pydantic model's ``model_json_schema()``. Returns the options plus
    the qualified tool name so callers can match on it when reading back the
    ``ToolUseBlock``.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: PLC0415

    name, description, input_schema = forced_tool
    server_name = "opensre_structured"

    async def _capture(_args: dict[str, Any]) -> dict[str, Any]:
        # The model's invocation is what we care about; the response is
        # discarded. Returning empty content keeps the SDK happy.
        return {"content": [{"type": "text", "text": "ok"}]}

    sdk_tool = tool(name, description, input_schema)(_capture)
    server = create_sdk_mcp_server(name=server_name, tools=[sdk_tool])
    qualified = f"mcp__{server_name}__{name}"

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model or _resolve_default_model(),
        mcp_servers={server_name: server},
        allowed_tools=[qualified],
        permission_mode=permission_mode,
        max_turns=1,
    )
    return options, qualified


def qualified_tool_name(surface: ToolSurface, tool_name: str) -> str:
    """Re-export of :func:`app.tools.sdk_bridge.mcp_tool_qualified_name` for callers."""
    return mcp_tool_qualified_name(surface, tool_name)


__all__ = [
    "ClaudeAgentOptions",
    "McpSdkServerConfig",
    "build_chat_options",
    "build_structured_options",
    "qualified_tool_name",
]
