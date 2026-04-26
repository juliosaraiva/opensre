"""Bridge OpenSRE's ``RegisteredTool`` registry into Claude Agent SDK MCP servers.

Phase 1 scaffolding: this module builds in-process MCP servers from the existing
tool registry but is **not yet wired into any runtime path**. It exists so that
Phase 4 (chat on ``ClaudeSDKClient``) and any future agentic investigation
surface can adopt the SDK without re-deriving tool schemas.

Two servers are produced — one per tool surface — so the chat and investigation
allow-lists stay isolated:

- ``opensre_chat``           — tools with surface ``"chat"``
- ``opensre_investigation``  — tools with surface ``"investigation"``

Per-call availability filtering (``RegisteredTool.is_available``) is expressed
through :func:`allowed_tools_for`, which produces an ``allowed_tools`` list to
pass into ``ClaudeAgentOptions``. Tools not currently available are simply
omitted from that list so the model cannot call them, even though they are
registered with the server.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool, create_sdk_mcp_server, tool

from app.tools.registered_tool import RegisteredTool
from app.tools.registry import get_registered_tools
from app.types.tools import ToolSurface

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

_SURFACE_SERVER_NAMES: dict[ToolSurface, str] = {
    "chat": "opensre_chat",
    "investigation": "opensre_investigation",
}


def mcp_tool_qualified_name(surface: ToolSurface, tool_name: str) -> str:
    """Return the SDK-side fully-qualified tool name (``mcp__<server>__<tool>``)."""
    server = _SURFACE_SERVER_NAMES[surface]
    return f"mcp__{server}__{tool_name}"


def _build_tool_handler(rt: RegisteredTool) -> ToolHandler:
    """Wrap a ``RegisteredTool.run`` into the async handler shape the SDK expects."""

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = rt.run(**args)
        except Exception as exc:  # noqa: BLE001 — surface tool errors to Claude
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "is_error": True,
            }
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        return {"content": [{"type": "text", "text": text}]}

    return _handler


def build_sdk_tool(rt: RegisteredTool) -> SdkMcpTool[Any]:
    """Wrap a single ``RegisteredTool`` as an SDK ``SdkMcpTool``.

    The handler closes over ``rt`` and forwards keyword arguments to
    ``rt.run``. Schemas are passed through verbatim — ``RegisteredTool`` already
    enforces JSON-Schema-shaped ``input_schema`` via ``ToolMetadata``.
    """
    handler = _build_tool_handler(rt)
    decorator = tool(rt.name, rt.description, rt.input_schema)
    return decorator(handler)


def build_mcp_server(surface: ToolSurface) -> McpSdkServerConfig:
    """Build the in-process MCP server for a given tool surface.

    All tools matching ``surface`` are registered, regardless of whether their
    integration is currently configured. Per-call gating happens via
    :func:`allowed_tools_for`, which is what the SDK enforces against the model.
    """
    if surface not in _SURFACE_SERVER_NAMES:
        valid = ", ".join(sorted(_SURFACE_SERVER_NAMES))
        raise ValueError(f"Unsupported tool surface '{surface}'. Expected one of: {valid}.")
    sdk_tools = [build_sdk_tool(rt) for rt in get_registered_tools(surface)]
    return create_sdk_mcp_server(name=_SURFACE_SERVER_NAMES[surface], tools=sdk_tools)


def build_mcp_servers(
    *surfaces: ToolSurface,
) -> dict[str, McpSdkServerConfig]:
    """Build a ``mcp_servers`` mapping for one or more surfaces.

    Returns a dict keyed by the SDK-facing server name (the same one referenced
    by ``allowed_tools=["mcp__<server>__<tool>"]``).
    """
    if not surfaces:
        surfaces = ("chat", "investigation")
    return {_SURFACE_SERVER_NAMES[s]: build_mcp_server(s) for s in surfaces}


def allowed_tools_for(
    surface: ToolSurface,
    resolved_integrations: dict[str, dict] | None = None,
) -> list[str]:
    """Return the ``allowed_tools`` list for a single ``query()`` / client call.

    Filters the registered tools for ``surface`` by ``is_available``, then maps
    each survivor to its ``mcp__<server>__<tool>`` qualified name.
    """
    sources = resolved_integrations or {}
    return [
        mcp_tool_qualified_name(surface, rt.name)
        for rt in get_registered_tools(surface)
        if rt.is_available(sources)
    ]
