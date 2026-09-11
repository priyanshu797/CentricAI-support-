"""
app/mcp/server_base.py
─────────────────────────────────────────────────────────────────────────────
Base class for all MCP servers in CentrixSupport.

Each concrete server subclasses MCPServer and:
  1. Calls super().__init__(name, server_type) in its constructor.
  2. Registers tools via  self._register(schema, handler_fn).
  3. Optionally overrides  start() / stop()  for connection management.

The base class provides:
  • Tool registry (name → schema + callable)
  • discover_tools() — returns MCPListToolsResult
  • call_tool(MCPToolCall) → MCPToolResult
  • Security: parameter validation, no shell execution, no arbitrary DB
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from app.mcp.protocol import (
    MCPListToolsResult,
    MCPServerInfo,
    MCPServerType,
    MCPToolCall,
    MCPToolResult,
    MCPToolSchema,
)

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Abstract base class for all CentrixSupport MCP servers.

    Subclasses register their tools in __init__ via _register(),
    then the MCPClient calls discover_tools() and call_tool() at runtime.
    """

    def __init__(self, name: str, server_type: MCPServerType) -> None:
        self._name        = name
        self._server_type = server_type
        self._connected   = False
        self._last_error: str | None = None

        # Registry: tool_name → (schema, callable)
        self._registry: dict[str, tuple[MCPToolSchema, Callable[..., dict[str, Any]]]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def _register(
        self,
        schema: MCPToolSchema,
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        """Register one tool with its JSON schema and Python handler."""
        schema.server_name = self._name
        schema.server_type = self._server_type
        self._registry[schema.name] = (schema, handler)
        logger.debug("[MCP:%s] registered tool: %s", self._name, schema.name)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Called by MCPClient during initialisation.
        Subclasses can override to perform connection setup.
        Returns True on success.
        """
        self._connected = True
        logger.info("[MCP:%s] server started (%d tools)", self._name, len(self._registry))
        return True

    def stop(self) -> None:
        """Called by MCPClient during shutdown."""
        self._connected = False
        logger.info("[MCP:%s] server stopped", self._name)

    # ── Tool discovery ────────────────────────────────────────────────────────

    def discover_tools(self) -> MCPListToolsResult:
        """
        Return all tools this server advertises.
        Called once by MCPClient on startup (and on reconnect).
        """
        t0 = time.perf_counter()
        schemas = [schema for schema, _ in self._registry.values()]
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return MCPListToolsResult(
            server_name=self._name,
            server_type=self._server_type,
            tools=schemas,
            latency_ms=latency_ms,
        )

    # ── Tool execution ────────────────────────────────────────────────────────

    def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """
        Execute a tool call and return a structured MCPToolResult.

        Security rules enforced here (belt-and-suspenders alongside agent prompt):
          • Only registered tools can be called.
          • Arguments are validated against expected parameter names.
          • session_name / document_id are always taken from the trusted
            MCPToolCall fields — never from the LLM-supplied arguments.
        """
        t0 = time.perf_counter()

        if call.tool_name not in self._registry:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return MCPToolResult(
                tool_name=call.tool_name,
                server_name=self._name,
                server_type=self._server_type,
                data={},
                success=False,
                error=f"Tool '{call.tool_name}' not found on server '{self._name}'",
                latency_ms=latency_ms,
            )

        schema, handler = self._registry[call.tool_name]

        # ── Parameter validation (no extra keys sneaking in) ──────────────────
        allowed = set(schema.parameters.get("properties", {}).keys())
        cleaned_args = {k: v for k, v in call.arguments.items() if k in allowed}

        # Inject trusted context — overrides whatever the LLM supplied
        if "session_name" in allowed and call.session_name:
            cleaned_args["session_name"] = call.session_name
        if "document_id" in allowed and call.document_id:
            cleaned_args["document_id"] = call.document_id

        try:
            data = handler(**cleaned_args)
            if not isinstance(data, dict):
                data = {"result": str(data)}
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return MCPToolResult(
                tool_name=call.tool_name,
                server_name=self._name,
                server_type=self._server_type,
                data=data,
                success=not bool(data.get("error")),
                error=data.get("error"),
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.error("[MCP:%s] %s raised: %s", self._name, call.tool_name, exc)
            return MCPToolResult(
                tool_name=call.tool_name,
                server_name=self._name,
                server_type=self._server_type,
                data={},
                success=False,
                error=str(exc),
                latency_ms=latency_ms,
            )

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def info(self) -> MCPServerInfo:
        return MCPServerInfo(
            name=self._name,
            server_type=self._server_type,
            connected=self._connected,
            tool_count=len(self._registry),
            last_error=self._last_error,
        )
