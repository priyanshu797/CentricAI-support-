"""Synchronous Flask adapter for real Model Context Protocol sessions.

The application keeps its existing business services in-process. Built-in
servers communicate with this client through the MCP SDK's bidirectional
memory transport, so discovery and execution still use MCP initialize,
tools/list, and tools/call messages. A configured Streamable HTTP endpoint
can be used instead without changing the agent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Callable
from urllib.parse import urlparse

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_connected_server_and_client_session

from app.mcp.context import TrustedContext, current_context
from app.observability import obs

logger = logging.getLogger(__name__)


@dataclass
class ServerSpec:
    """One MCP transport endpoint known to the client."""

    name: str
    category: str
    factory: Callable[[], Server] | None = None
    url: str = ""
    status: str = "disconnected"
    tool_count: int = 0
    discovery_ok: bool = False
    last_successful_connection: str | None = None
    discovery_latency_ms: float | None = None
    last_error_type: str | None = None
    tools: set[str] = field(default_factory=set)


class MCPClient:
    """Discover tools and execute them through MCP protocol sessions."""

    def __init__(self, server_specs: list[ServerSpec] | None = None) -> None:
        self._servers: dict[str, ServerSpec] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._routes: dict[str, str] = {}
        self._initialized = False
        self._lock = RLock()
        for spec in server_specs or _configured_servers():
            self.add_server(spec)

    def add_server(self, spec: ServerSpec) -> None:
        """Register another MCP endpoint before or after initialization."""
        if spec.name in self._servers:
            raise ValueError(f"Duplicate MCP server name: {spec.name}")
        self._servers[spec.name] = spec

    def initialize(self) -> dict[str, Any]:
        """Connect to every endpoint and discover tools with MCP tools/list."""
        with self._lock:
            schemas: dict[str, dict[str, Any]] = {}
            routes: dict[str, str] = {}
            for spec in self._servers.values():
                spec.tools.clear()
                started = time.perf_counter()
                try:
                    tools = anyio.run(self._list_tools, spec)
                    for tool in tools:
                        name = tool.name
                        if name in routes:
                            raise ValueError(f"Duplicate MCP tool name: {name}")
                        schema = {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": tool.description or "",
                                "parameters": dict(tool.inputSchema or {"type": "object"}),
                            },
                        }
                        schemas[name] = schema
                        routes[name] = spec.name
                        spec.tools.add(name)
                    spec.status = "connected"
                    spec.tool_count = len(tools)
                    spec.discovery_ok = True
                    spec.last_error_type = None
                    spec.last_successful_connection = datetime.now(timezone.utc).isoformat()
                except Exception as exc:  # noqa: BLE001 - MCP is optional infrastructure
                    spec.status = "disconnected"
                    spec.tool_count = 0
                    spec.discovery_ok = False
                    spec.last_error_type = type(exc).__name__
                    logger.warning("MCP discovery failed for %s (%s)", spec.name, type(exc).__name__)
                finally:
                    spec.discovery_latency_ms = round((time.perf_counter() - started) * 1000, 1)
            self._schemas = schemas
            self._routes = routes
            self._initialized = True
        return self.health()

    def schemas(self, context: TrustedContext) -> list[dict[str, Any]]:
        """Return schemas discovered from servers, filtered by trusted context."""
        if not self._initialized:
            self.initialize()
        document_tools = {
            "search_knowledge",
            "search_uploaded_document",
            "summarize_document",
        }
        with self._lock:
            return [
                schema
                for name, schema in self._schemas.items()
                if context.document_id or name not in document_tools
            ]

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: TrustedContext,
    ) -> dict[str, Any]:
        """Execute MCP tools/call and return its structured result."""
        started = time.perf_counter()
        with self._lock:
            server_name = self._routes.get(tool_name)
        metadata: dict[str, Any] = {
            "server": server_name or "unknown",
            "tool": tool_name,
            "status": "error",
        }
        with obs.mcp_tool_call(metadata):
            if server_name is None or server_name not in self._servers:
                metadata.update(error_type="UnknownTool", latency_ms=0.0)
                return self._failure(metadata)

            spec = self._servers[server_name]
            token = current_context.set(context)
            try:
                result = anyio.run(self._call_tool, spec, tool_name, dict(arguments))
                payload = _structured_result(result)
                success = not bool(result.isError) and not bool(payload.get("error"))
                metadata["status"] = "success" if success else "error"
                if not success:
                    metadata["error_type"] = payload.get("error_type", "MCPToolError")
                if tool_name == "search_web":
                    # Reflect whichever provider actually ran (SerpApi or DuckDuckGo)
                    metadata["provider"] = payload.get("provider", "DuckDuckGo")
                    metadata["results"] = len(payload.get("results") or [])
                spec.status = "connected"
                spec.last_error_type = None
            except Exception as exc:  # noqa: BLE001 - degrade instead of breaking chat
                payload = {
                    "error": "Tool unavailable or invalid request",
                    "error_type": type(exc).__name__,
                }
                metadata["error_type"] = type(exc).__name__
                spec.status = "disconnected"
                spec.last_error_type = type(exc).__name__
                logger.warning(
                    "MCP call failed for %s/%s (%s)",
                    server_name,
                    tool_name,
                    type(exc).__name__,
                )
            finally:
                current_context.reset(token)
                metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)

            payload["_mcp"] = dict(metadata)
            payload.setdefault("latency_ms", metadata["latency_ms"])
            return payload

    def health(self) -> dict[str, Any]:
        with self._lock:
            servers = {
                name: {
                    "status": spec.status,
                    "category": spec.category,
                    "tool_discovery": "success" if spec.discovery_ok else "failure",
                    "tool_count": spec.tool_count,
                    "last_successful_connection": spec.last_successful_connection,
                    "discovery_latency_ms": spec.discovery_latency_ms,
                    "last_error_type": spec.last_error_type,
                }
                for name, spec in self._servers.items()
            }
        all_ok = bool(servers) and all(row["status"] == "connected" for row in servers.values())
        return {
            "status": "ok" if all_ok else "degraded",
            "initialized": self._initialized,
            "servers": servers,
            "total_tools": len(self._schemas),
        }

    async def _list_tools(self, spec: ServerSpec) -> list[Any]:
        async with _session(spec) as session:
            discovered = []
            cursor = None
            while True:
                result = await session.list_tools(cursor) if cursor else await session.list_tools()
                discovered.extend(result.tools)
                cursor = result.nextCursor
                if not cursor:
                    return discovered

    async def _call_tool(
        self,
        spec: ServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        async with _session(spec) as session:
            return await session.call_tool(tool_name, arguments)

    @staticmethod
    def _failure(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "error": "Tool unavailable or invalid request",
            "error_type": metadata.get("error_type", "MCPError"),
            "_mcp": dict(metadata),
            "latency_ms": metadata.get("latency_ms", 0.0),
        }


@asynccontextmanager
async def _session(spec: ServerSpec):
    """Open and initialize the configured MCP transport."""
    timeout = float(os.getenv("MCP_SERVER_TIMEOUT_SECONDS", "20"))
    if spec.factory is not None:
        async with create_connected_server_and_client_session(
            spec.factory(),
            read_timeout_seconds=timedelta(seconds=timeout),
        ) as session:
            yield session
        return

    if not _safe_mcp_url(spec.url):
        raise ValueError("Invalid MCP server URL")
    with anyio.fail_after(timeout):
        async with streamable_http_client(spec.url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session


def _safe_mcp_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username


def _structured_result(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return dict(structured)
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {"results": parsed}
            except (TypeError, ValueError):
                return {"result": str(text)}
    return {}


def _configured_servers() -> list[ServerSpec]:
    """Build server specs from environment, defaulting to built-in MCP servers."""
    from app.mcp.data_server import create_server as data_server
    from app.mcp.knowledge_server import create_server as knowledge_server
    from app.mcp.web_server import create_server as web_server

    definitions = (
        ("centrix_data_mcp", "data", "MCP_DATA_SERVER", data_server),
        ("centrix_knowledge_mcp", "knowledge", "MCP_KNOWLEDGE_SERVER", knowledge_server),
        ("centrix_web_mcp", "web", "MCP_WEB_SERVER", web_server),
    )
    specs = []
    for name, category, env_name, factory in definitions:
        configured = os.getenv(env_name, "builtin").strip()
        if configured.lower() in {"", "builtin", "memory", "inmemory"}:
            specs.append(ServerSpec(name=name, category=category, factory=factory))
        else:
            specs.append(ServerSpec(name=name, category=category, url=configured))
    return specs


_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _client
    if _client is None:
        _client = MCPClient()
    return _client
