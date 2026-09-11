"""
app/mcp/protocol.py
─────────────────────────────────────────────────────────────────────────────
MCP message types — the data structures that flow between the MCPClient
and the MCP servers.

Follows the Model Context Protocol specification pattern:
  • MCPToolSchema      — tool definition exposed to the agent/LLM
  • MCPToolCall        — a request to execute a specific tool
  • MCPToolResult      — structured result returned from tool execution
  • MCPListToolsResult — response to a "list available tools" request
  • MCPServerInfo      — metadata about a connected server
  • MCPError           — structured error payload

All types are plain dataclasses with no external dependencies so they
can be imported anywhere without side effects.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Server categories ─────────────────────────────────────────────────────────

class MCPServerType(str, Enum):
    DATA      = "data"       # MongoDB / mood / task / memory
    KNOWLEDGE = "knowledge"  # RAG / document / vector / cache
    WEB       = "web"        # DuckDuckGo / real-time web search


# ── Tool schema (what the agent sees) ────────────────────────────────────────

@dataclass
class MCPToolSchema:
    """
    Describes one tool that an MCP server exposes.
    Mirrors the JSON schema format expected by Groq/OpenAI tool_calls.
    """
    name: str
    description: str
    parameters: dict[str, Any]        # JSON-Schema object for parameters
    server_name: str = ""             # injected by MCPClient during discovery
    server_type: MCPServerType = MCPServerType.DATA

    def to_groq_schema(self) -> dict[str, Any]:
        """Convert to the dict format Groq expects in the `tools` array."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ── Tool call (client → server) ───────────────────────────────────────────────

@dataclass
class MCPToolCall:
    """
    A request from the MCPClient to a specific server to execute a tool.
    """
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""           # forwarded from the LLM tool_call id
    session_name: str = ""      # trusted app-level context (never from LLM)
    document_id: str = ""       # trusted app-level context (never from LLM)


# ── Tool result (server → client) ─────────────────────────────────────────────

@dataclass
class MCPToolResult:
    """
    Structured result returned by an MCP server after executing a tool call.
    """
    tool_name: str
    server_name: str
    server_type: MCPServerType
    data: dict[str, Any] = field(default_factory=dict)   # actual payload
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for JSON injection into agent messages)."""
        d = dict(self.data)
        if self.error:
            d["error"] = self.error
        d["_mcp_server"]  = self.server_name
        d["_mcp_tool"]    = self.tool_name
        d["_latency_ms"]  = round(self.latency_ms, 1)
        return d


# ── List-tools response ───────────────────────────────────────────────────────

@dataclass
class MCPListToolsResult:
    """
    Response to the discover_tools() call.
    Contains all tools a server advertises.
    """
    server_name: str
    server_type: MCPServerType
    tools: list[MCPToolSchema] = field(default_factory=list)
    latency_ms: float = 0.0


# ── Server info ───────────────────────────────────────────────────────────────

@dataclass
class MCPServerInfo:
    """Runtime metadata for a connected MCP server."""
    name: str
    server_type: MCPServerType
    connected: bool = False
    tool_count: int = 0
    last_error: str | None = None
    last_latency_ms: float = 0.0


# ── Structured error ──────────────────────────────────────────────────────────

@dataclass
class MCPError(Exception):
    """
    Raised when an MCP call fails unrecoverably.
    Always caught by MCPClient — never surfaces to the end user.
    """
    server_name: str
    tool_name: str
    message: str
    recoverable: bool = True

    def __str__(self) -> str:
        return f"[MCP:{self.server_name}/{self.tool_name}] {self.message}"


# ── Web search result item ────────────────────────────────────────────────────

@dataclass
class WebSearchResult:
    """
    One result returned by the web_search tool.
    Preserves title + URL + snippet as required by the spec.
    """
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}
