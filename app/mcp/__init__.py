"""
app/mcp/__init__.py
─────────────────────────────────────────────────────────────────────────────
Model Context Protocol (MCP) package for CentrixSupport.

Exports
───────
  MCPClient         — the orchestrator class
  get_mcp_client()  — returns the app-wide singleton (or None before init)
  set_mcp_client()  — called once by main.py after wiring is complete

Architecture
────────────
  Flask (main.py)
       │ initialises
       ▼
  MCPClient (client.py)          ← discovers tools, routes calls, telemetry
       │
   ┌───┼──────────────┐
   ▼   ▼              ▼
 Data Knowledge      Web
 Server  Server    Server
   │       │          │
 MongoDB  RAG     DuckDuckGo
 Tasks    Docs    (web_search.py)
 Mood     Redis
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from app.mcp.client import MCPClient

# ── Module-level singleton ────────────────────────────────────────────────────
# Starts as None; set to a fully initialised MCPClient by main.py at startup.
_mcp_client: MCPClient | None = None


def get_mcp_client() -> MCPClient | None:
    """
    Return the initialised MCPClient singleton.
    Returns None if main.py has not yet called set_mcp_client() — this is
    handled gracefully by AgentRunner via _NullMCPClient.
    """
    return _mcp_client


def set_mcp_client(client: MCPClient) -> None:
    """
    Register the fully wired MCPClient as the app-wide singleton.
    Called exactly once from main.py after all services are initialised.
    """
    global _mcp_client
    _mcp_client = client


__all__ = ["MCPClient", "get_mcp_client", "set_mcp_client"]
