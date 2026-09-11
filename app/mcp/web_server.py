"""Centrix web MCP server backed by the shared DuckDuckGo search service."""

from app.mcp.server import build_server


def create_server():
    return build_server("centrix_web_mcp", ("search_web",))


# ── Class alias so client.py can import MCPWebServer by name ───────────────────
class MCPWebServer:
    """Thin wrapper so main.py / client.py can do isinstance() checks."""
    def __init__(self):
        self._server = create_server()

    @property
    def mcp_server(self):
        return self._server

    def wire(self, **_):
        """No-op: web search needs no injected services."""
