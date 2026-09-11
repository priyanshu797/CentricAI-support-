"""Centrix knowledge MCP server backed by the existing RAG pipeline."""

from app.mcp.server import build_server


def create_server():
    return build_server(
        "centrix_knowledge_mcp",
        ("search_knowledge", "search_uploaded_document", "summarize_document"),
    )


# ── Class alias so client.py can import MCPKnowledgeServer by name ─────────────
class MCPKnowledgeServer:
    """Thin wrapper so main.py / client.py can do isinstance() checks."""
    def __init__(self):
        self._server = create_server()

    @property
    def mcp_server(self):
        return self._server

    def wire(self, **_):
        """No-op: services are accessed through agent_tools module globals."""
