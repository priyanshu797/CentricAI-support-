"""Centrix data MCP server backed by the existing service wrappers."""

from app.mcp.server import build_server


def create_server():
    return build_server(
        "centrix_data_mcp",
        (
            "get_conversation_history",
            "get_tasks",
            "create_task",
            "get_mood_history",
            "log_mood",
            "create_self_care_plan",
            "start_breathing_exercise",
        ),
    )


# ── Class alias so client.py can import MCPDataServer by name ──────────────────
class MCPDataServer:
    """Thin wrapper so main.py / client.py can do isinstance() checks."""
    def __init__(self):
        self._server = create_server()

    @property
    def mcp_server(self):
        return self._server

    def wire(self, **_):
        """No-op: services are accessed through agent_tools module globals."""
