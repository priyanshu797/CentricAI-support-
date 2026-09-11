"""Host-bound context; never accepted from model tool arguments."""
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustedContext:
    session_name: str
    document_id: str = ""
    user_id: str = ""
    allowed_writes: frozenset[str] = frozenset()


current_context: ContextVar[TrustedContext | None] = ContextVar("mcp_context", default=None)
