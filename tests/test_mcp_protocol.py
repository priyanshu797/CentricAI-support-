from __future__ import annotations

import unittest
from unittest.mock import patch

from app import agent_tools
from app.mcp.client import MCPClient
from app.mcp.context import TrustedContext


class TaskStore:
    def __init__(self) -> None:
        self.sessions: list[str] = []

    def list_tasks(self, session_name, status=None):
        self.sessions.append(session_name)
        return [{"id": "task-1", "title": "Revise", "status": status or "pending"}]

    def add(self, session_name, title, due_date=None, priority="medium"):
        self.sessions.append(session_name)
        return {
            "id": "task-2",
            "title": title,
            "due_date": due_date,
            "priority": priority,
            "status": "pending",
        }


class MoodStore:
    def summary(self, session_name, limit=10):
        return {"entry_count": 1, "most_frequent": "calm", "entries": []}

    def log(self, **_kwargs):
        return "mood-1"


class ConversationStore:
    def load(self, session_name):
        return [{"role": "user", "content": "hello"}]


class RagAgent:
    def run(self, query, metrics=None, knowledge_only=False):
        metrics.cache_hit = False
        metrics.num_docs_retrieved = 2
        metrics.original_similarity_scores = [0.88]
        self.context = (query, knowledge_only)
        return "Document answer", ["guide.pdf"], "docs", None, query


class MCPProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = TaskStore()
        self.rag = RagAgent()
        agent_tools.init_tool_services(
            mood_store=MoodStore(),
            task_store=self.tasks,
            rag_agent_factory=lambda document_id: self.rag if document_id == "doc-1" else None,
            detect_emotion_fn=lambda text: ("neutral", 0.9),
            llm_chat_fn=None,
            conversation_store=ConversationStore(),
        )
        self.client = MCPClient()
        self.health = self.client.initialize()

    def test_protocol_discovery_returns_dynamic_public_schemas(self) -> None:
        self.assertEqual(self.health["status"], "ok")
        self.assertEqual(self.health["total_tools"], 11)
        names = {
            schema["function"]["name"]
            for schema in self.client.schemas(TrustedContext("trusted-session", "doc-1"))
        }
        self.assertEqual(
            names,
            {
                "get_conversation_history",
                "get_tasks",
                "create_task",
                "get_mood_history",
                "log_mood",
                "create_self_care_plan",
                "start_breathing_exercise",
                "search_knowledge",
                "search_uploaded_document",
                "summarize_document",
                "search_web",
            },
        )
        get_tasks = next(
            row for row in self.client.schemas(TrustedContext("trusted-session"))
            if row["function"]["name"] == "get_tasks"
        )
        self.assertNotIn("session_name", get_tasks["function"]["parameters"]["properties"])

    def test_trusted_context_is_injected_and_impersonation_is_rejected(self) -> None:
        context = TrustedContext("trusted-session")
        result = self.client.call_tool("get_tasks", {"status": "pending"}, context)
        self.assertEqual(result["tasks"][0]["title"], "Revise")
        self.assertEqual(self.tasks.sessions, ["trusted-session"])
        self.assertEqual(result["_mcp"]["status"], "success")

        rejected = self.client.call_tool(
            "get_tasks",
            {"status": "pending", "session_name": "another-user"},
            context,
        )
        self.assertIn("error", rejected)
        self.assertEqual(self.tasks.sessions, ["trusted-session"])

    def test_write_requires_host_authority(self) -> None:
        denied = self.client.call_tool(
            "create_task",
            {"title": "Revise chapter 2", "priority": "high"},
            TrustedContext("trusted-session"),
        )
        self.assertEqual(denied["_mcp"]["status"], "error")

        allowed = self.client.call_tool(
            "create_task",
            {"title": "Revise chapter 2", "priority": "high"},
            TrustedContext("trusted-session", allowed_writes=frozenset({"create_task"})),
        )
        self.assertTrue(allowed["success"])
        self.assertEqual(allowed["task"]["title"], "Revise chapter 2")

    def test_knowledge_and_web_results_are_structured(self) -> None:
        knowledge = self.client.call_tool(
            "search_knowledge",
            {"query": "stress management"},
            TrustedContext("trusted-session", "doc-1"),
        )
        self.assertEqual(knowledge["answer"], "Document answer")
        self.assertEqual(knowledge["chunks_retrieved"], 2)
        self.assertEqual(knowledge["top_similarity"], 0.88)

        rows = [{"title": "Study", "url": "https://example.org/study", "snippet": "Evidence"}]
        with patch.dict(agent_tools.TOOL_FUNCTIONS, {"search_web": lambda query, num_results=5: rows}):
            web_client = MCPClient()
            web_client.initialize()
            web = web_client.call_tool(
                "search_web",
                {"query": "recent stress research", "num_results": 5},
                TrustedContext("trusted-session"),
            )
        self.assertEqual(web["results"], rows)
        self.assertEqual(web["_mcp"]["provider"], "DuckDuckGo")
        self.assertEqual(web["_mcp"]["results"], 1)


if __name__ == "__main__":
    unittest.main()
