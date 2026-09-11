from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app import agent_runner


def schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
        },
    }


def tool_call(name, arguments, index):
    return {
        "id": f"call-{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class StubMCPClient:
    def __init__(self):
        self.calls = []

    def schemas(self, _context):
        return [schema("get_mood_history"), schema("search_knowledge"), schema("search_web")]

    def call_tool(self, name, arguments, context):
        self.calls.append((name, arguments, context))
        base = {
            "_mcp": {
                "server": {
                    "get_mood_history": "centrix_data_mcp",
                    "search_knowledge": "centrix_knowledge_mcp",
                    "search_web": "centrix_web_mcp",
                }[name],
                "tool": name,
                "status": "success",
                "latency_ms": 1.0,
                **({"provider": "DuckDuckGo", "results": 1} if name == "search_web" else {}),
            },
            "latency_ms": 1.0,
        }
        if name == "get_mood_history":
            return {**base, "summary": {"most_frequent": "stressed", "entry_count": 2}}
        if name == "search_knowledge":
            return {**base, "answer": "Document guidance", "sources": ["stress.pdf"], "chunks_retrieved": 2}
        return {
            **base,
            "results": [{"title": "Research", "url": "https://example.org/research", "snippet": "Recent evidence"}],
        }


class AgentMCPOrchestrationTests(unittest.TestCase):
    def test_combined_request_can_use_three_mcp_servers(self) -> None:
        client = StubMCPClient()
        plans = [
            {
                "content": "",
                "tool_calls": [
                    tool_call("get_mood_history", {"limit": 5}, 1),
                    tool_call("search_knowledge", {"query": "stress recommendations"}, 2),
                    tool_call("search_web", {"query": "2026 stress management research"}, 3),
                ],
            },
            {"content": "", "tool_calls": []},
        ]
        question = (
            "Check my recent mood, use my uploaded stress document, and compare it "
            "with the latest 2026 web research."
        )
        with (
            patch.object(agent_runner, "_plan_step", side_effect=[(plans[0], "groq"), (plans[1], "groq")]),
            patch.object(agent_runner, "_stream_synthesis", return_value=iter(["Comparison"])),
        ):
            runner = agent_runner.AgentRunner(client)
            response = "".join(runner.stream(question, "trusted", "doc-1"))

        self.assertIn("Comparison", response)
        self.assertEqual([row[0] for row in client.calls], [
            "get_mood_history", "search_knowledge", "search_web"
        ])
        self.assertEqual(runner.insights["mcp"]["used"], True)
        self.assertEqual(len(runner.insights["mcp"]["calls"]), 3)
        self.assertEqual(runner.insights["web_search"]["results"], 1)
        self.assertIn("https://example.org/research", response)

    def test_casual_chat_offers_and_calls_no_tools(self) -> None:
        client = StubMCPClient()
        with (
            patch.object(agent_runner, "_plan_step") as planner,
            patch.object(agent_runner, "_stream_synthesis", return_value=iter(["Hello!"])),
        ):
            runner = agent_runner.AgentRunner(client)
            self.assertEqual("".join(runner.stream("Hello", "trusted")), "Hello!")
        planner.assert_not_called()
        self.assertEqual(client.calls, [])
        self.assertFalse(runner.insights["mcp"]["used"])


if __name__ == "__main__":
    unittest.main()
