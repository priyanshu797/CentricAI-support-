from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import agent_runner


def _groq_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


class AgentSynthesisTests(unittest.TestCase):
    def test_synthesis_replaces_tool_selection_system_prompt(self) -> None:
        messages = [
            {"role": "system", "content": agent_runner.AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": "Help me prepare for exams"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "analyze_emotion",
                "content": json.dumps(
                    {
                        "emotion": "stressed",
                        "confidence": 0.9,
                        "intensity": "High",
                    }
                ),
            },
        ]

        result = agent_runner._synthesis_messages(messages)

        self.assertEqual(result[0]["content"], agent_runner.SYNTHESIS_SYSTEM_PROMPT)
        self.assertNotIn(
            agent_runner.AGENT_SYSTEM_PROMPT, [m["content"] for m in result]
        )
        self.assertIn("Emotion detected: stressed", result[-1]["content"])

    def test_groq_synthesis_explicitly_disables_tools(self) -> None:
        calls: list[dict] = []
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Take a break."))]
        )

        def create(**kwargs):
            calls.append(kwargs)
            return iter([chunk])

        messages = [
            {"role": "system", "content": agent_runner.AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": "I feel stressed"},
        ]
        with patch.object(
            agent_runner,
            "_get_groq",
            return_value=_groq_client(create),
        ):
            text = "".join(agent_runner._stream_synthesis(messages, "groq"))

        self.assertEqual(text, "Take a break.")
        self.assertEqual(calls[0]["tool_choice"], "none")
        self.assertEqual(
            calls[0]["messages"][0]["content"],
            agent_runner.SYNTHESIS_SYSTEM_PROMPT,
        )

    def test_failed_groq_stream_skips_groq_during_fallback(self) -> None:
        def failed_stream(**_kwargs):
            def chunks():
                raise RuntimeError("Tool choice is none, but model called a tool")
                yield  # pragma: no cover

            return chunks()

        fallback_calls: list[dict] = []

        def fallback(_messages, **kwargs):
            fallback_calls.append(kwargs)
            return iter(["fallback response"]), "gemini"

        messages = [{"role": "user", "content": "hello"}]
        with (
            patch.object(
                agent_runner,
                "_get_groq",
                return_value=_groq_client(failed_stream),
            ),
            patch.object(agent_runner, "llm_chat", side_effect=fallback),
        ):
            text = "".join(agent_runner._stream_synthesis(messages, "groq"))

        self.assertEqual(text, "fallback response")
        self.assertTrue(fallback_calls[0]["skip_groq"])


if __name__ == "__main__":
    unittest.main()
