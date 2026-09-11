from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import google.generativeai as genai

from app import llm_fallback

MESSAGES = [{"role": "user", "content": "hello"}]


def _groq_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


class LlmFallbackStreamingTests(unittest.TestCase):
    def test_groq_stream_uses_supported_arguments(self) -> None:
        calls: list[dict] = []
        chunk = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))],
        )

        def create(**kwargs):
            calls.append(kwargs)
            return iter([chunk])

        with patch.object(llm_fallback, "_get_groq", return_value=_groq_client(create)):
            chunks, provider = llm_fallback.llm_chat(MESSAGES, stream=True)

        self.assertEqual(provider, "groq")
        self.assertEqual("".join(chunks), "hello")
        self.assertTrue(calls[0]["stream"])
        self.assertNotIn("stream_options", calls[0])
        self.assertEqual(calls[0]["model"], llm_fallback.GROQ_MODEL)

    def test_stream_creation_error_falls_back_before_iteration(self) -> None:
        def failed_create(**_kwargs):
            raise TypeError(
                "Completions.create() got an unexpected keyword argument "
                "'stream_options'"
            )

        gemini_model = SimpleNamespace(
            generate_content=lambda *_args, **_kwargs: iter(
                [SimpleNamespace(text="fallback")]
            )
        )
        config_calls: list[dict] = []

        def generation_config(**kwargs):
            config_calls.append(kwargs)
            return kwargs

        with (
            patch.object(
                llm_fallback,
                "_get_groq",
                return_value=_groq_client(failed_create),
            ),
            patch.object(llm_fallback, "_init_gemini", return_value=True),
            patch.object(genai, "GenerativeModel", return_value=gemini_model),
            patch.object(
                genai.types,
                "GenerationConfig",
                side_effect=generation_config,
            ),
        ):
            chunks, provider = llm_fallback.llm_chat(MESSAGES, stream=True)

        self.assertEqual(provider, "gemini")
        self.assertEqual("".join(chunks), "fallback")
        self.assertNotIn("temperature", config_calls[0])
        self.assertEqual(config_calls[0]["max_output_tokens"], 2000)

    def test_skip_groq_uses_gemini_directly(self) -> None:
        gemini_model = SimpleNamespace(
            generate_content=lambda *_args, **_kwargs: iter(
                [SimpleNamespace(text="gemini only")]
            )
        )

        with (
            patch.object(llm_fallback, "_get_groq") as get_groq,
            patch.object(llm_fallback, "_init_gemini", return_value=True),
            patch.object(genai, "GenerativeModel", return_value=gemini_model),
        ):
            chunks, provider = llm_fallback.llm_chat(
                MESSAGES,
                stream=True,
                skip_groq=True,
            )

        self.assertEqual(provider, "gemini")
        self.assertEqual("".join(chunks), "gemini only")
        get_groq.assert_not_called()


if __name__ == "__main__":
    unittest.main()
