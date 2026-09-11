"""
llm_fallback.py
───────────────
Centralised LLM call helper with automatic provider fallback.

Primary  : Groq  – openai/gpt-oss-120b
Fallback : Google Gemini – gemini-3.5-flash-lite

Every call site in the project should use `llm_complete()` (single-turn)
or `llm_chat()` (multi-turn messages list) instead of calling the SDK
clients directly.  If Groq raises any error the helper retries the same
request against Gemini and logs a WARNING so the issue stays visible.

Public API
──────────
llm_complete(prompt, *, system, temperature, max_tokens, trace)
    → (text: str, prompt_tokens: int, completion_tokens: int, provider: str)

llm_chat(messages, *, temperature, max_tokens, stream, trace, span_name)
    → non-streaming : (text: str, prompt_tokens: int, completion_tokens: int, provider: str)
    → streaming     : (generator[str], provider: str)
        where the generator yields raw text chunks

Both functions raise RuntimeError only when *both* providers fail.

Observability
─────────────
Pass the Langfuse trace object (from obs.trace()) as `trace=` to get a
Generation span recorded for every LLM call, including token usage,
latency, model name, and provider.  Streaming calls record the generation
after the generator is drained by the caller (post-stream).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

# ── Model identifiers ──────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ── API keys (read from environment — set in .env) ─────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Lazy-initialised SDK clients ───────────────────────────────────────────────
_groq_client: Any = None
_gemini_available: bool = False


# ── Observability (imported lazily to avoid circular imports) ──────────────────
def _obs() -> Any:
    """Return the global Observability singleton (lazy import)."""
    try:
        from app.observability import obs  # type: ignore[import]

        return obs
    except Exception:  # noqa: BLE001
        return None


# ── Groq client ────────────────────────────────────────────────────────────────
def _get_groq() -> Any:
    """Return a cached Groq client, or None if the SDK is not installed."""
    global _groq_client  # noqa: PLW0603
    if _groq_client is not None:
        return _groq_client
    if not GROQ_API_KEY:
        logger.warning(
            "llm_fallback: GROQ_API_KEY is not set – primary provider unavailable"
        )
        return None
    try:
        from groq import Groq

        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.debug("llm_fallback: Groq client initialised")
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm_fallback: Groq client init failed – %s", exc)
        _groq_client = None
    return _groq_client


# ── Gemini client ──────────────────────────────────────────────────────────────
def _init_gemini() -> bool:
    """Configure the google-generativeai SDK once. Returns True on success."""
    global _gemini_available  # noqa: PLW0603
    if _gemini_available:
        return True
    if not GEMINI_API_KEY:
        logger.warning(
            "llm_fallback: GEMINI_API_KEY is not set – fallback will be unavailable"
        )
        return False
    try:
        import google.generativeai as genai  # type: ignore[import]

        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_available = True
        logger.debug("llm_fallback: Gemini SDK configured")
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm_fallback: Gemini SDK init failed – %s", exc)
    return _gemini_available


# ── Gemini message adapter ─────────────────────────────────────────────────────
def _groq_messages_to_gemini(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """
    Convert an OpenAI-style messages list to the Gemini `contents` format.
    Returns (system_instruction, contents).
    """
    system_instruction: str | None = None
    contents: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            system_instruction = text
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text}]})
    return system_instruction, contents


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def llm_complete(
    prompt: str,
    *,
    system: str = (
        "You are a knowledgeable and helpful AI assistant. "
        "Answer questions clearly and accurately."
    ),
    temperature: float = 0.4,
    max_tokens: int = 700,
    trace: Any = None,
    span_name: str = "llm-complete",
) -> tuple[str, int, int, str]:
    """
    Single-turn LLM completion.

    Returns
    -------
    (text, prompt_tokens, completion_tokens, provider)
        provider is "groq" or "gemini"
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text, pt, ct, provider = llm_chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        trace=trace,
        span_name=span_name,
    )
    return text, pt, ct, provider  # type: ignore[return-value]


def llm_chat(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    stream: bool = False,
    trace: Any = None,
    span_name: str = "llm-call",
    skip_groq: bool = False,
) -> tuple[str, int, int, str] | tuple[Generator[str, None, None], str]:
    """
    Multi-turn LLM call with automatic Groq → Gemini fallback.

    Parameters
    ----------
    messages   : OpenAI-style list of {"role": ..., "content": ...} dicts
    temperature: sampling temperature
    max_tokens : maximum tokens to generate
    stream     : if True, returns (chunk_generator, provider) instead of
                 (full_text, prompt_tokens, completion_tokens, provider)
    trace      : Langfuse trace object (from obs.trace()) — optional
    span_name  : label used for the Langfuse Generation span
    skip_groq  : go directly to Gemini; used after a Groq attempt has already
                 failed while a stream was being consumed

    Raises
    ------
    RuntimeError  when both providers fail.
    """
    # ── 1. Try Groq ────────────────────────────────────────────────────────────
    groq = None if skip_groq else _get_groq()
    if groq is not None:
        try:
            if stream:
                # Streaming: instrument via a wrapping generator so we can
                # record the generation *after* the stream is fully drained.
                return _groq_stream_traced(
                    groq, messages, temperature, max_tokens, trace, span_name
                )
            t0 = time.perf_counter()
            resp = groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0

            # ── Langfuse generation span ───────────────────────────────────
            if trace is not None:
                _obs_instance = _obs()
                if _obs_instance:
                    _obs_instance.llm_generation(
                        trace,
                        "groq",
                        GROQ_MODEL,
                        messages,
                        text,
                        pt,
                        ct,
                        latency_ms,
                        name=span_name,
                    )
            return text, pt, ct, "groq"

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "llm_fallback: Groq call failed (%s) – switching to Gemini", exc
            )
            # Record the fallback event in Langfuse
            if trace is not None:
                _obs_instance = _obs()
                if _obs_instance:
                    _obs_instance.fallback_event(
                        trace, "groq", "gemini", str(exc)[:300]
                    )

    # ── 2. Fallback to Gemini ──────────────────────────────────────────────────
    if not _init_gemini():
        raise RuntimeError(
            "Both Groq and Gemini are unavailable. "
            "Check API keys and network connectivity."
        )
    try:
        import google.generativeai as genai  # type: ignore[import]

        system_instruction, contents = _groq_messages_to_gemini(messages)
        model_kwargs: dict[str, Any] = {"model_name": GEMINI_MODEL}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        gemini_model = genai.GenerativeModel(**model_kwargs)
        # Gemini 3.x is tuned for its default sampler and the current API asks
        # callers to omit the legacy temperature/top-p/top-k parameters.
        gen_config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
        if not GEMINI_MODEL.startswith("gemini-3"):
            gen_config_kwargs["temperature"] = temperature
        gen_config = genai.types.GenerationConfig(  # type: ignore[attr-defined]
            **gen_config_kwargs
        )

        if stream:
            return _gemini_stream_traced(
                gemini_model, contents, gen_config, trace, span_name
            )

        t0 = time.perf_counter()
        response = gemini_model.generate_content(contents, generation_config=gen_config)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        text = response.text or ""
        meta = getattr(response, "usage_metadata", None)
        pt = getattr(meta, "prompt_token_count", 0) or 0
        ct = getattr(meta, "candidates_token_count", 0) or 0

        logger.info("llm_fallback: Gemini served the response (pt=%d ct=%d)", pt, ct)

        # ── Langfuse generation span ───────────────────────────────────────
        if trace is not None:
            _obs_instance = _obs()
            if _obs_instance:
                _obs_instance.llm_generation(
                    trace,
                    "gemini",
                    GEMINI_MODEL,
                    messages,
                    text,
                    pt,
                    ct,
                    latency_ms,
                    name=span_name,
                )
        return text, pt, ct, "gemini"

    except Exception as exc:  # noqa: BLE001
        if trace is not None:
            _obs_instance = _obs()
            if _obs_instance:
                _obs_instance.llm_generation(
                    trace,
                    "gemini",
                    GEMINI_MODEL,
                    messages,
                    "",
                    0,
                    0,
                    0.0,
                    name=span_name,
                    error=str(exc)[:300],
                )
        raise RuntimeError(
            f"Both Groq and Gemini failed. Last Gemini error: {exc}"
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# Streaming helpers  (traced variants wrap the generators)
# ══════════════════════════════════════════════════════════════════════════════


def _groq_stream_traced(
    groq_client: Any,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    trace: Any,
    span_name: str,
) -> tuple[Generator[str, None, None], str]:
    """
    Return a (chunk_generator, 'groq') tuple.
    After the generator is exhausted the accumulated text + token stats
    are sent to Langfuse as a generation span.
    """

    # Start the request eagerly. This is deliberately outside the generator:
    # model/argument/auth failures must be raised while llm_chat() is still in
    # its Groq try/except so it can switch to Gemini.
    t0 = time.perf_counter()
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    def _gen() -> Generator[str, None, None]:
        full_text = ""
        pt = ct = 0
        try:
            for chunk in response:
                # Accumulate token counts from usage field when available
                usage = getattr(chunk, "usage", None)
                if usage:
                    pt = getattr(usage, "prompt_tokens", pt) or pt
                    ct = getattr(usage, "completion_tokens", ct) or ct
                delta = (chunk.choices[0].delta.content if chunk.choices else "") or ""
                if delta:
                    full_text += delta
                    yield delta
        finally:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            if trace is not None:
                _obs_instance = _obs()
                if _obs_instance:
                    _obs_instance.llm_generation(
                        trace,
                        "groq",
                        GROQ_MODEL,
                        messages,
                        full_text,
                        pt,
                        ct,
                        latency_ms,
                        name=span_name,
                    )

    return _gen(), "groq"


def _gemini_stream_traced(
    gemini_model: Any,
    contents: list[dict],
    gen_config: Any,
    trace: Any,
    span_name: str,
) -> tuple[Generator[str, None, None], str]:
    """
    Return a (chunk_generator, 'gemini') tuple.
    After the generator is exhausted the accumulated text is sent to Langfuse.
    """

    # Eager creation also ensures Gemini setup failures are converted into the
    # public "both providers failed" RuntimeError by llm_chat().
    t0 = time.perf_counter()
    response = gemini_model.generate_content(
        contents,
        generation_config=gen_config,
        stream=True,
    )

    def _gen() -> Generator[str, None, None]:
        full_text = ""
        try:
            for chunk in response:
                text = getattr(chunk, "text", "") or ""
                if text:
                    full_text += text
                    yield text
        finally:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            if trace is not None:
                _obs_instance = _obs()
                if _obs_instance:
                    _obs_instance.llm_generation(
                        trace,
                        "gemini",
                        GEMINI_MODEL,
                        contents,
                        full_text,
                        0,
                        0,
                        latency_ms,
                        name=span_name,
                    )

    return _gen(), "gemini"


# ── Non-traced stream helpers (kept for backwards compat) ──────────────────────


def _groq_stream(
    groq_client: Any,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> tuple[Generator[str, None, None], str]:
    return _groq_stream_traced(
        groq_client, messages, temperature, max_tokens, None, "llm-stream"
    )


def _gemini_stream(
    gemini_model: Any,
    contents: list[dict],
    gen_config: Any,
) -> tuple[Generator[str, None, None], str]:
    return _gemini_stream_traced(gemini_model, contents, gen_config, None, "llm-stream")
