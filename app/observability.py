"""
app/observability.py  —  Langfuse v4 observability for CentrixSupport
══════════════════════════════════════════════════════════════════════════════
Langfuse 4.x uses OpenTelemetry-based context propagation.
There is no "trace object" to pass around.  Instead:

  1. Call  obs.start_trace(name, session_id, query, tags)  as a context
     manager at the top of a request.  Everything inside that `with` block
     automatically belongs to that trace.

  2. Use   obs.generation(name, model, input, output, usage_details, ...)
     and     obs.span(name, input, metadata, ...)
     as nested context managers for individual steps.

  3. Call  obs.flush()  at shutdown.

All methods are safe no-ops when Langfuse is disabled or keys are missing.

Quick start
───────────
  from app.observability import obs

  with obs.start_trace("search", session_id=sid, query=q, tags=["stream"]):
      obs.set_trace_output(response_text)   # call before exiting the with block

      with obs.span("emotion-classification", metadata={...}):
          pass

      with obs.generation("llm-call", model="gpt-oss-120b",
                           input=messages, usage={"input":10,"output":5}):
          pass
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)

# ── Read config ────────────────────────────────────────────────────────────────
_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() not in ("false", "0", "no")

# ── Init Langfuse client ───────────────────────────────────────────────────────
_lf: Any = None

if _ENABLED and _SECRET_KEY and _PUBLIC_KEY:
    try:
        from langfuse import Langfuse  # type: ignore[import]

        _lf = Langfuse(
            secret_key=_SECRET_KEY,
            public_key=_PUBLIC_KEY,
            host=_HOST,
        )
        logger.info(
            "Langfuse v4 observability enabled → %s  (key=%s…)",
            _HOST,
            _PUBLIC_KEY[:16],
        )
    except Exception as _exc:  # noqa: BLE001
        logger.warning("Langfuse init failed — tracing disabled: %s", _exc)
        _lf = None
else:
    if _ENABLED:
        logger.warning(
            "Langfuse tracing disabled: set LANGFUSE_SECRET_KEY and "
            "LANGFUSE_PUBLIC_KEY in .env"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _noop_cm() -> Any:
    """Return a context manager that does nothing."""
    return nullcontext()


class Observability:
    """
    Thin, safe wrapper around Langfuse v4.
    Every method returns a context manager.  When Langfuse is disabled
    every context manager is a nullcontext() — zero overhead.
    """

    @contextmanager
    def _safe_observation(self, **kwargs):
        cm = None
        try:
            if _lf is not None:
                cm = _lf.start_as_current_observation(**kwargs)
                cm.__enter__()
        except Exception:
            cm = None
        try:
            yield
        except BaseException as exc:
            if cm is not None:
                try:
                    # Record only the type, never an exception containing secrets.
                    _lf.update_current_span(level="ERROR", status_message=type(exc).__name__)
                except Exception:
                    pass
            raise
        finally:
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass

    @contextmanager
    def mcp_tool_call(self, metadata):
        with self.span("mcp-tool-call", metadata=dict(metadata)):
            try:
                yield
            finally:
                if _lf is not None:
                    try:
                        _lf.update_current_span(metadata=dict(metadata),
                            level="ERROR" if metadata.get("status") == "error" else "DEFAULT")
                    except Exception:
                        pass

    # ── Trace ──────────────────────────────────────────────────────────────────

    @contextmanager
    def start_trace(
        self,
        name: str,
        *,
        session_id: str = "",
        query: str = "",
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Generator[None, None, None]:
        """
        Top-level trace context manager.  Every observation started inside
        this block is automatically attached to this trace.

        Usage
        -----
        with obs.start_trace("search", session_id=sid, query=q, tags=["stream"]):
            obs.set_trace_output(final_answer)
        """
        if _lf is None:
            yield
            return
        from contextlib import ExitStack
        stack = ExitStack()
        try:
            from langfuse import propagate_attributes
            stack.enter_context(propagate_attributes(trace_name=name, session_id=session_id or None,
                                                      tags=tags or [], metadata=metadata or {}))
            stack.enter_context(self._safe_observation(name=name, as_type="span",
                                                       input=query[:500], metadata=metadata or {}))
        except Exception:
            logger.debug("Trace setup unavailable")
        try:
            yield
        finally:
            try:
                stack.close()
            except Exception:
                logger.debug("Trace teardown unavailable")

    def set_trace_output(self, output: str) -> None:
        """Update the current span/trace with the final output text."""
        if _lf is None:
            return
        try:
            _lf.update_current_span(output=output[:2000] if output else "")
        except Exception as exc:  # noqa: BLE001
            logger.debug("obs.set_trace_output error: %s", exc)

    # ── Generation span ────────────────────────────────────────────────────────

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        model: str = "",
        input: Any = None,
        output: str = "",
        usage: dict | None = None,
        metadata: dict | None = None,
        level: str = "DEFAULT",
        error: str | None = None,
    ) -> Generator[None, None, None]:
        """
        Record an LLM generation.

        usage dict keys: input, output, total  (token counts)
        """
        if _lf is None:
            yield
            return
        usage_details = None
        if usage:
            usage_details = {
                "input": int(usage.get("input", usage.get("prompt_tokens", 0))),
                "output": int(usage.get("output", usage.get("completion_tokens", 0))),
                "total": int(usage.get("total", 0)),
            }
        with self._safe_observation(
                name=name,
                as_type="generation",
                model=model or None,
                input=input,
                output=output[:4000] if output else None,
                usage_details=usage_details,
                metadata={**(metadata or {}), **({"error": error} if error else {})},
                level=level if not error else "ERROR",
        ):
            yield

    # ── Generic span ───────────────────────────────────────────────────────────

    @contextmanager
    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        output: Any = None,
        metadata: dict | None = None,
        level: str = "DEFAULT",
    ) -> Generator[None, None, None]:
        """Generic named span."""
        if _lf is None:
            yield
            return
        with self._safe_observation(
                name=name,
                as_type=as_type,
                input=input,
                output=output,
                metadata=metadata or {},
                level=level,
        ):
            yield

    # ── Convenience wrappers ───────────────────────────────────────────────────

    @contextmanager
    def retrieval_span(
        self,
        query: str,
        *,
        n_results: int = 0,
        top_score: float = 0.0,
        doc_names: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> Generator[None, None, None]:
        with self.span(
            "rag-retrieval",
            as_type="retriever",
            input=query[:300],
            metadata={
                "n_results": n_results,
                "top_score": round(top_score, 4),
                "doc_names": (doc_names or [])[:10],
                "latency_ms": round(latency_ms, 1),
            },
        ):
            yield

    @contextmanager
    def rerank_span(
        self,
        *,
        n_in: int = 0,
        n_out: int = 0,
        top_scores: list[float] | None = None,
        latency_ms: float = 0.0,
    ) -> Generator[None, None, None]:
        with self.span(
            "reranking",
            metadata={
                "chunks_in": n_in,
                "chunks_out": n_out,
                "top_rerank_scores": [round(s, 4) for s in (top_scores or [])[:5]],
                "latency_ms": round(latency_ms, 1),
            },
        ):
            yield

    @contextmanager
    def emotion_span(
        self,
        emotion: str,
        confidence: float,
        intensity: str,
        latency_ms: float = 0.0,
    ) -> Generator[None, None, None]:
        with self.span(
            "emotion-classification",
            metadata={
                "emotion": emotion,
                "confidence": round(confidence, 3),
                "intensity": intensity,
                "latency_ms": round(latency_ms, 1),
            },
        ):
            yield

    @contextmanager
    def groundedness_span(
        self,
        score: float,
        verdict: str,
        supported: int,
        unsupported: int,
        latency_ms: float = 0.0,
    ) -> Generator[None, None, None]:
        with self.span(
            "groundedness-check",
            as_type="evaluator",
            metadata={
                "score_pct": round(score * 100, 1),
                "verdict": verdict,
                "supported_claims": supported,
                "unsupported_claims": unsupported,
                "latency_ms": round(latency_ms, 1),
            },
        ):
            yield

    @contextmanager
    def cache_span(
        self,
        hit: bool,
        query: str,
        latency_ms: float = 0.0,
    ) -> Generator[None, None, None]:
        with self.span(
            "cache-lookup",
            metadata={
                "cache_hit": hit,
                "query": query[:200],
                "latency_ms": round(latency_ms, 1),
            },
        ):
            yield

    @contextmanager
    def fallback_span(
        self,
        from_provider: str,
        to_provider: str,
        reason: str = "",
    ) -> Generator[None, None, None]:
        with self.span(
            "provider-fallback",
            level="WARNING",
            metadata={
                "from": from_provider,
                "to": to_provider,
                "reason": reason[:300],
            },
        ):
            yield

    # ── MCP tool call span ────────────────────────────────────────────────────

    @contextmanager
    def mcp_span(
        self,
        server: str,
        tool: str,
        *,
        server_type: str = "",
        status: str = "success",
        latency_ms: float = 0.0,
        error: str | None = None,
        extra: dict | None = None,
    ) -> Generator[None, None, None]:
        """
        Record one MCP tool call in Langfuse.

        Emitted metadata shape:
        {
            "server":      "centrix_data_mcp",
            "tool":        "get_mood_history",
            "server_type": "data",
            "status":      "success" | "error",
            "latency_ms":  143,
            "error":       null | "...",
        }

        Usage
        -----
        with obs.mcp_span(server="centrix_web_mcp", tool="search_web",
                          server_type="web", status="success",
                          latency_ms=392):
            pass
        """
        metadata: dict = {
            "server":      server,
            "tool":        tool,
            "server_type": server_type,
            "status":      status,
            "latency_ms":  round(latency_ms, 1),
        }
        if error:
            metadata["error"] = error[:300]
        if extra:
            # merge safe non-secret extra keys (caller's responsibility)
            metadata.update({k: v for k, v in extra.items()
                             if isinstance(v, (str, int, float, bool))})
        level = "DEFAULT" if status == "success" else "WARNING"
        with self.span(
            f"mcp:{tool}",
            as_type="span",
            metadata=metadata,
            level=level,
        ):
            yield

    # ── Score ──────────────────────────────────────────────────────────────────

    def score(
        self,
        name: str,
        value: float,
        comment: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Attach a numeric evaluation score to the current trace."""
        if _lf is None:
            return
        try:
            _lf.create_score(
                name=name,
                value=value,
                trace_id=trace_id or _lf.get_current_trace_id(),
                comment=comment,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("obs.score error: %s", exc)

    # ── Flush ──────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        if _lf is None:
            return
        try:
            _lf.flush()
            logger.debug("Langfuse flush complete")
        except Exception as exc:  # noqa: BLE001
            logger.debug("obs.flush error: %s", exc)

    @property
    def enabled(self) -> bool:
        return _lf is not None


# ── Singleton ──────────────────────────────────────────────────────────────────
obs = Observability()
