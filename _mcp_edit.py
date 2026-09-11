from pathlib import Path

p = Path('app/content_retrieval.py')
s = p.read_text(encoding='utf-8')
a = s.index('    @staticmethod\n    def web_search(')
b = s.index('\nclass QueryClassifier', a)
# Keep the section marker before QueryClassifier; the old search is now a compatibility wrapper.
s = s[:a] + '''    @staticmethod
    def web_search(query: str) -> str:
        """Compatibility adapter to the shared DuckDuckGo implementation."""
        from app.web_search import search_web
        try:
            return "\\n".join(f"{r['title']} ({r['url']}): {r['snippet']}" for r in search_web(query))
        except Exception:
            logger.warning("DuckDuckGo unavailable")
            return ""

''' + s[b:]
s = s.replace('        conv_history: list[dict] | None = None,\n    ) -> tuple', '        conv_history: list[dict] | None = None,\n        knowledge_only: bool = False,\n    ) -> tuple')
a = s.index('    def run(', s.index('class Agent:'))
head, body = s[:a], s[a:]
body = body.replace('        # ── 0. Conversational', '        cache_scope = self.cache_scope + (":private-only-v1" if knowledge_only else "")\n        # ── 0. Conversational', 1)
body = body.replace('scope=self.cache_scope', 'scope=cache_scope')
body = body.replace('        classification = QueryClassifier.classify(eq)', '        classification = QueryClassifier.classify(eq)\n        if knowledge_only:\n            classification["intent"] = "document_qa"')
body = body.replace('if self._is_vague(', 'if not knowledge_only and self._is_vague(')
body = body.replace('            # 5c. Off-topic', '            if knowledge_only:\n                return "The document does not provide enough relevant evidence to answer this question.", [], "docs", None, rewritten_query\n\n            # 5c. Off-topic')
body = body.replace('        # 6. No index', '        if knowledge_only:\n            return "Document knowledge is unavailable.", [], "docs", None, rewritten_query\n\n        # 6. No index')
# Existing contextmanager calls were never entered, so they emitted no spans.
body = body.replace('                _obs_inst.cache_span(True, eq, _cache_ms)', '                with _obs_inst.cache_span(True, eq, _cache_ms):\n                    pass')
body = body.replace('            _obs_inst.cache_span(False, eq, _cache_ms)', '            with _obs_inst.cache_span(False, eq, _cache_ms):\n                pass')
body = body.replace('                _obs_inst.retrieval_span(', '                with _obs_inst.retrieval_span(')
body = body.replace('                    doc_names=doc_names,\n                )', '                    doc_names=doc_names,\n                ):\n                    pass')
body = body.replace('                    _obs_inst.rerank_span(', '                    with _obs_inst.rerank_span(')
body = body.replace('                        latency_ms=retrieval_ms,\n                    )', '                        latency_ms=retrieval_ms,\n                    ):\n                        pass')
p.write_text(head + body, encoding='utf-8')

p = Path('app/agent_tools.py')
s = p.read_text(encoding='utf-8')
a = s.index('def web_search(query: str)')
b = s.index('# ─────', a)
s = s[:a] + '''def web_search(query: str) -> dict[str, Any]:
    """Legacy alias; all search business logic lives in app.web_search."""
    try:
        results = search_web(query)
        return {"results": results, "found": bool(results)}
    except Exception:
        return {"error": "DuckDuckGo unavailable", "results": [], "found": False}


''' + s[b:]
p.write_text(s, encoding='utf-8')

p = Path('app/observability.py')
s = p.read_text(encoding='utf-8')
# Prevent generator context managers from yielding twice on application errors.
s = s.replace('        try:\n            with _lf.start_as_current_observation(', '        with self._safe_observation(')
s = s.replace('            ):\n                yield\n        except Exception as exc:  # noqa: BLE001\n            logger.debug("obs.generation error: %s", exc)\n            yield', '        ):\n            yield')
s = s.replace('            ):\n                yield\n        except Exception as exc:  # noqa: BLE001\n            logger.debug("obs.span error: %s", exc)\n            yield', '        ):\n            yield')
a = s.index('        try:\n            from langfuse import propagate_attributes')
b = s.index('    def set_trace_output', a)
s = s[:a] + '''        from contextlib import ExitStack
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

''' + s[b:]
a = s.index('    # ── Trace')
s = s[:a] + '''    @contextmanager
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

''' + s[a:]
p.write_text(s, encoding='utf-8')
