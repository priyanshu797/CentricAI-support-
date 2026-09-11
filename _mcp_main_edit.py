from pathlib import Path
p = Path('main.py')
s = p.read_text(encoding='utf-8')
s = s.replace('from app.observability import obs', '''from app.observability import obs
from app.mcp.client import get_mcp_client
from app.request_analysis import analyze_request
from app.request_context import trusted_context, trusted_session_name, register_document, require_document, owner_id''')
s = s.replace('app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_key_change_in_production")', '''import secrets
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")''')
s = s.replace('        llm_chat_fn=_llm_chat if LLM_FALLBACK_AVAILABLE else None,', '        llm_chat_fn=_llm_chat if LLM_FALLBACK_AVAILABLE else None,\n        conversation_store=_conversation_store,')
a = s.index('\ndef _used_uploaded_documents')
s = s[:a] + '''
_mcp_client = get_mcp_client()
try:
    _mcp_client.initialize()
except Exception as exc:
    logger.warning("MCP startup unavailable (%s)", type(exc).__name__)


@app.route("/health/mcp", methods=["GET"])
def mcp_health():
    # Explicitly internal. No URLs, credentials, prompts or arguments returned.
    import hmac
    token = os.getenv("MCP_HEALTH_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    if not token or not hmac.compare_digest(supplied, "Bearer " + token):
        return jsonify({"error": "Not found"}), 404
    health = _mcp_client.initialize()
    return jsonify(health), 200 if health["status"] == "ok" else 503

''' + s[a:]
s = s.replace('''        document_id = _indexing_service.submit(
            saved_files, [item["original_name"] for item in file_info]
        )''', '''        document_id = _indexing_service.submit(
            saved_files, [item["original_name"] for item in file_info]
        )
        register_document(document_id, [os.path.basename(path) for path in saved_files])''')
s = s.replace('def document_status(document_id: str):', 'def document_status(document_id: str):\n    require_document(document_id)')
# All existing data endpoints share the same trusted namespace as MCP.
s = s.replace('session_name = request.args.get("session_name", "default_session")', 'session_name = trusted_session_name(request.args.get("session_name", "default_session"))')
s = s.replace('session_name = data.get("session_name", "default_session")', 'session_name = trusted_session_name(data.get("session_name", "default_session"))')
s = s.replace('session_name: str = data.get("session_name", "default_session")', 'session_name: str = trusted_session_name(data.get("session_name", "default_session"))')
s = s.replace('document_id: str = data.get("document_id", "")', 'document_id: str = require_document(data.get("document_id", ""))')
s = s.replace('document_id = request.args.get("document_id", "").strip()', 'document_id = require_document(request.args.get("document_id", "").strip())\n    request_analysis = analyze_request(question)\n    trusted_user = owner_id()')
# Replace only the legacy /search dispatch branches with the common agent runner;
# preserve route payload, crisis handling, stores and metrics services.
a = s.index('            # Branch A — agentic RAG')
b = s.index('\n    except Exception as exc:', a)
s = s[:a] + '''            runner = AgentRunner(_mcp_client)
            response = "".join(runner.stream(question, session_name, document_id, conv_history,
                request_analysis=analyze_request(question), user_id=owner_id()))
            insights = runner.insights
            insights["emotion"] = {"label": emotion, "confidence": confidence}
            conv_history.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": response}])
            with obs.span("MongoDB", metadata={"operation": "save_conversation"}):
                _save_conversation(session_name, conv_history)
            obs.set_trace_output(response)
            _save_agent_metrics(question, session_name, lang, emotion, insights)
            return jsonify({"success": True, "response": response, "sources": runner.sources,
                "emotion_detected": emotion, "emotion_emoji": emotion_emoji,
                "emotion_confidence": round(confidence, 2), "emotion_intensity": intensity,
                "emotion_suggestions": suggestions, "language": lang,
                "model_used": insights["llm"]["model"], "rag_used": insights["rag"]["used"],
                "tool_used": "agent", "groundedness": insights["rag"],
                "response_insights": insights, "time": round(time.time() - t0, 2)})
''' + s[b:]
s = s.replace('                runner = AgentRunner()', '                runner = AgentRunner(_mcp_client)')
s = s.replace('                _tools_emitted: set[str] = set()\n', '')
s = s.replace('                    conv_history=conv_history,\n                ):', '                    conv_history=conv_history,\n                    request_analysis=request_analysis,\n                    user_id=trusted_user,\n                ):')
a = s.index('                    # Emit a progress frame')
b = s.index('                    if chunk:', a)
s = s[:a] + s[b:]
s = s.replace('                insights = runner.insights', '                insights = runner.insights\n                insights["emotion"] = {"label": emotion, "confidence": conf}')
s = s.replace('            _save_conversation(session_name, conv_history)\n\n            # ── RAG', '            with obs.span("MongoDB", metadata={"operation": "save_conversation"}):\n                _save_conversation(session_name, conv_history)\n\n            # ── RAG')
a = s.index('            # ── Agent tool spans in Langfuse')
b = s.index('        yield _sse({"type": "done", "sources": sources})', a)
s = s[:a] + '        sources = runner.sources\n\n' + s[b:]
# Common metrics recorder for non-streaming agent requests, using measured values.
a = s.index('# ── Main search endpoint')
s = s[:a] + '''def _save_agent_metrics(question, session_name, lang, emotion, insights):
    if not METRICS_AVAILABLE or _metrics_store is None:
        return
    try:
        rag, llm = insights["rag"], insights["llm"]
        metrics = RagMetrics(query=question, session_name=session_name, language=lang,
            emotion_detected=emotion, model_name=llm.get("model") or "",
            tool_used="docs" if rag["used"] else "agent", cache_hit=insights["cache"]["hit"],
            num_docs_retrieved=rag.get("chunks_retrieved") or 0,
            total_latency_ms=insights["total_latency_ms"], llm_latency_ms=llm.get("latency_ms") or 0)
        metrics.finalise()
        _metrics_store.save(metrics)
    except Exception as exc:
        logger.warning("Agent metrics unavailable (%s)", type(exc).__name__)


''' + s[a:]
p.write_text(s, encoding='utf-8')

p = Path('static/js/script8.js')
s = p.read_text(encoding='utf-8')
s = s.replace('  const val    = d.validation || {};', '  const val    = d.validation || {};\n  const mcp = d.mcp || {calls: []};\n  const web = d.web_search || {};')
s = s.replace('        <!-- RAG -->', '''        <div class="ins-section">
          <div class="ins-section-title">MCP Activity</div>
          <div class="ins-kv"><span>Used</span>${bool(mcp.used)}</div>
          <ul class="ins-tool-list">${(mcp.calls || []).map(c => `
            <li class="ins-tool-item">
              <span>${c.status === "success" ? "✓" : "✕"}</span>
              <span>${escapeHtml(c.tool || "")}</span>
              <span>${escapeHtml(c.server || "")}</span>
              <span>${escapeHtml(c.status || "")}</span>
              <span>${ms(c.latency_ms)}</span>
            </li>`).join("")}</ul>
        </div>
        <div class="ins-section">
          <div class="ins-section-title">Web Search</div>
          <div class="ins-row ins-row-sm">
            <div class="ins-kv"><span>Used</span>${bool(web.used)}</div>
            <div class="ins-kv"><span>Provider</span><strong>${escapeHtml(web.provider || "—")}</strong></div>
            <div class="ins-kv"><span>Results</span><strong>${fmt(web.results)}</strong></div>
            <div class="ins-kv"><span>Latency</span><strong>${ms(web.latency_ms)}</strong></div>
          </div>
        </div>
        <!-- RAG -->''')
s = s.replace('val.passed !== false', 'val.passed === true')
p.write_text(s, encoding='utf-8')
